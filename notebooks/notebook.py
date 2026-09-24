import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")

with app.setup:
    from pathlib import Path

    import marimo as mo
    import polars as pl

    from bank_clients_ml.column_groups import (
        get_binary_identity_features_cols,
        get_credit_card_cols,
        get_saving_account_cols,
    )
    from bank_clients_ml.config import get_settings
    from bank_clients_ml.eda import (
        columns_with_zeros,
        count_row_matches,
        filter_columns_by_cardinality,
        filter_nonzero,
        inspect_dataframe,
        low_cardinality_value_counts,
        mins_in_range,
        print_describe,
    )
    from bank_clients_ml.feature_engineering import (
        aggregate_monthly_to_client,
        target_encode_columns,
        with_extra_transformations,
        with_transformations,
    )
    from bank_clients_ml.redundant_column_filter import (
        BinRange,
        BinTransformation,
        RedundantColumnFilter,
    )
    from bank_clients_ml.sampling import (
        get_date_windows,
        stratified_train_test_split,
    )
    from bank_clients_ml.training import (
        GroupsLGBMTrainer,
        LGBMTrainer,
    )

    settings = get_settings()
    notebook_dir = Path(__file__).parent
    raw_data = pl.read_parquet(notebook_dir.parent / "data" / "data.parquet")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # EDA (Exploratory Data Analysis)
    """)
    return


@app.cell
def _():
    print("raw_data.shape:", raw_data.shape)
    print_describe(raw_data)

    less_than_zero_columns = [
        "SavingAccount_Balance_Average",
        "CreditCard_Balance_ARG",
        "CreditCard_Balance_DOLLAR",
        "CreditCard_Total_Spending",
        "CreditCard_Spending_1_Installment",
        "CreditCard_Spending_Aut_Debits",
        "CreditCard_Revolving",
    ]
    greater_than_thirty_one_columns = [
        "SavingAccount_Days_with_use",
        "SavingAccount_Days_with_Debits",
    ]
    mo.output.append(
        count_row_matches(
            raw_data, columns=less_than_zero_columns, threshold=0, condition="<"
        )
    )
    mo.output.append(
        count_row_matches(
            raw_data,
            columns=greater_than_thirty_one_columns,
            threshold=31,
            condition=">",
        )
    )
    mo.output.append(low_cardinality_value_counts(raw_data))
    mo.output.append(
        filter_columns_by_cardinality(raw_data, condition=">", threshold=10)
    )
    mo.output.append(inspect_dataframe(raw_data))
    mo.output.append(raw_data.null_count().transpose(include_header=True))
    mo.output.append(
        raw_data.filter(pl.col(settings.col_target).is_null()).transpose(
            include_header=True
        )
    )
    mo.output.append(
        raw_data.filter(pl.col(settings.col_target).is_null()).select(settings.col_id)
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Limpieza inicial
    """)
    return


@app.cell
def _():
    print("raw_data.shape:", raw_data.shape)
    clean_data = raw_data.filter(pl.col(settings.col_target).is_not_null())
    mo.output.append(inspect_dataframe(clean_data))
    clean_data = clean_data.with_columns(
        pl.col("Month", "First_product_dt", "Last_product_dt").str.to_date(),
        pl.col(settings.col_id).cast(pl.Int64),
    )
    return (clean_data,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Obtener meses relevantes
    """)
    return


@app.cell
def _(clean_data):
    training_months, prediction_months = get_date_windows(
        clean_data, "Month", prediction_window_size=2
    )

    last_training_month = training_months[-1]
    first_prediction_month = prediction_months[0]

    print("last_training_month:", last_training_month)
    print("first_prediction_month:", first_prediction_month)
    mo.output.append(training_months)
    mo.output.append(prediction_months)
    return (
        first_prediction_month,
        last_training_month,
        prediction_months,
        training_months,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Definir Universo y Target
    """)
    return


@app.cell
def _(clean_data):
    month_count_by_client = clean_data.group_by(settings.col_id).len(name="month_count")
    mo.output.append(month_count_by_client["month_count"].value_counts(sort=True))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Mantengo en el universo los clientes que:
    - tienen 9 meses de historia
    - no tienen 'Package_Active' y 'CreditCard_CoBranding' en el ultimo mes de la ventana de entrenamiento

    y para cada cliente del universo mantengo la columna de target de la ventana de predicción
    """)
    return


@app.cell
def _(clean_data, last_training_month, prediction_months, training_months):
    is_valid_client = (
        (pl.col("Month") == last_training_month)
        & (pl.col("Package_Active") == "No")
        & (pl.col("CreditCard_CoBranding") == "No")
    )

    universe_and_target = (
        clean_data.filter(
            (pl.len().over(settings.col_id) == 9)
            & is_valid_client.any().over(settings.col_id)
            & pl.col("Month").is_in(prediction_months)
        )
        .select(settings.col_id, settings.col_target)
        .unique()
    )

    universe_and_target_data = clean_data.drop(settings.col_target).join(
        universe_and_target, on=settings.col_id, how="inner"
    )

    training_data = universe_and_target_data.filter(
        pl.col("Month").is_in(training_months)
    )
    prediction_data = universe_and_target_data.filter(
        pl.col("Month").is_in(prediction_months)
    )

    print(f"universe_and_target.shape: {universe_and_target.shape} \n")
    print(f"prediction_data.shape: {prediction_data.shape} \n")
    mo.output.append(mo.md("### training_data['Month'].value_counts():"))
    mo.output.append(training_data["Month"].value_counts())
    inspect_dataframe(training_data)
    return prediction_data, training_data, universe_and_target_data


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Feature Engineering
    ## Valores Nulos
    """)
    return


@app.cell
def _(training_data):
    null_cols = ["SavingAccount_Balance_Average", "Region", "CreditCard_Product"]
    print(
        f"SavingAccount_Balance_Average unique values: "
        f"{training_data.select('SavingAccount_Balance_Average').n_unique()} \n"
    )
    print_describe(training_data.select(null_cols))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Completando 'SavingAccount_Balance_Average'
    Primero se analizan registros con nulos en SavingAccount_Balance_Average y valores monetarios de SavingAccount sin nulos:
    """)
    return


@app.cell
def _(training_data):
    saving_account_cols = get_saving_account_cols()
    mo.output.append(
        training_data.select([settings.col_id, *saving_account_cols]).filter(
            pl.col("SavingAccount_Balance_Average").is_null()
        )
    )
    mo.output.append(filter_nonzero(training_data, saving_account_cols))
    return (saving_account_cols,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Luego saco el promedio entre "SavingAccount_Balance_FirstDate" y "SavingAccount_Balance_LastDate".
    Este no es el calculo correcto para "SavingAccount_Balance_Average", pero no va a afectar tanto al modelo porque son solo 4 registros con nulos ademas de que no hay una forma sencilla de calcular el "SavingAccount_Balance_Average" con los datos que tenemos
    """)
    return


@app.cell
def _(training_data):
    training_data_1 = training_data.with_columns(
        pl.col("SavingAccount_Balance_Average").fill_null(
            (
                pl.col("SavingAccount_Balance_FirstDate")
                + pl.col("SavingAccount_Balance_LastDate")
            )
            / 2.0
        )
    )
    print_describe(training_data_1.select("SavingAccount_Balance_Average"))
    mo.output.append(inspect_dataframe(training_data_1))
    return (training_data_1,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Completando 'Region'
    Traigo las regiones de los clientes desde la ventana de predicción y pongo la Region mas común para llenar los nulos restantes
    """)
    return


@app.cell
def _(prediction_data, training_data_1, universe_and_target_data):
    clients_region = prediction_data.select(settings.col_id, "Region").unique()

    training_data_2 = training_data_1.drop("Region").join(
        clients_region.with_columns(pl.col("Region").fill_null("BUENOS AIRES")),
        on=settings.col_id,
        how="left",
    )

    print(f"{clients_region.shape} \n")
    print(f"{training_data_2.shape} \n")
    mo.output.append(clients_region["Region"].value_counts(sort=True))
    mo.output.append(training_data_2["Region"].value_counts(sort=True))
    print_describe(universe_and_target_data.select("Region"))
    print_describe(clients_region)
    print_describe(training_data_2.select("Region"))
    return (training_data_2,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Completando 'CreditCard_Product'

    Traigo los CreditCard_Product de la ventana de predicción.

    Hay algunos clientes que tienen un CreditCard_Product en el primer mes de predicción y otro CreditCard_Product en el segundo mes de predicción

    Por lo tanto se obtiene el valor del primer mes de la ventana de predicción y, si este es null o no existe, toma el valor del segundo mes como fallback, incluso si también es null

    Luego para llenar los nulos restantes, pongo la CreditCard_Product mas común cuando el cliente no tiene `CreditCard_Active` en la ventana de predicción pero si tiene `CreditCard_Active` en la ventana de entrenamiento. en los demás casos lleno los nulls con "0" (cuando no tiene `CreditCard_Active` en la ventana de predicción ni en la ventana de entrenamiento o cuando no tiene `CreditCard_Active` en la ventana de entrenamiento, por mas que lo tenga en la ventana de predicción)
    """)
    return


@app.cell
def _(
    first_prediction_month,
    prediction_data,
    training_data_2,
    universe_and_target_data,
):
    clients_credit_card_product = (
        prediction_data.sort(pl.col("Month") == first_prediction_month, descending=True)
        .group_by(settings.col_id)
        .agg(pl.col("CreditCard_Product").drop_nulls().first())
    )
    training_data_3 = (
        training_data_2.drop("CreditCard_Product")
        .join(clients_credit_card_product, on=settings.col_id, how="left")
        .with_columns(
            pl.when(
                pl.col("CreditCard_Product").is_null()
                & (pl.col("CreditCard_Active") == "Yes")
            )
            .then(pl.col("CreditCard_Product").fill_null("J55660104XX012"))
            .otherwise(pl.col("CreditCard_Product").fill_null("0"))
            .alias("CreditCard_Product")
        )
    )
    mo.output.append(
        clients_credit_card_product["CreditCard_Product"].value_counts(sort=True)
    )
    mo.output.append(training_data_3["CreditCard_Product"].value_counts(sort=True))
    print_describe(universe_and_target_data.select("CreditCard_Product"))
    print_describe(clients_credit_card_product)
    print_describe(training_data_3.select("CreditCard_Product"))
    inspect_dataframe(training_data_3)
    return (training_data_3,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Identity Features
    """)
    return


@app.cell
def _(last_training_month, training_data_3):
    binary_dict: dict[str, int] = {"Yes": 1, "No": 0, "M": 1, "F": 0}

    binary_identity_features_columns = get_binary_identity_features_cols()
    categorical_cols = ["Client_Age_grp", "Region", "CreditCard_Product"]
    identity_features_columns = [
        settings.col_id,
        settings.col_target,
        *categorical_cols,
        "First_product_dt",
        "Last_product_dt",
        *binary_identity_features_columns,
    ]

    training_data_4 = training_data_3.with_columns(
        pl.col(binary_identity_features_columns)
        .replace_strict(binary_dict)
        .cast(pl.UInt8)
    )

    identity_features = training_data_4.filter(
        pl.col("Month") == last_training_month
    ).select(identity_features_columns)

    mo.output.append(
        low_cardinality_value_counts(identity_features.drop(categorical_cols))
    )
    mo.output.append(inspect_dataframe(identity_features))
    print_describe(identity_features)
    return categorical_cols, identity_features, training_data_4


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Variables Categóricas
    """)
    return


@app.cell
def _(categorical_cols, identity_features, training_data_4):
    mo.output.append(
        low_cardinality_value_counts(identity_features.select(categorical_cols))
    )

    identity_features_1 = target_encode_columns(identity_features, categorical_cols)

    mo.output.append(
        low_cardinality_value_counts(identity_features_1.select(categorical_cols))
    )

    mo.output.append(f"training_data before drop: {training_data_4.shape}")
    training_data_5 = training_data_4.drop(categorical_cols)
    mo.output.append(inspect_dataframe(training_data_5))
    return identity_features_1, training_data_5


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Fechas
    """)
    return


@app.cell
def _(identity_features_1, last_training_month):
    identity_features_2 = identity_features_1.with_columns(
        [
            (pl.col("Last_product_dt") - pl.col("First_product_dt"))
            .dt.total_days()
            .alias("Days_between_first_and_last_product"),
            (
                pl.lit(last_training_month).dt.offset_by("1mo")
                - pl.col("Last_product_dt")
            )
            .dt.total_days()
            .alias("Recency_in_days"),
        ]
    ).drop(["First_product_dt", "Last_product_dt"])

    mo.output.append(inspect_dataframe(identity_features_2))
    print_describe(identity_features_2)
    return (identity_features_2,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Transform features
    Primero analizo:
    - valores mínimos y ceros
    - valores monetarios de CreditCard
    """)
    return


@app.cell
def _(training_data_5):
    mo.output.append(mins_in_range(training_data_5, -1, 1))
    mo.output.append(columns_with_zeros(training_data_5))
    credit_card_cols = get_credit_card_cols()
    mo.output.append(filter_nonzero(training_data_5, credit_card_cols))
    return (credit_card_cols,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Generando transformaciones
    """)
    return


@app.cell
def _(training_data_5):
    print(training_data_5.shape)
    training_data_6 = with_transformations(training_data_5)

    mo.output.append(inspect_dataframe(training_data_6))
    print_describe(training_data_6)

    mo.output.append(
        training_data_6.select(
            [
                settings.col_id,
                "SavingAccount_Transactions_Transactions",
                "Operations_total",
                "CreditCard_Payment_total",
            ]
        ).filter(
            (pl.col("SavingAccount_Transactions_Transactions") != 0)
            & (pl.col("Operations_total") != 0)
            & (pl.col("CreditCard_Payment_total") != 0)
        )
    )

    greater_than_one_hundred_columns = [
        "SavingAccount_Transfer_In_Amount_pct",
        "SavingAccount_Transfer_In_Amount_CR_pct",
        "SavingAccount_Balance_last_minus_first_date_pct",
    ]
    count_row_matches(
        training_data_6,
        columns=greater_than_one_hundred_columns,
        threshold=100,
        condition=">",
    )
    return (training_data_6,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Aggregate Features
    """)
    return


@app.cell
def _(
    credit_card_cols,
    identity_features_2,
    saving_account_cols,
    training_data_6,
):
    data_agg = aggregate_monthly_to_client(
        saving_account_cols, credit_card_cols, training_data_6, identity_features_2
    )
    inspect_dataframe(data_agg)
    return (data_agg,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # ABT (analytical base table) and Train/Test split
    - Agrego transformadas extras luego de generar la ABT
    - Divido los datos es train y test
    """)
    return


@app.cell
def _(data_agg, identity_features_2):
    abt = identity_features_2.join(data_agg, on=settings.col_id, how="inner")
    mo.output.append(inspect_dataframe(abt))

    abt = with_extra_transformations(abt)
    mo.output.append(mins_in_range(abt, -1, 1))
    mo.output.append(inspect_dataframe(abt))

    train, test = stratified_train_test_split(abt)
    mo.output.append(mo.md(f".\n\n train.shape: {train.shape}"))
    print_describe(train)
    return test, train


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Feature Selection
    ## Reducción de dimensionalidad
    Elimino:
    - columnas con valores únicos
    - columnas binarias con baja representatividad
    - columnas correlacionadas entre si
    """)
    return


@app.cell
def _(test, train):
    column_filter = RedundantColumnFilter(
        train, test, imbalanced_binary_threshold=0.10, correlation_threshold=0.80
    )
    column_filter.print_constant_cols()
    column_filter.print_imbalanced_binary_columns()
    uncorrelated_train, _ = column_filter.get_uncorrelated()
    mo.output.append(
        mo.md(f".\n\n uncorrelated_train.shape: {uncorrelated_train.shape}")
    )
    return column_filter, uncorrelated_train


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Ordeno las variables por fuente según importancia usando lightGBM para quedarme con las mas importantes

    No estandarizo el dataframe por que lightGBM no lo necesita

    - primero entreno con todas las variables, para tener roc de referencia:
    - luego entreno con cada grupo por separado
    - luego entreno con los mejores de cada grupo al mismo tiempo
    """)
    return


@app.cell
def _(uncorrelated_train):
    groups_trainer = GroupsLGBMTrainer(uncorrelated_train)
    groups_trainer.print_search_logs()
    groups_trainer.print_groups_lengths()
    groups_trainer.print_searchers()
    groups_trainer.plot_top_features()
    return (groups_trainer,)


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    {mo.image(src=notebook_dir / "images" / "plot_top_features" / "all_columns.svg")}
    {mo.image(src=notebook_dir / "images" / "plot_top_features" / "saving_account_days_transactions.svg")}
    {mo.image(src=notebook_dir / "images" / "plot_top_features" / "saving_account_monetary.svg")}
    {mo.image(src=notebook_dir / "images" / "plot_top_features" / "operations.svg")}
    {mo.image(src=notebook_dir / "images" / "plot_top_features" / "credit_card_payment.svg")}
    {mo.image(src=notebook_dir / "images" / "plot_top_features" / "credit_card_monetary.svg")}
    {mo.image(src=notebook_dir / "images" / "plot_top_features" / "others.svg")}
    """)
    return


@app.cell
def _(groups_trainer, uncorrelated_train):
    top_grouped_features = groups_trainer.get_top_grouped_features()
    trainer = LGBMTrainer(uncorrelated_train, top_grouped_features)
    trainer.print_search_logs()
    trainer.print_searcher()
    trainer.plot_top_features("top_grouped_features")
    return top_grouped_features, trainer


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    {mo.image(src=notebook_dir / "images" / "plot_top_features" / "top_grouped_features.svg")}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Análisis Bivariado
    """)
    return


@app.cell
def _(column_filter, top_grouped_features):
    column_filter.plot_uncorrelated(top_grouped_features, "uncorrelated")
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    {mo.image(src=notebook_dir / "images" / "bivariate_analysis" / "uncorrelated" / "Client_Age_grp.svg")}
    {mo.image(src=notebook_dir / "images" / "bivariate_analysis" / "uncorrelated" / "CreditCard_Balance_ARG_SP_pct_max.svg")}
    {mo.image(src=notebook_dir / "images" / "bivariate_analysis" / "uncorrelated" / "CreditCard_Payment_total_max.svg")}
    {mo.image(src=notebook_dir / "images" / "bivariate_analysis" / "uncorrelated" / "CreditCard_Product.svg")}
    {mo.image(src=notebook_dir / "images" / "bivariate_analysis" / "uncorrelated" / "CreditCard_Total_Limit_diff_rel.svg")}
    {mo.image(src=notebook_dir / "images" / "bivariate_analysis" / "uncorrelated" / "Operations_total_min.svg")}
    {mo.image(src=notebook_dir / "images" / "bivariate_analysis" / "uncorrelated" / "Quantity_Active_Products_min.svg")}
    {mo.image(src=notebook_dir / "images" / "bivariate_analysis" / "uncorrelated" / "SavingAccount_Transfer_In_Amount_max.svg")}
    {mo.image(src=notebook_dir / "images" / "bivariate_analysis" / "uncorrelated" / "SavingAccount_Transfer_In_Transactions_pct_max.svg")}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Buscando variables correlacionadas eliminadas anteriormente
    Para poder intercambiar las variables mas importantes por variables mas fáciles de interpretar, si es que existen.
    """)
    return


@app.cell
def _(column_filter, trainer):
    top_ranked_features = trainer.get_top_ranked_features()
    column_filter.print_correlations_for_each(top_ranked_features)
    return


@app.cell
def _(column_filter):
    correlated_features = [
        "Operations_total_mean",
        "Operations_total_median",
        "CreditCard_Active",
        "CreditCard_Balance_ARG_SP_pct_mean",
        "Quantity_Active_Products_median",
        "Quantity_Active_Products_mean",
    ]
    column_filter.plot_correlated(correlated_features, "correlated")
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    {mo.image(src=notebook_dir / "images" / "bivariate_analysis" / "correlated" / "Operations_total_mean.svg")}
    {mo.image(src=notebook_dir / "images" / "bivariate_analysis" / "correlated" / "Operations_total_median.svg")}
    {mo.image(src=notebook_dir / "images" / "bivariate_analysis" / "correlated" / "CreditCard_Active.svg")}
    {mo.image(src=notebook_dir / "images" / "bivariate_analysis" / "correlated" / "CreditCard_Balance_ARG_SP_pct_mean.svg")}
    {mo.image(src=notebook_dir / "images" / "bivariate_analysis" / "correlated" / "Quantity_Active_Products_median.svg")}
    {mo.image(src=notebook_dir / "images" / "bivariate_analysis" / "correlated" / "Quantity_Active_Products_mean.svg")}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Transformando mejores variables según análisis bivariado y LightGBM

    Transformo variables agregándoles el porcentaje de target y agrupo los valores.
    A las variables categóricas solo las agrupo (ya las transforme anteriormente)

    variables a modificar:
    - Client_Age_grp
        - agrupo "Entre 50 y 59 años" + "Entre 60 y 64 años" + "Entre 65 y 69 años" (final: "Entre 50 y 69 años")
        - default -> junto todas las demás edades ("Entre 18 y 29 años" + "Entre 30 y 39 años" + "Entre 40 y 49 años" + "Mayor a 70 años")
    - Operations_total_mean
    - Operations_total_median
    - CreditCard_Product
        - mantengo tipo tarjeta 202 y 104 separados
        - default ->  junto los demás tipos de tarjetas de bajo porcentaje de target y los tipos de tarjetas poco representativas en un solo bin (sin tarjeta de crédito + 102 + 123 + 124 + 702 + 1002)
    - CreditCard_Active (no hace falta transformar)
    - Quantity_Active_Products_min
    """)
    return


@app.cell
def _(column_filter):
    bins_transformations = [
        BinTransformation("Client_Age_grp", [BinRange(4, 5), BinRange(6, 7)]),
        BinTransformation(
            "Operations_total_mean",
            [
                BinRange(2, 4),
                BinRange(5, 6),
                BinRange(7, 8),
                BinRange(9, 10),
                BinRange(11, 12),
                BinRange(13, 14),
                BinRange(15, 16),
            ],
        ),
        BinTransformation(
            "Operations_total_median",
            [BinRange(2, 4), BinRange(5, 7), BinRange(8, 9), BinRange(10, 11)],
        ),
        BinTransformation("CreditCard_Product", [BinRange(5, 5), BinRange(7, 7)]),
        BinTransformation(
            "Quantity_Active_Products_min", [BinRange(1, 4), BinRange(6, 9)]
        ),
    ]
    best_features = [
        "Client_Age_grp",
        "Operations_total_mean",
        # "Operations_total_median",
        "CreditCard_Product",
        # "CreditCard_Active",  # unmodified
        "Quantity_Active_Products_min",
    ]
    final_train, final_test = column_filter.apply_bin_transformations(
        bins_transformations
    )
    mo.output.append(inspect_dataframe(final_train))

    final_cols = [settings.col_id, settings.col_target, *best_features]
    final_train = final_train.select(final_cols)
    final_test = final_test.select(final_cols)
    mo.output.append(inspect_dataframe(final_train))
    print_describe(final_train)
    return best_features, final_test, final_train


@app.cell
def _(best_features, column_filter, final_train):
    column_filter.plot_adhoc(final_train, best_features, "best_features")
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    {mo.image(src=notebook_dir / "images" / "bivariate_analysis" / "best_features" / "CreditCard_Product.svg")}
    {mo.image(src=notebook_dir / "images" / "bivariate_analysis" / "best_features" / "Client_Age_grp.svg")}
    {mo.image(src=notebook_dir / "images" / "bivariate_analysis" / "best_features" / "Operations_total_mean.svg")}
    {mo.image(src=notebook_dir / "images" / "bivariate_analysis" / "best_features" / "Quantity_Active_Products_min.svg")}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Final Training and Performance
    ## Entreno con las mejores features y mejores hiperparámetros
    No realizo ningún balanceo porque la proporción del target ya es del 30%
    """)
    return


@app.cell
def _(best_features, final_test, final_train):
    renames_dict = {
        "Client_Age_grp": "Age range",
        "Operations_total_mean": "Average quantity of operations",
        "CreditCard_Product": "Credit Card Type",
        "Quantity_Active_Products_min": "Minimum quantity of active products",
    }
    final_trainer = LGBMTrainer(final_train, best_features, n_iter=20, test=final_test)
    final_trainer.print_search_logs()
    final_trainer.print_searcher()
    final_trainer.plot_top_features(plot_name="best_features", renames=renames_dict)
    final_trainer.plot_evaluation_metrics(plot_name="lightgbm")
    final_trainer.plot_deciles(plot_name="deciles")
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    {mo.image(src=notebook_dir / "images" / "plot_top_features" / "best_features.svg")}

    ## Metrics results
    {mo.image(src=notebook_dir / "images" / "plot_evaluation_metrics" / "lightgbm.svg")}
    {mo.image(src=notebook_dir / "images" / "plot_evaluation_metrics" / "deciles.svg")}

    ### Training
    - ordena todos los deciles bien
    - deciles más o menos parejos
    - lift del primer decil = 2,4
    - KS = 47.04 en el 5to decil

    ### Testing
    - ordena todos los deciles bien
    - deciles más o menos parejos
    - lift del primer decil = 2,36
    - KS = 47.22 en el 4to decil

    ### Diferencias
    - lift -> 0,4
    - ~~KS -> 0,2~~
    """)
    return


if __name__ == "__main__":
    app.run()
