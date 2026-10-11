import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import polars as pl

    from bank_clients_ml import cleaning, column_groups, config, eda, sampling, training
    from bank_clients_ml import feature_engineering as feat
    from bank_clients_ml import redundant_column_filter as col_filter

    settings = config.get_settings()


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 1. Exploratory Data Analysis (EDA)

    Se analiza:
    - Valores atípicos de variables de montos y de días
    - Cardinalidad de todas las variables.
    - (En la siguiente celda) Se buscan valores nulos
    """)
    return


@app.cell
def _():
    raw_data = pl.read_parquet(settings.paths.data / "data.parquet")

    _less_than_zero_columns = [
        "SavingAccount_Balance_Average",
        "CreditCard_Balance_ARG",
        "CreditCard_Balance_DOLLAR",
        "CreditCard_Total_Spending",
        "CreditCard_Spending_1_Installment",
        "CreditCard_Spending_Aut_Debits",
        "CreditCard_Revolving",
    ]
    _greater_than_thirty_one_columns = [
        "SavingAccount_Days_with_use",
        "SavingAccount_Days_with_Debits",
    ]
    print("raw_data.shape:", raw_data.shape)
    mo.ui.tabs(
        {
            "Describe raw_data": eda.describe(raw_data),
            "Monetary columns less than 0": eda.count_row_matches(
                raw_data, columns=_less_than_zero_columns, threshold=0, condition="<"
            ),
            "Days columns greater than 31": eda.count_row_matches(
                raw_data,
                columns=_greater_than_thirty_one_columns,
                threshold=31,
                condition=">",
            ),
            "Low cardinality columns": eda.low_cardinality_value_counts(raw_data),
            "High cardinality columns": eda.filter_columns_by_cardinality(
                raw_data, condition=">", threshold=10
            ),
        }
    )
    return (raw_data,)


@app.cell
def _(raw_data):
    mo.ui.tabs(
        {
            "Inspect raw_data": eda.inspect_dataframe(raw_data),
            "raw_data.null_count()": raw_data.null_count().transpose(
                include_header=True
            ),
            "Null row": raw_data.filter(
                pl.col(settings.dataset.target).is_null()
            ).transpose(include_header=True),
            "Null row id": raw_data.filter(
                pl.col(settings.dataset.target).is_null()
            ).select(settings.dataset.id),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 1.1 Initial Data Cleaning

    Se elimina el registro identificado en la celda anterior, el cual solo tenia valores nulos. También se modifican los tipos de las variables de fechas y la variable id para evitar que se propaguen errores de tipo.
    """)
    return


@app.cell
def _(raw_data):
    print("raw_data.shape:", raw_data.shape)
    clean_data = cleaning.clean_up_data(raw_data)
    del raw_data
    mo.ui.tabs(
        {
            "Inspect clean_data": eda.inspect_dataframe(clean_data),
            "Number of clients by number of months": eda.get_clients_by_months(
                clean_data
            ),
        }
    )
    return (clean_data,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 2. Universe and Target Definition

    Se obtienen los meses de las ventanas de entrenamiento y predicción, con una lead window de 1 mes.

    ## 2.1 Universe Filtering Criteria

    Se conservan en el universo los clientes que:
    - Tienen 9 meses de historia
    - No tienen en el último mes de la ventana de entrenamiento:
        - `Package_Active`, para que el modelo aprenda a predecir la primera compra y no la renovación o el uso ya existente.
        - `CreditCard_CoBranding`, por regla de negocio ya se sabe que estos clientes no suelen comprar, esta inversamente correlacionada con el target.
    """)
    return


@app.cell
def _(clean_data):
    time_window_split = sampling.define_time_window_split(
        clean_data, prediction_window_size=2
    )
    del clean_data
    print(
        f"universe_and_target.shape: {time_window_split.universe_and_target.shape} \n"
    )
    print(f"prediction_data.shape: {time_window_split.prediction_data.shape} \n")
    mo.ui.tabs(
        {
            "Time Windows Dates": mo.vstack(
                [
                    mo.md("#### last_training_month:"),
                    time_window_split.training_months[-1],
                    mo.md("#### first_prediction_month:"),
                    time_window_split.prediction_months[0],
                    mo.md("#### training_months:"),
                    time_window_split.training_months,
                    mo.md("#### prediction_months:"),
                    time_window_split.prediction_months,
                ]
            ),
            "Number of clients per Month": time_window_split.training_data.get_column(
                settings.dataset.month
            ).value_counts(),
            "Inspect training_data": eda.inspect_dataframe(
                time_window_split.training_data
            ),
        }
    )
    return (time_window_split,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 3. Feature Engineering
    ## 3.1 Missing Value Imputation
    """)
    return


@app.cell
def _(time_window_split):
    _null_cols = ["SavingAccount_Balance_Average", "Region", "CreditCard_Product"]
    print(
        f"SavingAccount_Balance_Average unique values: "
        f"{time_window_split.training_data.select('SavingAccount_Balance_Average').n_unique()} \n"
    )
    mo.output.append(eda.describe(time_window_split.training_data.select(_null_cols)))
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 3.1.1 Saving Account Balance (Average)

    Primero se analizan registros con nulos en `SavingAccount_Balance_Average` y valores monetarios de `SavingAccount` sin nulos.
    """)
    return


@app.cell
def _(time_window_split):
    mo.ui.tabs(
        {
            'When "Saving Account Balance (Average)" is Null': time_window_split.training_data.select(
                [settings.dataset.id, *column_groups.get_saving_account_cols()]
            ).filter(pl.col("SavingAccount_Balance_Average").is_null()),
            "Non-null monetary values of Saving Account": eda.filter_nonzero(
                time_window_split.training_data, column_groups.get_saving_account_cols()
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Ante la imposibilidad de recalcular el promedio diario exacto con los datos disponibles, se aproxima como el promedio entre el saldo inicial y final (`SavingAccount_Balance_FirstDate` y `SavingAccount_Balance_LastDate`). Son solo cuatro registros, por lo que el impacto en el modelo es mínimo.
    """)
    return


@app.cell
def _(time_window_split):
    training_data_1 = time_window_split.training_data.with_columns(
        pl.col("SavingAccount_Balance_Average").fill_null(
            (
                pl.col("SavingAccount_Balance_FirstDate")
                + pl.col("SavingAccount_Balance_LastDate")
            )
            / 2.0
        )
    )
    mo.ui.tabs(
        {
            'Describe "Saving Account Balance (Average)"': eda.describe(
                training_data_1.select("SavingAccount_Balance_Average")
            ),
            "Inspect training_data": eda.inspect_dataframe(training_data_1),
        }
    )
    return (training_data_1,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 3.1.2 Region

    `Region` es una variable estable del cliente, por lo que un nulo en entrenamiento suele deberse a un vacío de carga y no a un cambio real. Por lo tanto, se obtienen estos datos desde la ventana de predicción, y para los nulos restantes, se los llena con la Region mas común (`BUENOS AIRES`).
    """)
    return


@app.cell
def _(time_window_split, training_data_1):
    _clients_region = time_window_split.prediction_data.select(
        settings.dataset.id, "Region"
    ).unique()

    training_data_2 = training_data_1.drop("Region").join(
        _clients_region.with_columns(pl.col("Region").fill_null("BUENOS AIRES")),
        on=settings.dataset.id,
        how="left",
    )
    del training_data_1

    print(f"clients_region.shape: {_clients_region.shape} \n")
    print(f"training_data_2.shape: {training_data_2.shape} \n")
    mo.ui.tabs(
        {
            "Number of clients: clients_region": _clients_region.get_column(
                "Region"
            ).value_counts(sort=True),
            "Number of clients: training_data": training_data_2.get_column(
                "Region"
            ).value_counts(sort=True),
            "Describe DataFrames": mo.vstack(
                [
                    mo.md("### Describe universe_and_target_data"),
                    eda.describe(
                        time_window_split.universe_and_target_data.select("Region")
                    ),
                    mo.md("### Describe clients_region"),
                    eda.describe(_clients_region),
                    mo.md("### Describe training_data"),
                    eda.describe(training_data_2.select("Region")),
                ]
            ),
        }
    )
    return (training_data_2,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### 3.1.3 CreditCard Product

    Se traen los tipos de tarjeta de crédito (`CreditCard_Product`) de la ventana de predicción.

    Hay algunos clientes que tienen un tipo de tarjeta en el primer mes de predicción y otro tipo de tarjeta en el segundo mes de predicción. Por lo tanto se obtiene el valor del primer mes de la ventana de predicción y, si este es null o no existe, se toma el valor del segundo mes como fallback, incluso si también es null.

    Luego para llenar los nulos restantes, se usa el tipo de tarjeta (`CreditCard_Product`) mas común cuando el cliente no tiene tarjeta activa (`CreditCard_Active`) en la ventana de predicción pero si tiene tarjeta activa en la ventana de entrenamiento. En los demás casos se llenan los nulls con "0" (cuando no tiene tarjeta activa en la ventana de predicción ni en la ventana de entrenamiento o cuando no tiene tarjeta activa en la ventana de entrenamiento, por mas que lo tenga en la ventana de predicción).
    """)
    return


@app.cell
def _(time_window_split, training_data_2):
    _clients_credit_card_product = (
        time_window_split.prediction_data.sort(
            pl.col(settings.dataset.month) == time_window_split.prediction_months[0],
            descending=True,
        )
        .group_by(settings.dataset.id)
        .agg(pl.col("CreditCard_Product").drop_nulls().first())
    )
    training_data_3 = (
        training_data_2.drop("CreditCard_Product")
        .join(_clients_credit_card_product, on=settings.dataset.id, how="left")
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
    del training_data_2
    mo.ui.tabs(
        {
            "Credit Card Type (clients in prediction window)": _clients_credit_card_product.get_column(
                "CreditCard_Product"
            ).value_counts(sort=True),
            "Credit Card Type (training_data)": training_data_3.get_column(
                "CreditCard_Product"
            ).value_counts(sort=True),
            "Inspect training_data": eda.inspect_dataframe(training_data_3),
            "Describe DataFrames": mo.vstack(
                [
                    mo.md("### Describe Credit Card Type (universe_and_target_data)"),
                    eda.describe(
                        time_window_split.universe_and_target_data.select(
                            "CreditCard_Product"
                        )
                    ),
                    mo.md("### Describe clients in prediction window"),
                    eda.describe(_clients_credit_card_product),
                    mo.md("### Describe Credit Card Type (training_data)"),
                    eda.describe(training_data_3.select("CreditCard_Product")),
                ]
            ),
        }
    )
    return (training_data_3,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3.2 Identity Features

    Las Identity Features se toman del último mes de la ventana de entrenamiento. Ademas, las variables con valores `Yes/No` y `M/F` se pasan a valores binarios (0 ó 1).
    """)
    return


@app.cell
def _(time_window_split, training_data_3):
    _binary_dict: dict[str, int] = {"Yes": 1, "No": 0, "M": 1, "F": 0}

    training_data_4 = training_data_3.with_columns(
        pl.col(column_groups.get_binary_identity_features_cols())
        .replace_strict(_binary_dict)
        .cast(pl.UInt8)
    )

    identity_features = training_data_4.filter(
        pl.col(settings.dataset.month) == time_window_split.training_months[-1]
    ).select(column_groups.get_identity_features_cols())

    del training_data_3
    mo.ui.tabs(
        {
            "Low cardinality columns of identity_features": eda.low_cardinality_value_counts(
                identity_features.drop(column_groups.get_categorical_cols())
            ),
            "Inspect identity_features": eda.inspect_dataframe(identity_features),
            "Describe identity_features": eda.describe(identity_features),
        }
    )
    return identity_features, training_data_4


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3.3 Categorical Features Encoding

    Las variables `Client_Age_grp`, `Region` y `CreditCard_Product` se codifican con su porcentaje respecto al target, para capturar su relación con el target sin expandir la dimensionalidad con técnicas como "one hot encoding".
    """)
    return


@app.cell
def _(identity_features, training_data_4):
    _categorical_cols = column_groups.get_categorical_cols()
    _temp_value_counts = eda.low_cardinality_value_counts(
        identity_features.select(_categorical_cols)
    )
    identity_features_1 = feat.target_encode_columns(
        identity_features, _categorical_cols
    )
    training_data_5 = training_data_4.drop(_categorical_cols)
    del identity_features
    mo.ui.tabs(
        {
            "Categorical columns before encode": _temp_value_counts,
            "Categorical columns after encode": eda.low_cardinality_value_counts(
                identity_features_1.select(_categorical_cols)
            ),
            "Inspect training_data before drop": eda.inspect_dataframe(training_data_4),
            "Inspect training_data after drop": eda.inspect_dataframe(training_data_5),
        }
    )
    return identity_features_1, training_data_5


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3.4 Date-Derived Features

    Las fechas absolutas se reemplazan por "diferencia entre primer y último producto" y por "recencia" desde el último producto, que expresan lo mismo en una escala relativa y estable en el tiempo.
    """)
    return


@app.cell
def _(identity_features_1, time_window_split, training_data_4):
    del training_data_4
    identity_features_2 = feat.transform_date_features(
        identity_features_1, time_window_split.training_months
    )
    del time_window_split
    del identity_features_1
    mo.ui.tabs(
        {
            "Inspect identity_features": eda.inspect_dataframe(identity_features_2),
            "Describe identity_features": eda.describe(identity_features_2),
        }
    )
    return (identity_features_2,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3.5 Feature Transformation
    Se analizan:
    - Valores mínimos y ceros
    - Valores monetarios de CreditCard
    """)
    return


@app.cell
def _(training_data_5):
    mo.ui.tabs(
        {
            "Columns with zeros: training_data": eda.columns_with_zeros(
                training_data_5
            ),
            "Rows with no zeros: training_data": eda.filter_nonzero(
                training_data_5, column_groups.get_credit_card_cols()
            ),
            "Mins in (-1, 1): training_data": eda.mins_in_range(training_data_5, -1, 1),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Se aplican transformaciones.
    """)
    return


@app.cell
def _(training_data_5):
    training_data_6 = feat.with_transformations(training_data_5)
    _greater_than_one_hundred_columns = [
        "SavingAccount_Transfer_In_Amount_pct",
        "SavingAccount_Transfer_In_Amount_CR_pct",
        "SavingAccount_Balance_last_minus_first_date_pct",
    ]

    print(f"training_data shape before transformations {training_data_5.shape}")
    del training_data_5
    mo.ui.tabs(
        {
            "Inspect training_data": eda.inspect_dataframe(training_data_6),
            "Describe training_data": eda.describe(training_data_6),
            "Some columns with non-zero rows": training_data_6.select(
                [
                    settings.dataset.id,
                    "SavingAccount_Transactions_Transactions",
                    "Operations_total",
                    "CreditCard_Payment_total",
                ]
            ).filter(
                (pl.col("SavingAccount_Transactions_Transactions") != 0)
                & (pl.col("Operations_total") != 0)
                & (pl.col("CreditCard_Payment_total") != 0)
            ),
            "Number of rows in percentage columns exceeding 100": eda.count_row_matches(
                training_data_6,
                columns=_greater_than_one_hundred_columns,
                threshold=100,
                condition=">",
            ),
        }
    )
    return (training_data_6,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3.6 Aggregate Features

    La historia de nueve meses debe convertirse en una tabla con un registro por cliente para el entrenamiento supervisado (ABT). En esta parte se realizan las agregaciones para poder conseguir esto.
    """)
    return


@app.cell
def _(identity_features_2, training_data_6):
    data_agg = feat.aggregate_monthly_to_client(
        column_groups.get_saving_account_cols(),
        column_groups.get_credit_card_cols(),
        training_data_6,
        identity_features_2,
    )
    del training_data_6
    eda.inspect_dataframe(data_agg)
    return (data_agg,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 4. Analytical Base Table (ABT) and Train-Test Split

    Luego de obtener la ABT se aplican algunas transformaciones adicionales. Luego se realiza una partición estratificada para obtener el set de "train" y el set de "test", preservando la tasa del target en ambos conjuntos.
    """)
    return


@app.cell
def _(data_agg, identity_features_2):
    _abt = identity_features_2.join(data_agg, on=settings.dataset.id, how="inner")
    _temp_abt_inspect = eda.inspect_dataframe(_abt)
    _abt = feat.with_extra_transformations(_abt)
    train, test = sampling.stratified_train_test_split(_abt)
    del identity_features_2
    del data_agg
    mo.ui.tabs(
        {
            "Inspect ABT before extra transformations": _temp_abt_inspect,
            "Inspect ABT after extra transformations": eda.inspect_dataframe(_abt),
            "Mins in (-1, 1): ABT": eda.mins_in_range(_abt, -1, 1),
            "Inspect train split": eda.inspect_dataframe(train),
            "Describe train split": eda.describe(train),
        }
    )
    return test, train


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 5. Feature Selection
    ## 5.1 Dimensionality Reduction

    Se eliminan columnas constantes, columnas binarias desbalanceadas y columnas altamente correlacionadas, para disminuir el costo computacional antes del ranking con LightGBM.
    """)
    return


@app.cell
def _(test, train):
    column_filter = col_filter.RedundantColumnFilter(
        train, test, imbalanced_binary_threshold=0.10, correlation_threshold=0.80
    )
    del train
    del test
    mo.ui.tabs(
        {
            "Constant columns removed": column_filter.constant_cols,
            "Inspect train without Constant columns": eda.inspect_dataframe(
                column_filter.reduced_train
            ),
            "Imbalanced binary columns removed (Value counts)": column_filter.get_imbalanced_binary_counts(),
            "Inspect train without Imbalanced binary columns": eda.inspect_dataframe(
                column_filter.correlated_train
            ),
            "Inspect train without correlated columns": eda.inspect_dataframe(
                column_filter.uncorrelated_train
            ),
        },
        orientation="vertical",
    )
    return (column_filter,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5.2 Group-Wise Importance Ranking
    Para poder ordenar las variables según su importancia, se agrupan las variables por fuente de negocio y se entrena un modelo LightGBM por grupo, además de un modelo con todas las variables. Esta comparación permite identificar qué origen aporta más. Luego se entrena un modelo con las mejores variables de todos los grupos.

    No se estandariza el dataframe porque los algoritmos basados en arboles como lightGBM no lo necesitan.
    """)
    return


@app.cell
def _(column_filter):
    groups_trainer = training.GroupsLGBMTrainer(column_filter.uncorrelated_train)
    groups_trainer.plot_top_features()
    groups_names = {
        "all_columns": "All Columns",
        "saving_account_days_transactions": "Savings Account (Days transactions)",
        "saving_account_monetary": "Savings Account (Amount)",
        "operations": "Operations",
        "credit_card_payment": "Credit Card (Payments)",
        "credit_card_monetary": "Credit Card (Amount)",
        "others": "Others",
    }
    mo.ui.tabs(
        {
            "Groups lengths": groups_trainer.get_groups_lengths_views(),
            '"Others" Columns': groups_trainer.trainers["others"].columns,
            "Groups Cross Validation Results": mo.accordion(
                groups_trainer.get_cv_results(include_logs=True)
            ),
        }
    )
    return groups_names, groups_trainer


@app.cell
def _(groups_names):
    eda.build_plot_tabs(
        groups_names,
        settings.paths.images / "plot_top_features",
        orientation="vertical",
    )
    return


@app.cell
def _(column_filter, groups_names, groups_trainer):
    del groups_names
    trainer = training.LGBMTrainer(
        column_filter.uncorrelated_train, groups_trainer.get_top_grouped_features()
    )
    trainer.plot_top_features("top_grouped_features")
    mo.ui.tabs(
        {
            "Top grouped features importances": mo.image(
                src=settings.paths.images
                / "plot_top_features"
                / "top_grouped_features.svg"
            ),
            "Cross Validation Results": trainer.get_cv_results(include_logs=True),
        }
    )
    return (trainer,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5.3 Bivariate Analysis

    Se realiza análisis bivariado sobre las variables mas relevantes.
    """)
    return


@app.cell
def _(column_filter, groups_trainer):
    column_filter.plot_uncorrelated(
        groups_trainer.get_top_grouped_features(), "uncorrelated"
    )
    _uncorrelated_names = {
        "Client_Age_grp": "Client Age Group",
        "CreditCard_Balance_ARG_SP_pct_max": "Credit Card (Balance ARS / Total spendings) (Max)",
        "CreditCard_Payment_total_max": "Credit Card Total Payment (Max)",
        "CreditCard_Product": "Credit Card Product",
        "CreditCard_Total_Limit_diff_rel": "Credit Card Limit (Relative Diff)",
        "Operations_total_min": "Total Operations (Min)",
        "Quantity_Active_Products_min": "Active Products Quantity (Min)",
        "SavingAccount_Transfer_In_Amount_max": "Savings Account Transfer-In Amount (Max)",
        "SavingAccount_Transfer_In_Transactions_pct_max": "Savings Account Transfer-In Transactions (Max %)",
    }
    del groups_trainer
    eda.build_plot_tabs(
        _uncorrelated_names,
        settings.paths.images / "bivariate_analysis" / "uncorrelated",
        orientation="vertical",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5.4 Review of Correlated Features

    Se revisan las variables descartadas por correlación para poder intercambiar las variables mas importantes por variables mas fáciles de interpretar, si es que existen.
    """)
    return


@app.cell
def _(column_filter, trainer):
    _top_ranked_features = trainer.get_top_ranked_features()
    _correlations_dict = column_filter.correlations_for_each(_top_ranked_features)
    del trainer
    mo.ui.tabs(_correlations_dict, orientation="vertical")
    return


@app.cell
def _(column_filter):
    _correlated_names = {
        "Operations_total_mean": "Total Operations (Mean)",
        "Operations_total_median": "Total Operations (Median)",
        "CreditCard_Active": "Credit Card Active Status",
        "CreditCard_Balance_ARG_SP_pct_mean": "Credit Card (Balance ARS / Total spendings) (Mean)",
        "Quantity_Active_Products_median": "Active Products Quantity (Median)",
        "Quantity_Active_Products_mean": "Active Products Quantity (Mean)",
    }
    column_filter.plot_correlated(list(_correlated_names.keys()), "correlated")
    eda.build_plot_tabs(
        _correlated_names,
        settings.paths.images / "bivariate_analysis" / "correlated",
        orientation="vertical",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5.5 Feature Binning

    Las features mas relevantes se agrupan en bines y se codifican con su porcentaje respecto al target. A las features categóricas solo se las agrupa en bines (ya se codificaron anteriormente).


    Features mas relevantes:

    - `Client_Age_grp`: se fusionan los tramos de 50 a 69 años; el resto se fusionan en un solo bin ("Entre 18 y 49 años" + "Mayor a 70 años").
    - `Operations_total_mean` y `Operations_total_median`.
    - `CreditCard_Product`: se conservan los bines de los tipos de tarjeta 202 y 104 por separado; los demás tipos de tarjetas, que tienen bajo porcentaje de target o son tipos de tarjetas poco representativas, se consolidan en un solo bin (sin tarjeta de crédito + 102 + 123 + 124 + 702 + 1002).
    - `CreditCard_Active` (no hace falta codificar ni agrupar en bines).
    - `Quantity_Active_Products_min`.
    """)
    return


@app.cell
def _(column_filter):
    final_train, final_test = column_filter.apply_bin_transformations(
        [
            col_filter.BinTransformation(
                "Client_Age_grp", (col_filter.BinRange(4, 5), col_filter.BinRange(6, 7))
            ),
            col_filter.BinTransformation(
                "Operations_total_mean",
                (
                    col_filter.BinRange(2, 4),
                    col_filter.BinRange(5, 6),
                    col_filter.BinRange(7, 8),
                    col_filter.BinRange(9, 10),
                    col_filter.BinRange(11, 12),
                    col_filter.BinRange(13, 14),
                    col_filter.BinRange(15, 16),
                ),
            ),
            col_filter.BinTransformation(
                "Operations_total_median",
                (
                    col_filter.BinRange(2, 4),
                    col_filter.BinRange(5, 7),
                    col_filter.BinRange(8, 9),
                    col_filter.BinRange(10, 11),
                ),
            ),
            col_filter.BinTransformation(
                "CreditCard_Product",
                (col_filter.BinRange(5, 5), col_filter.BinRange(7, 7)),
            ),
            col_filter.BinTransformation(
                "Quantity_Active_Products_min",
                (col_filter.BinRange(1, 4), col_filter.BinRange(6, 9)),
            ),
        ]
    )
    best_features_names = {
        "Client_Age_grp": "Client Age Group",
        "Operations_total_mean": "Total Operations (Mean)",
        "CreditCard_Product": "Credit Card Product",
        "Quantity_Active_Products_min": "Active Products Quantity (Min)",
    }
    result = col_filter.select_features(final_train, final_test, best_features_names)
    mo.ui.tabs(
        {
            "Inspect final_train": eda.inspect_dataframe(result.train),
            "Describe final_train": eda.describe(result.train),
        }
    )
    return (result,)


@app.cell
def _(column_filter, result):
    column_filter.plot_adhoc(result.train, result.features_names, "best_features")
    del column_filter
    eda.build_plot_tabs(
        result.features_renames,
        settings.paths.images / "bivariate_analysis" / "best_features",
        orientation="vertical",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 6. Final Training and Performance Evaluation

    El modelo final se entrena solo con las 4 features finales seleccionadas luego del binning y se evalúa el mejor modelo con los mejores hiperparámetros encontrados. No se aplica balanceo sobre los datos porque la proporción del target, cercana al 30%, ya es suficiente.
    """)
    return


@app.cell
def _(result):
    _final_trainer = training.LGBMTrainer(
        result.train,
        result.features_names,
        n_iter=20,
        test=result.test,
    )
    _final_trainer.plot_top_features(
        plot_name="best_features", renames=result.features_renames
    )
    _final_trainer.plot_evaluation_metrics(plot_name="lightgbm")
    _final_trainer.plot_deciles(plot_name="deciles")
    del result
    mo.ui.tabs(
        {
            "Best Features Importances": mo.image(
                src=settings.paths.images / "plot_top_features" / "best_features.svg"
            ),
            "Best Features Cross Validation Results": _final_trainer.get_cv_results(
                include_logs=True
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## 6.1 Performance Metrics
    {mo.image(src=settings.paths.images / "plot_evaluation_metrics" / "lightgbm.svg")}
    {mo.image(src=settings.paths.images / "plot_evaluation_metrics" / "deciles.svg")}

    ### Training

    - Ordenamiento correcto en todos los deciles, con una distribución relativamente homogénea.
    - Lift del 1er decil de 2,4 y KS de 47,03 en el 5to decil.

    ### Testing

    - Ordenamiento correcto en todos los deciles, con una distribución relativamente homogénea.
    - Lift del 1er decil de 2,36 y KS de 47,22 en el 4to decil.

    ### Stability Gap

    - Diferencia de lift de 0,04 en el mismo decil.
    - Diferencia de KS de 0,19, con 1 decil de diferencia.
    """)
    return


if __name__ == "__main__":
    app.run()
