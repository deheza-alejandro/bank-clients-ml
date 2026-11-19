"""Cuaderno principal del proyecto bank_clients_ml.

Documenta el proceso de modelado en Marimo, desde el análisis exploratorio hasta
la evaluación final. Las decisiones de arquitectura están documentadas en
`docs/architecture/decisions.md`.
"""

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    import polars as pl

    from bank_clients_ml import binning, cleaning, column_groups, config, eda, sampling
    from bank_clients_ml import bivariate_analysis as bi_analysis
    from bank_clients_ml import feature_engineering as feat
    from bank_clients_ml import marimo_utils as mo_utils
    from bank_clients_ml import redundant_column_filter as col_filter

    settings = config.Settings.settings_cached()


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
    raw_data = pl.read_parquet(settings.paths.raw_data)

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
                raw_data.select(_less_than_zero_columns), condition="<", threshold=0
            ),
            "Days columns greater than 31": eda.count_row_matches(
                raw_data.select(_greater_than_thirty_one_columns),
                condition=">",
                threshold=31,
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

    Se elimina el registro detectado en la celda anterior, que solo contenía valores nulos. También se modifican los tipos de las variables de fechas y la variable id para evitar errores de tipo en las etapas posteriores.
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

    Se definen los meses de las ventanas de entrenamiento y predicción, con un mes de separación (lead window).

    ## 2.1 Universe Filtering Criteria

    Se conservan en el universo los clientes que cumplan estas condiciones:
    - Tienen 9 meses de historia.
    - No tienen, en el último mes de la ventana de entrenamiento:
        - `Package_Active`, para que el modelo aprenda a predecir la primera
          compra y no la renovación o el uso ya existente.
        - `CreditCard_CoBranding`, por regla de negocio ya se sabe que estos
          clientes no suelen comprar y la variable está inversamente
          correlacionada con el target.
    """)
    return


@app.cell
def _(clean_data):
    time_window_split = sampling.TimeWindowSplit.generate(
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
                    time_window_split.last_training_month,
                    mo.md("#### first_prediction_month:"),
                    time_window_split.first_prediction_month,
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

    Se analiza la ausencia de valores.
    """)
    return


@app.cell
def _(time_window_split):
    print(
        f"SavingAccount_Balance_Average unique values: "
        f"{time_window_split.training_data.select('SavingAccount_Balance_Average').n_unique()} \n"
    )
    mo.output.append(
        eda.describe(time_window_split.training_data).filter(pl.col("null_count") > 0)
    )
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
    _with_id = [settings.dataset.id, *column_groups.SAVING_ACCOUNT_COLS]
    mo.ui.tabs(
        {
            'When "Saving Account Balance (Average)" is Null': time_window_split.training_data.select(
                _with_id
            ).filter(pl.col("SavingAccount_Balance_Average").is_null()),
            "Non-null monetary values of Saving Account": eda.filter_nonzero(
                time_window_split.training_data.select(_with_id)
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Ante la imposibilidad de recalcular el promedio diario exacto con los datos
    disponibles, se aproxima como el promedio entre el saldo inicial y final
    (`SavingAccount_Balance_FirstDate` y `SavingAccount_Balance_LastDate`). En
    esta ejecución de ejemplo son solo cuatro registros, por lo que el impacto en
    el modelo es mínimo (ver ADR-004 en `docs/architecture/decisions.md`).
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

    `Region` es una variable estable del cliente, por lo que un nulo en
    entrenamiento suele deberse a un vacío de carga y no a un cambio real. Por
    lo tanto, se obtienen estos datos desde la ventana de predicción y se completan los
    nulos restantes con la región más común (`BUENOS AIRES`) (ver ADR-004 en
    `docs/architecture/decisions.md`).
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

    Se incorpora el tipo de tarjeta (`CreditCard_Product`) desde la ventana de
    predicción: se toma el valor del primer mes y, si es nulo o no existe, el del
    segundo, incluso si también es nulo.

    Para los nulos restantes, si `CreditCard_Active` es "No" en la ventana de predicción, y es "Yes"  en la ventana de entrenamiento, se asigna el tipo de tarjeta mas común `J55660104XX012`; en los demás casos se asigna "0" (cuando no tiene tarjeta activa en la ventana de predicción ni en la ventana de entrenamiento o cuando no tiene tarjeta activa en la ventana de entrenamiento, por mas que lo tenga en la ventana de predicción), que
    representa la ausencia de tarjeta (ver ADR-004 en `docs/architecture/decisions.md`).
    """)
    return


@app.cell
def _(time_window_split, training_data_2):
    _clients_credit_card_product = (
        time_window_split.prediction_data.sort(
            pl.col(settings.dataset.month) == time_window_split.first_prediction_month,
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

    Se toman las identity features del último mes de la ventana de entrenamiento y
    se convierten las variables con valores `Yes/No` y `M/F` a valores binarios
    (0 ó 1).
    """)
    return


@app.cell
def _(time_window_split, training_data_3):
    _binary_dict: dict[str, int] = {"Yes": 1, "No": 0, "M": 1, "F": 0}

    training_data_4 = training_data_3.with_columns(
        pl.col(column_groups.BINARY_IDENTITY_FEATURES_COLS)
        .replace_strict(_binary_dict)
        .cast(pl.UInt8)
    )

    identity_features = training_data_4.filter(
        pl.col(settings.dataset.month) == time_window_split.last_training_month
    ).select(column_groups.get_identity_features_cols())

    del training_data_3
    mo.ui.tabs(
        {
            "Low cardinality columns of identity_features": eda.low_cardinality_value_counts(
                identity_features.drop(column_groups.CATEGORICAL_COLS)
            ),
            "Inspect identity_features": eda.inspect_dataframe(identity_features),
            "Describe identity_features": eda.describe(identity_features),
            "Inspect training_data": eda.inspect_dataframe(training_data_4),
        }
    )
    return identity_features, training_data_4


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3.3 Categorical Features Encoding

    Se codifican `Client_Age_grp`, `Region` y `CreditCard_Product` con su porcentaje
    respecto al target (ver ADR-003 en `docs/architecture/decisions.md`).
    """)
    return


@app.cell
def _(identity_features, training_data_4):
    _categorical_cols = column_groups.CATEGORICAL_COLS
    _temp_value_counts = eda.low_cardinality_value_counts(
        identity_features.select(_categorical_cols)
    )
    identity_features_1, training_data_5 = feat.target_encode_columns(
        identity_features=identity_features, training_data=training_data_4
    )
    del identity_features
    del training_data_4
    mo.ui.tabs(
        {
            "Categorical columns before encode": _temp_value_counts,
            "Categorical columns after encode": eda.low_cardinality_value_counts(
                identity_features_1.select(_categorical_cols)
            ),
            "Inspect training_data after drop": eda.inspect_dataframe(training_data_5),
        }
    )
    return identity_features_1, training_data_5


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3.4 Date-Derived Features

    Se reemplazan las fechas absolutas por la diferencia entre el primer y el último
    producto y por la recencia hasta el mes siguiente al último mes de
    entrenamiento (ver ADR-005 en `docs/architecture/decisions.md`).
    """)
    return


@app.cell
def _(identity_features_1, time_window_split):
    identity_features_2 = feat.transform_date_features(
        identity_features_1, time_window_split.last_training_month
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

    Se analizan antes de aplicar las transformaciones:
    - Valores con ceros
    - Valores monetarios de CreditCard
    - Valores mínimos
    """)
    return


@app.cell
def _(training_data_5):
    mo.ui.tabs(
        {
            "Columns with zeros: training_data": eda.columns_with_zeros(
                training_data_5
            ),
            "CreditCard rows with no zeros: training_data": eda.filter_nonzero(
                training_data_5.select(
                    settings.dataset.id, *column_groups.CREDIT_CARD_COLS
                )
            ),
            "Mins in (-1, 1): training_data": eda.mins_in_range(
                training_data_5, low=-1, high=1
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Se aplican las transformaciones.
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
                training_data_6.select(_greater_than_one_hundred_columns),
                condition=">",
                threshold=100,
            ),
        }
    )
    return (training_data_6,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 3.6 Aggregate Features

    Se consolida la historia de nueve meses en una tabla con un registro por
    cliente para el entrenamiento supervisado.
    """)
    return


@app.cell
def _(training_data_6):
    data_agg = feat.aggregate_monthly_to_client(training_data_6)
    del training_data_6
    eda.inspect_dataframe(data_agg)
    return (data_agg,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 4. Analytical Base Table (ABT) and Train-Test Split

    Se aplican transformaciones adicionales sobre la ABT y se realiza una partición
    estratificada para preservar la tasa del target en el set de entrenamiento y el set de prueba.
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
            "Mins in (-1, 1): ABT": eda.mins_in_range(_abt, low=-1, high=1),
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

    Se eliminan columnas constantes, columnas binarias desbalanceadas y columnas
    altamente correlacionadas antes del ranking con LightGBM (ver ADR-006 en
    `docs/architecture/decisions.md`).
    """)
    return


@app.cell
def _(test, train):
    column_filter = col_filter.RedundantColumnFilter.generate(
        train=train,
        test=test,
        imbalanced_binary_threshold=0.10,
        correlation_threshold=0.80,
    )
    del train
    del test
    mo.ui.tabs(
        {
            "Constant columns removed": column_filter.constant_cols,
            "Imbalanced binary columns removed (Value counts)": column_filter.imbalanced_binary_counts,
            "Correlated columns removed": column_filter.correlated_columns,
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

    Se entrena un modelo LightGBM por grupo de variables y luego un modelo con
    las mejores variables de cada grupo (ver ADR-007 en `docs/architecture/decisions.md`).

    No se aplica estandarización (ver ADR-009 en `docs/architecture/decisions.md`).
    """)
    return


@app.cell
def _(column_filter):
    group_trainers, top_grouped_features_trainer, correlations_dict = (
        column_filter.heavy_operations_and_correlations("top_grouped_features")
    )
    bivariate_analyzer = bi_analysis.BivariateAnalyzer.from_redundant_column_filter(
        column_filter
    )
    del column_filter
    mo.ui.tabs(
        {
            "Groups lengths": mo.vstack(group_trainers.groups_lengths),
            '"Others" Columns': group_trainers.others_columns,
            "Groups Cross Validation Results": mo.accordion(
                mo_utils.build_group_cv_views(group_trainers, include_logs=True)
            ),
        }
    )
    return (
        bivariate_analyzer,
        correlations_dict,
        group_trainers,
        top_grouped_features_trainer,
    )


@app.cell
def _(group_trainers):
    mo_utils.build_plot_tabs(
        group_trainers.groups_name_mapping, settings.paths.plots / "plot_top_features"
    )
    return


@app.cell
def _(group_trainers, top_grouped_features_trainer):
    del group_trainers
    mo.ui.tabs(
        {
            "Top grouped features importances": mo.image(
                src=settings.paths.plots
                / "plot_top_features"
                / "top_grouped_features.svg"
            ),
            "Cross Validation Results": mo_utils.build_trainer_cv_view(
                top_grouped_features_trainer, include_logs=True
            ),
        }
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5.3 Bivariate Analysis

    Se analiza la relación entre las variables más relevantes y el target.
    """)
    return


@app.cell
def _(bivariate_analyzer, top_grouped_features_trainer):
    del top_grouped_features_trainer
    _uncorrelated_name_mapping = binning.NameMapping.from_dict(
        {
            "Client_Age_grp": "Client Age Group",
            "CreditCard_Balance_ARG_SP_pct_max": "Credit Card (Balance ARS / Total spendings) (Max)",
            "CreditCard_Payment_total_max": "Credit Card Total Payment (Max)",
            "CreditCard_Product": "Credit Card Product",
            "CreditCard_Total_Limit_min": "Credit Card Limit (Min)",
            "Operations_total_min": "Total Operations (Min)",
            "Quantity_Active_Products_min": "Active Products Quantity (Min)",
            "SavingAccount_Transfer_In_Amount_max": "Savings Account Transfer-In Amount (Max)",
            "SavingAccount_Transfer_In_Transactions_pct_max": "Savings Account Transfer-In Transactions (Max %)",
        }
    )
    bivariate_analyzer.plot_bivariate(_uncorrelated_name_mapping, "uncorrelated")
    mo_utils.build_plot_tabs(
        _uncorrelated_name_mapping,
        settings.paths.plots / "bivariate_analysis" / "uncorrelated",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5.4 Review of Correlated Features

    Se revisan las variables descartadas por correlación para intercambiar las variables más
    importantes por otras más fáciles de interpretar, cuando existan.
    """)
    return


@app.cell
def _(correlations_dict):
    mo_utils.build_correlation_tabs(correlations_dict)
    return


@app.cell
def _(bivariate_analyzer, correlations_dict):
    del correlations_dict
    _correlated_name_mapping = binning.NameMapping.from_dict(
        {
            "Operations_total_mean": "Total Operations (Mean)",
            "Operations_total_median": "Total Operations (Median)",
            "CreditCard_Active": "Credit Card Active Status",
            "CreditCard_Balance_ARG_SP_pct_mean": "Credit Card (Balance ARS / Total spendings) (Mean)",
            "Quantity_Active_Products_median": "Active Products Quantity (Median)",
            "Quantity_Active_Products_mean": "Active Products Quantity (Mean)",
        }
    )
    bivariate_analyzer.plot_bivariate(_correlated_name_mapping, "correlated")
    mo_utils.build_plot_tabs(
        _correlated_name_mapping,
        settings.paths.plots / "bivariate_analysis" / "correlated",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 5.5 Feature Binning

    Se agrupan las features relevantes en bines y se codifican con su porcentaje
    respecto al target (ver ADR-008 en `docs/architecture/decisions.md`).

    Features más relevantes:

    - `Client_Age_grp`: se fusionan los tramos de 50 a 69 años; el resto se fusionan en
      un solo bin ("Entre 18 y 49 años" + "Mayor a 70 años").
    - `Operations_total_mean`.
    - `CreditCard_Product`: se conservan los bines de los tipos 202 y 104 por
      separado; se consolidan los demás tipos, con bajo porcentaje de target o poca
      representatividad, en un solo bin (sin tarjeta de crédito + 102 + 123 +
      124 + 702 + 1002).
    - `CreditCard_Active` (no requiere codificación ni binning).
    - `Quantity_Active_Products_min`.
    """)
    return


@app.cell
def _(bivariate_analyzer):
    result = bivariate_analyzer.apply_binning(
        [
            binning.Binner("Client_Age_grp", "Client Age Group")
            .merge_bins(4, 5)
            .merge_bins(6, 7),
            binning.Binner("Operations_total_mean", "Total Operations (Mean)")
            .merge_bins(2, 4)
            .merge_bins(5, 6)
            .merge_bins(7, 8)
            .merge_bins(9, 10)
            .merge_bins(11, 12)
            .merge_bins(13, 14)
            .merge_bins(15, 16),
            binning.Binner("CreditCard_Product", "Credit Card Product")
            .merge_bins(5)
            .merge_bins(7),
            binning.Binner(
                "Quantity_Active_Products_min", "Active Products Quantity (Min)"
            )
            .merge_bins(1, 4)
            .merge_bins(6, 9),
        ]
    )
    del bivariate_analyzer
    mo.ui.tabs(
        {
            "Inspect final_train": eda.inspect_dataframe(result.train),
            "Describe final_train": eda.describe(result.train),
        }
    )
    return (result,)


@app.cell
def _(result):
    result.plot_bivariate_analysis("best_features")
    mo_utils.build_plot_tabs(
        result.name_mapping,
        settings.paths.plots / "bivariate_analysis" / "best_features",
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 6. Final Training and Performance Evaluation

    Se entrena el modelo final con las cuatro features seleccionadas después del
    binning y se evalúa el mejor modelo con los hiperparámetros encontrados
    (ver ADR-010 en `docs/architecture/decisions.md`). No se aplica balanceo (ver ADR-009 en
    `docs/architecture/decisions.md`).
    """)
    return


@app.cell
def _(result):
    result.plot_top_features(plot_name="best_features")
    result.evaluate_and_plot_metrics(roc_curve_name="roc_curve", deciles_name="deciles")
    mo.ui.tabs(
        {
            "Best Features Importances": mo.image(
                src=settings.paths.plots / "plot_top_features" / "best_features.svg"
            ),
            "Best Features Cross Validation Results": mo_utils.build_trainer_cv_view(
                result.trainer, include_logs=True
            ),
        }
    )
    return


@app.cell
def _(result):
    del result
    return


@app.cell(hide_code=True)
def _():
    mo.md(rf"""
    ## 6.1 Performance Metrics
    {mo.image(src=settings.paths.plots / "plot_evaluation_metrics" / "roc_curve.svg")}
    {mo.image(src=settings.paths.plots / "plot_evaluation_metrics" / "deciles.svg")}

    Los valores siguientes corresponden a la ejecución del notebook con
    `data/example_data.parquet`; son ilustrativos y pueden variar si cambia el
    conjunto de datos.

    ### Training

    - Ordenamiento correcto en todos los deciles, con una distribución
      relativamente homogénea.
    - Lift del primer decil de 2,4 y KS de 47,03 en el quinto decil.

    ### Testing

    - Ordenamiento correcto en todos los deciles, con una distribución
      relativamente homogénea.
    - Lift del primer decil de 2,39 y KS de 48,19 en el cuarto decil.

    ### Stability Gap

    - Diferencia de lift de 0,01 en el mismo decil.
    - Diferencia de KS de 1,16, con 1 decil de diferencia.
    """)
    return


if __name__ == "__main__":
    app.run()
