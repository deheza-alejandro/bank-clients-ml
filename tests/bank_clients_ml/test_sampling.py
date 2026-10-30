"""Tests unitarios para el módulo de muestreo y ventanas temporales."""

from datetime import date
from typing import cast

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from bank_clients_ml import config, sampling


@pytest.fixture
def clean_data() -> pl.DataFrame:
    """Devuelve datos limpios con clientes válidos e inválidos para las ventanas.

    ## Returns:
        DataFrame con diez meses de historia y siete clientes que cubren los
        criterios de cantidad de meses y banderas de producto.
    """
    months = [date(2024, month, 1) for month in range(1, 11)]
    clients = {
        1: ((date(2024, 2, 1),), "No", "No", 1),
        2: ((date(2024, 3, 1),), "No", "No", 0),
        3: ((date(2024, 4, 1),), "Yes", "No", 1),
        4: ((date(2024, 5, 1),), "No", "Yes", 0),
        5: ((), "No", "No", 1),
        6: ((date(2024, 7, 1),), "No", "No", 0),
        7: ((date(2024, 2, 1), date(2024, 3, 1)), "No", "No", 1),
    }
    rows = []
    for client_id, client in clients.items():
        missing_months, package_active, co_branding, target = client
        for month in months:
            if month in missing_months:
                continue
            is_last_training_month = month == date(2024, 7, 1)
            rows.append(
                {
                    "client_id": client_id,
                    "Month": month,
                    "Package_Active": (
                        package_active if is_last_training_month else "No"
                    ),
                    "CreditCard_CoBranding": (
                        co_branding if is_last_training_month else "No"
                    ),
                    "Target": target,
                    "Balance": float(client_id * 100 + month.month),
                }
            )
    return pl.DataFrame(
        rows,
        schema={
            "client_id": pl.Int64,
            "Month": pl.Date,
            "Package_Active": pl.String,
            "CreditCard_CoBranding": pl.String,
            "Target": pl.Int64,
            "Balance": pl.Float64,
        },
    )


@pytest.fixture
def custom_settings() -> config.Settings:
    """Devuelve una configuración con nombres de columna alternativos."""
    return config.Settings(
        dataset=config.DatasetColumns(
            id="customer_id",
            target="churn",
            month="snapshot_month",
            package_active="has_package",
            credit_card_co_branding="has_branding",
        )
    )


@pytest.fixture
def renamed_clean_data(clean_data: pl.DataFrame) -> pl.DataFrame:
    """Renombra las columnas de los datos limpios según la configuración alternativa.

    ## Args:
        clean_data: Datos limpios con los nombres por defecto.

    ## Returns:
        DataFrame con los nombres de columna alternativos.
    """
    return clean_data.rename(
        {
            "client_id": "customer_id",
            "Target": "churn",
            "Month": "snapshot_month",
            "Package_Active": "has_package",
            "CreditCard_CoBranding": "has_branding",
        }
    )


@pytest.fixture
def abt_data() -> pl.DataFrame:
    """Devuelve una tabla balanceada para probar la partición estratificada.

    ## Returns:
        DataFrame con veinte clientes por cada clase del target.
    """
    return pl.DataFrame(
        {
            "client_id": list(range(1, 41)),
            "Target": [0] * 20 + [1] * 20,
            "Feature": [float(value) for value in range(40)],
        }
    )


@pytest.fixture
def imbalanced_train() -> pl.DataFrame:
    """Devuelve un conjunto desbalanceado con mayoría de target cero.

    ## Returns:
        DataFrame con treinta clientes de target cero y cinco de target uno.
    """
    return pl.DataFrame(
        {
            "client_id": list(range(1, 36)),
            "Target": [0] * 30 + [1] * 5,
            "Feature": [float(value) for value in range(35)],
        }
    )


@pytest.fixture
def twelve_months_data() -> pl.DataFrame:
    """Devuelve un DataFrame con los doce meses de 2024.

    ## Returns:
        DataFrame con una fila por mes.
    """
    return pl.DataFrame(
        {"Month": [date(2024, month, 1) for month in range(1, 13)]},
        schema={"Month": pl.Date},
    )


@pytest.mark.parametrize(
    ("prediction_window_size", "expected_training", "expected_prediction"),
    [
        pytest.param(
            1,
            tuple(date(2024, month, 1) for month in range(1, 11)),
            (date(2024, 12, 1),),
            id="one-month",
        ),
        pytest.param(
            2,
            tuple(date(2024, month, 1) for month in range(1, 10)),
            (date(2024, 11, 1), date(2024, 12, 1)),
            id="two-months",
        ),
        pytest.param(
            3,
            tuple(date(2024, month, 1) for month in range(1, 9)),
            (
                date(2024, 10, 1),
                date(2024, 11, 1),
                date(2024, 12, 1),
            ),
            id="three-months",
        ),
    ],
)
def test_should_build_training_and_prediction_windows_with_one_lead_month(
    twelve_months_data: pl.DataFrame,
    settings: config.Settings,
    prediction_window_size: int,
    expected_training: tuple[date, ...],
    expected_prediction: tuple[date, ...],
) -> None:
    """Verifica que las ventanas respeten el tamaño y dejen un mes de separación.

    ## Args:
        twelve_months_data: Datos con los doce meses del año.
        settings: Configuración con el nombre de la columna de meses.
        prediction_window_size: Cantidad de meses reservados para predicción.
        expected_training: Meses esperados en la ventana de entrenamiento.
        expected_prediction: Meses esperados en la ventana de predicción.
    """
    training_months, prediction_months = sampling._build_date_windows(
        twelve_months_data, prediction_window_size, settings
    )

    lead_month = date(2024, 12 - prediction_window_size, 1)
    assert training_months == expected_training
    assert prediction_months == expected_prediction
    assert training_months[-1] < lead_month < prediction_months[0]


def test_should_build_expected_windows_and_expose_boundary_months(
    clean_data: pl.DataFrame,
    settings: config.Settings,
) -> None:
    """Verifica las ventanas generadas y los meses límite expuestos.

    ## Args:
        clean_data: Datos limpios con diez meses de historia.
        settings: Configuración con los nombres de las columnas.
    """
    split = sampling.TimeWindowSplit.generate(clean_data, 2, settings=settings)

    expected_training = tuple(date(2024, month, 1) for month in range(1, 8))
    expected_prediction = (date(2024, 9, 1), date(2024, 10, 1))
    assert split.training_months == expected_training
    assert split.prediction_months == expected_prediction
    assert split.last_training_month == date(2024, 7, 1)
    assert split.first_prediction_month == date(2024, 9, 1)


@pytest.mark.parametrize("prediction_window_size", [0, -1])
def test_should_raise_value_error_if_prediction_window_size_is_less_than_one(
    clean_data: pl.DataFrame,
    settings: config.Settings,
    prediction_window_size: int,
) -> None:
    """Verifica que se rechacen tamaños de ventana de predicción inválidos.

    ## Args:
        clean_data: Datos limpios con diez meses de historia.
        settings: Configuración con los nombres de las columnas.
        prediction_window_size: Tamaño de ventana inválido.
    """
    with pytest.raises(ValueError, match="Invalid prediction_window_size"):
        sampling.TimeWindowSplit.generate(
            clean_data, prediction_window_size, settings=settings
        )


def test_should_select_only_clients_with_nine_months_and_valid_flags(
    clean_data: pl.DataFrame,
    settings: config.Settings,
) -> None:
    """Verifica que el universo solo incluya clientes elegibles.

    Los clientes deben tener nueve meses y no tener paquete ni tarjeta
    co-branding en el último mes de entrenamiento.

    ## Args:
        clean_data: Datos limpios con clientes válidos e inválidos.
        settings: Configuración con los nombres de las columnas.
    """
    split = sampling.TimeWindowSplit.generate(clean_data, 2, settings=settings)

    assert split.universe_and_target.columns == ["client_id", "Target"]
    assert split.universe_and_target.height == 2
    assert set(split.universe_and_target["client_id"].to_list()) == {1, 2}
    assert split.universe_and_target.sort("client_id")["Target"].to_list() == [
        1,
        0,
    ]


def test_should_keep_full_history_for_selected_clients(
    clean_data: pl.DataFrame,
    settings: config.Settings,
) -> None:
    """Verifica que los datos del universo conserven toda la historia.

    ## Args:
        clean_data: Datos limpios con clientes válidos e inválidos.
        settings: Configuración con los nombres de las columnas.
    """
    split = sampling.TimeWindowSplit.generate(clean_data, 2, settings=settings)

    expected_months = {date(2024, month, 1) for month in range(1, 11)}
    assert split.universe_and_target_data.height == 18
    assert set(split.universe_and_target_data["client_id"].to_list()) == {1, 2}
    assert set(split.universe_and_target_data["Month"].to_list()) == expected_months


def test_should_keep_a_single_target_column_in_universe_data(
    clean_data: pl.DataFrame,
    settings: config.Settings,
) -> None:
    """Verifica que el target no se duplique al unir el universo con la historia.

    ## Args:
        clean_data: Datos limpios con clientes válidos e inválidos.
        settings: Configuración con los nombres de las columnas.
    """
    split = sampling.TimeWindowSplit.generate(clean_data, 2, settings=settings)

    assert split.universe_and_target_data.columns.count("Target") == 1
    assert set(split.universe_and_target_data.columns) == {
        "client_id",
        "Month",
        "Package_Active",
        "CreditCard_CoBranding",
        "Balance",
        "Target",
    }


def test_should_restrict_training_data_to_training_months(
    clean_data: pl.DataFrame,
    settings: config.Settings,
) -> None:
    """Verifica que el entrenamiento use solo la ventana de entrenamiento.

    ## Args:
        clean_data: Datos limpios con clientes válidos e inválidos.
        settings: Configuración con los nombres de las columnas.
    """
    split = sampling.TimeWindowSplit.generate(clean_data, 2, settings=settings)

    assert split.training_data.height == 12
    assert split.training_data["client_id"].n_unique() == 2
    assert set(split.training_data["Month"].to_list()) == set(split.training_months)
    assert split.training_data["Month"].max() == split.last_training_month


def test_should_restrict_prediction_data_to_prediction_months(
    clean_data: pl.DataFrame,
    settings: config.Settings,
) -> None:
    """Verifica que la predicción use solo la ventana de predicción.

    ## Args:
        clean_data: Datos limpios con clientes válidos e inválidos.
        settings: Configuración con los nombres de las columnas.
    """
    split = sampling.TimeWindowSplit.generate(clean_data, 2, settings=settings)

    assert split.prediction_data.height == 4
    assert split.prediction_data["client_id"].n_unique() == 2
    assert set(split.prediction_data["Month"].to_list()) == set(split.prediction_months)
    assert split.prediction_data["Month"].min() == split.first_prediction_month


def test_should_exclude_lead_month_from_training_and_prediction_data(
    clean_data: pl.DataFrame,
    settings: config.Settings,
) -> None:
    """Verifica que el mes intermedio no aparezca en ninguno de los conjuntos.

    ## Args:
        clean_data: Datos limpios con clientes válidos e inválidos.
        settings: Configuración con los nombres de las columnas.
    """
    split = sampling.TimeWindowSplit.generate(clean_data, 2, settings=settings)

    lead_month = date(2024, 8, 1)
    assert lead_month not in split.training_months
    assert lead_month not in split.prediction_months
    assert lead_month not in split.training_data["Month"].to_list()
    assert lead_month not in split.prediction_data["Month"].to_list()


def test_should_take_universe_target_from_prediction_months(
    clean_data: pl.DataFrame,
    settings: config.Settings,
) -> None:
    """Verifica que el target del universo provenga de la ventana de predicción.

    El cliente 1 tiene target 0 en entrenamiento y 1 en predicción, por lo que
    el universo debe conservar el valor 1.

    ## Args:
        clean_data: Datos limpios con clientes válidos e inválidos.
        settings: Configuración con los nombres de las columnas.
    """
    data = clean_data.with_columns(
        pl.when((pl.col("client_id") == 1) & (pl.col("Month") < date(2024, 8, 1)))
        .then(pl.lit(0, dtype=pl.Int64))
        .otherwise(pl.col("Target"))
        .alias("Target")
    )

    split = sampling.TimeWindowSplit.generate(data, 2, settings=settings)

    universe_target = split.universe_and_target.filter(pl.col("client_id") == 1)[
        "Target"
    ].to_list()
    training_targets = (
        split.training_data.filter(pl.col("client_id") == 1)["Target"]
        .unique()
        .to_list()
    )
    prediction_targets = (
        split.prediction_data.filter(pl.col("client_id") == 1)["Target"]
        .unique()
        .to_list()
    )
    assert universe_target == [1]
    assert training_targets == [1]
    assert prediction_targets == [1]


def test_should_ignore_product_flags_outside_last_training_month(
    settings: config.Settings,
) -> None:
    """Verifica que solo importen las banderas del último mes de entrenamiento.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    months = [date(2024, month, 1) for month in range(1, 10)]
    data = pl.DataFrame(
        {
            "client_id": [1] * 9,
            "Month": months,
            "Package_Active": [
                "Yes" if month == date(2024, 2, 1) else "No" for month in months
            ],
            "CreditCard_CoBranding": ["No"] * 9,
            "Target": [1] * 9,
        },
        schema={
            "client_id": pl.Int64,
            "Month": pl.Date,
            "Package_Active": pl.String,
            "CreditCard_CoBranding": pl.String,
            "Target": pl.Int64,
        },
    )

    split = sampling.TimeWindowSplit.generate(data, 2, settings=settings)

    assert split.universe_and_target["client_id"].to_list() == [1]


def test_should_return_empty_datasets_when_no_client_has_exactly_nine_months(
    clean_data: pl.DataFrame,
    settings: config.Settings,
) -> None:
    """Verifica el resultado vacío cuando ningún cliente cumple el historial.

    ## Args:
        clean_data: Datos limpios con clientes válidos e inválidos.
        settings: Configuración con los nombres de las columnas.
    """
    data = clean_data.filter(pl.col("client_id") == 5)

    split = sampling.TimeWindowSplit.generate(data, 2, settings=settings)

    assert split.universe_and_target.height == 0
    assert split.universe_and_target_data.height == 0
    assert split.training_data.height == 0
    assert split.prediction_data.height == 0
    assert set(split.training_data.columns) == set(data.columns)


def test_should_use_custom_column_names_in_time_window_split(
    renamed_clean_data: pl.DataFrame,
    custom_settings: config.Settings,
) -> None:
    """Verifica que se respeten los nombres de columna configurados.

    ## Args:
        renamed_clean_data: Datos con nombres de columna alternativos.
        custom_settings: Configuración con los nombres alternativos.
    """
    split = sampling.TimeWindowSplit.generate(
        renamed_clean_data, 2, settings=custom_settings
    )

    assert split.universe_and_target.columns == ["customer_id", "churn"]
    assert set(split.universe_and_target["customer_id"].to_list()) == {1, 2}
    assert "churn" in split.training_data.columns
    assert "customer_id" in split.prediction_data.columns


def test_should_use_global_settings_when_settings_is_not_provided_for_time_windows(
    renamed_clean_data: pl.DataFrame,
    custom_settings: config.Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifica que se use la configuración global si no se recibe una explícita.

    ## Args:
        renamed_clean_data: Datos con nombres de columna alternativos.
        custom_settings: Configuración con los nombres alternativos.
        monkeypatch: Utilidad para reemplazar la configuración global.
    """
    monkeypatch.setattr(config, "get_settings", lambda: custom_settings)

    split = sampling.TimeWindowSplit.generate(renamed_clean_data, 2)

    assert split.universe_and_target.columns == ["customer_id", "churn"]
    assert set(split.universe_and_target["customer_id"].to_list()) == {1, 2}


@pytest.mark.parametrize("test_ratio", [0.0, 1.0, -0.1, 1.1])
def test_should_raise_value_error_if_test_ratio_is_out_of_open_interval(
    abt_data: pl.DataFrame,
    settings: config.Settings,
    test_ratio: float,
) -> None:
    """Verifica que se rechacen proporciones de prueba inválidas.

    ## Args:
        abt_data: Tabla balanceada para la partición.
        settings: Configuración con el nombre del target.
        test_ratio: Proporción de prueba inválida.
    """
    with pytest.raises(ValueError, match="Invalid test_ratio"):
        sampling.stratified_train_test_split(abt_data, test_ratio, settings=settings)


def test_should_split_each_class_according_to_test_ratio(
    abt_data: pl.DataFrame,
    settings: config.Settings,
) -> None:
    """Verifica que la partición conserve la proporción de cada clase.

    ## Args:
        abt_data: Tabla balanceada para la partición.
        settings: Configuración con el nombre del target.
    """
    train, test = sampling.stratified_train_test_split(abt_data, 0.3, settings=settings)

    assert train.height == 28
    assert test.height == 12
    assert train.filter(pl.col("Target") == 0).height == 14
    assert train.filter(pl.col("Target") == 1).height == 14
    assert test.filter(pl.col("Target") == 0).height == 6
    assert test.filter(pl.col("Target") == 1).height == 6
    assert test.height / abt_data.height == pytest.approx(0.3)


def test_should_preserve_all_rows_after_split(
    abt_data: pl.DataFrame,
    settings: config.Settings,
) -> None:
    """Verifica que la partición no pierda ni duplique filas.

    ## Args:
        abt_data: Tabla balanceada para la partición.
        settings: Configuración con el nombre del target.
    """
    train, test = sampling.stratified_train_test_split(abt_data, 0.3, settings=settings)

    combined = pl.concat([train, test]).sort("client_id")
    expected = abt_data.sort("client_id")
    assert_frame_equal(combined, expected)


def test_should_return_only_original_columns_after_split(
    abt_data: pl.DataFrame,
    settings: config.Settings,
) -> None:
    """Verifica que se eliminen las columnas internas de la partición.

    ## Args:
        abt_data: Tabla balanceada para la partición.
        settings: Configuración con el nombre del target.
    """
    train, test = sampling.stratified_train_test_split(abt_data, 0.3, settings=settings)

    assert train.columns == abt_data.columns
    assert test.columns == abt_data.columns


def test_should_produce_same_split_for_same_random_state(
    abt_data: pl.DataFrame,
    settings: config.Settings,
) -> None:
    """Verifica que la partición sea reproducible con la misma semilla.

    ## Args:
        abt_data: Tabla balanceada para la partición.
        settings: Configuración con el nombre del target y la semilla.
    """
    first_train, first_test = sampling.stratified_train_test_split(
        abt_data, 0.3, settings=settings
    )
    second_train, second_test = sampling.stratified_train_test_split(
        abt_data, 0.3, settings=settings
    )

    assert_frame_equal(first_train, second_train)
    assert_frame_equal(first_test, second_test)


def test_should_split_single_class_data_according_to_test_ratio(
    settings: config.Settings,
) -> None:
    """Verifica que la partición funcione con una sola clase en el target.

    ## Args:
        settings: Configuración con el nombre del target.
    """
    data = pl.DataFrame(
        {
            "client_id": list(range(1, 11)),
            "Target": [1] * 10,
            "Feature": [float(value) for value in range(10)],
        }
    )

    train, test = sampling.stratified_train_test_split(data, 0.3, settings=settings)

    assert test.height == 3
    assert train.height == 7


def test_should_use_custom_target_column_in_split(
    abt_data: pl.DataFrame,
    custom_settings: config.Settings,
) -> None:
    """Verifica que se use el nombre de target configurado.

    ## Args:
        abt_data: Tabla balanceada para la partición.
        custom_settings: Configuración con el target alternativo.
    """
    data = abt_data.rename({"client_id": "customer_id", "Target": "churn"})

    train, test = sampling.stratified_train_test_split(
        data, 0.3, settings=custom_settings
    )

    assert train.columns == data.columns
    assert test.height == 12
    assert test.filter(pl.col("churn") == 0).height == 6
    assert test.filter(pl.col("churn") == 1).height == 6


def test_should_use_global_settings_when_settings_is_not_provided_for_split(
    abt_data: pl.DataFrame,
    custom_settings: config.Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifica que la partición use la configuración global si no se indica.

    ## Args:
        abt_data: Tabla balanceada para la partición.
        custom_settings: Configuración con el target alternativo.
        monkeypatch: Utilidad para reemplazar la configuración global.
    """
    data = abt_data.rename({"client_id": "customer_id", "Target": "churn"})
    monkeypatch.setattr(config, "get_settings", lambda: custom_settings)

    train, test = sampling.stratified_train_test_split(data, 0.3)

    assert train.height + test.height == data.height
    assert test.height == 12


@pytest.mark.parametrize("target_proportion", [0.0, 1.0, -0.1, 1.1])
def test_should_raise_value_error_if_target_proportion_is_out_of_open_interval(
    imbalanced_train: pl.DataFrame,
    settings: config.Settings,
    target_proportion: float,
) -> None:
    """Verifica que se rechacen proporciones objetivo inválidas.

    ## Args:
        imbalanced_train: Conjunto desbalanceado de entrenamiento.
        settings: Configuración con los nombres de columnas.
        target_proportion: Proporción objetivo inválida.
    """
    with pytest.raises(ValueError, match="Invalid target_proportion"):
        sampling.oversample_with_unique_ids(
            imbalanced_train, target_proportion, settings=settings
        )


def test_should_keep_all_majority_rows_unchanged(
    imbalanced_train: pl.DataFrame,
    settings: config.Settings,
) -> None:
    """Verifica que la clase mayoritaria se conserve intacta.

    ## Args:
        imbalanced_train: Conjunto desbalanceado de entrenamiento.
        settings: Configuración con los nombres de columnas.
    """
    balanced = sampling.oversample_with_unique_ids(
        imbalanced_train, 0.5, settings=settings
    )

    expected_majority = imbalanced_train.filter(pl.col("Target") == 0).sort("client_id")
    actual_majority = balanced.filter(pl.col("Target") == 0).sort("client_id")
    assert_frame_equal(actual_majority, expected_majority)


@pytest.mark.parametrize(
    ("target_proportion", "expected_minority_count"),
    [
        pytest.param(0.5, 30, id="half"),
        pytest.param(0.4, 20, id="forty-percent"),
        pytest.param(0.25, 10, id="quarter"),
    ],
)
def test_should_oversample_minority_class_to_requested_proportion(
    imbalanced_train: pl.DataFrame,
    settings: config.Settings,
    target_proportion: float,
    expected_minority_count: int,
) -> None:
    """Verifica que la clase minoritaria alcance la proporción solicitada.

    ## Args:
        imbalanced_train: Conjunto desbalanceado de entrenamiento.
        settings: Configuración con los nombres de columnas.
        target_proportion: Proporción objetivo de la clase minoritaria.
        expected_minority_count: Cantidad esperada de filas minoritarias.
    """
    balanced = sampling.oversample_with_unique_ids(
        imbalanced_train, target_proportion, settings=settings
    )

    majority_count = balanced.filter(pl.col("Target") == 0).height
    minority_count = balanced.filter(pl.col("Target") == 1).height
    assert majority_count == 30
    assert minority_count == expected_minority_count
    assert balanced.height == 30 + expected_minority_count
    assert minority_count / balanced.height == pytest.approx(target_proportion)


def test_should_assign_unique_ids_to_all_rows(
    imbalanced_train: pl.DataFrame,
    settings: config.Settings,
) -> None:
    """Verifica que el oversampling asigne identificadores únicos.

    ## Args:
        imbalanced_train: Conjunto desbalanceado de entrenamiento.
        settings: Configuración con los nombres de columnas.
    """
    balanced = sampling.oversample_with_unique_ids(
        imbalanced_train, 0.5, settings=settings
    )

    max_original_id = cast(int, imbalanced_train["client_id"].max())
    minority_ids = balanced.filter(pl.col("Target") == 1)["client_id"]
    assert balanced["client_id"].n_unique() == balanced.height
    assert cast(int, minority_ids.min()) > max_original_id


def test_should_preserve_columns_and_schema_after_oversampling(
    imbalanced_train: pl.DataFrame,
    settings: config.Settings,
) -> None:
    """Verifica que el oversampling conserve columnas y tipos de datos.

    ## Args:
        imbalanced_train: Conjunto desbalanceado de entrenamiento.
        settings: Configuración con los nombres de columnas.
    """
    balanced = sampling.oversample_with_unique_ids(
        imbalanced_train, 0.5, settings=settings
    )

    assert balanced.columns == imbalanced_train.columns
    assert balanced.schema == imbalanced_train.schema


def test_should_produce_same_balanced_data_for_same_random_state(
    imbalanced_train: pl.DataFrame,
    settings: config.Settings,
) -> None:
    """Verifica que el oversampling sea reproducible con la misma semilla.

    ## Args:
        imbalanced_train: Conjunto desbalanceado de entrenamiento.
        settings: Configuración con los nombres de columnas y la semilla.
    """
    first_balanced = sampling.oversample_with_unique_ids(
        imbalanced_train, 0.5, settings=settings
    )
    second_balanced = sampling.oversample_with_unique_ids(
        imbalanced_train, 0.5, settings=settings
    )

    assert_frame_equal(first_balanced, second_balanced)


def test_should_use_custom_column_names_in_oversampling(
    imbalanced_train: pl.DataFrame,
    custom_settings: config.Settings,
) -> None:
    """Verifica que se usen los nombres de identificador y target configurados.

    ## Args:
        imbalanced_train: Conjunto desbalanceado de entrenamiento.
        custom_settings: Configuración con los nombres alternativos.
    """
    data = imbalanced_train.rename({"client_id": "customer_id", "Target": "churn"})

    balanced = sampling.oversample_with_unique_ids(data, 0.5, settings=custom_settings)

    assert balanced.columns == data.columns
    assert balanced.filter(pl.col("churn") == 1).height == 30
    assert balanced["customer_id"].n_unique() == balanced.height


def test_should_use_global_settings_when_settings_is_not_provided_for_oversampling(
    imbalanced_train: pl.DataFrame,
    custom_settings: config.Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verifica que el oversampling use la configuración global si no se indica.

    ## Args:
        imbalanced_train: Conjunto desbalanceado de entrenamiento.
        custom_settings: Configuración con los nombres alternativos.
        monkeypatch: Utilidad para reemplazar la configuración global.
    """
    data = imbalanced_train.rename({"client_id": "customer_id", "Target": "churn"})
    monkeypatch.setattr(config, "get_settings", lambda: custom_settings)

    balanced = sampling.oversample_with_unique_ids(data, 0.5)

    assert balanced.filter(pl.col("churn") == 1).height == 30
    assert balanced.filter(pl.col("churn") == 0).height == 30
