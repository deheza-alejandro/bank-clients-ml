"""Tests unitarios para el módulo de muestreo y ventanas temporales."""

from collections.abc import Callable
from datetime import date
from typing import cast

import polars as pl
import polars.testing
import pytest

import tests.builders.unit.sampling
from bank_clients_ml import config, sampling


@pytest.mark.parametrize(
    ("prediction_window_size", "expected_training", "expected_prediction"),
    [
        pytest.param(
            1,
            tuple(date(2024, month, 1) for month in range(1, 9)),
            (date(2024, 10, 1),),
            id="one-month",
        ),
        pytest.param(
            2,
            tuple(date(2024, month, 1) for month in range(1, 8)),
            (date(2024, 9, 1), date(2024, 10, 1)),
            id="two-months",
        ),
        pytest.param(
            3,
            tuple(date(2024, month, 1) for month in range(1, 7)),
            (date(2024, 8, 1), date(2024, 9, 1), date(2024, 10, 1)),
            id="three-months",
        ),
    ],
)
def test_should_build_training_and_prediction_windows_with_one_lead_month(
    settings: config.Settings,
    prediction_window_size: int,
    expected_training: tuple[date, ...],
    expected_prediction: tuple[date, ...],
) -> None:
    """Verifica que las ventanas respeten el tamaño y dejen un mes de separación.

    ## Args:
        settings: Configuración con el nombre de la columna de meses.
        prediction_window_size: Cantidad de meses reservados para predicción.
        expected_training: Meses esperados en la ventana de entrenamiento.
        expected_prediction: Meses esperados en la ventana de predicción.
    """
    clean_data = tests.builders.unit.sampling.build_clean_data()
    split = sampling.TimeWindowSplit.generate(
        clean_data, prediction_window_size, settings=settings
    )

    lead_month = date(2024, 10 - prediction_window_size, 1)
    assert split.training_months == expected_training
    assert split.prediction_months == expected_prediction
    assert split.training_months[-1] < lead_month < split.prediction_months[0]


def test_should_build_expected_windows_and_expose_boundary_months(
    settings: config.Settings,
) -> None:
    """Verifica las ventanas generadas y los meses límite expuestos.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    time_window_split = tests.builders.unit.sampling.build_time_window_split(
        tests.builders.unit.sampling.build_clean_data(), settings
    )
    split = time_window_split

    expected_training = tuple(date(2024, month, 1) for month in range(1, 8))
    expected_prediction = (date(2024, 9, 1), date(2024, 10, 1))
    assert split.training_months == expected_training
    assert split.prediction_months == expected_prediction
    assert split.last_training_month == date(2024, 7, 1)
    assert split.first_prediction_month == date(2024, 9, 1)


@pytest.mark.parametrize("prediction_window_size", [0, -1])
def test_should_raise_value_error_if_prediction_window_size_is_less_than_one(
    settings: config.Settings,
    prediction_window_size: int,
) -> None:
    """Verifica que se rechacen tamaños de ventana de predicción inválidos.

    ## Args:
        settings: Configuración con los nombres de las columnas.
        prediction_window_size: Tamaño de ventana inválido.
    """
    clean_data = tests.builders.unit.sampling.build_clean_data()
    with pytest.raises(ValueError, match="Invalid prediction_window_size"):
        sampling.TimeWindowSplit.generate(
            clean_data, prediction_window_size, settings=settings
        )


def test_should_select_only_clients_with_nine_months_and_valid_flags(
    settings: config.Settings,
) -> None:
    """Verifica que el universo solo incluya clientes elegibles.

    Los clientes deben tener nueve meses y no tener paquete ni tarjeta
    co-branding en el último mes de entrenamiento.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    time_window_split = tests.builders.unit.sampling.build_time_window_split(
        tests.builders.unit.sampling.build_clean_data(), settings
    )
    split = time_window_split

    assert split.universe_and_target.columns == ["client_id", "Target"]
    assert split.universe_and_target.height == 2
    assert set(split.universe_and_target["client_id"].to_list()) == {1, 2}
    assert split.universe_and_target.sort("client_id")["Target"].to_list() == [
        1,
        0,
    ]


def test_should_keep_full_history_for_selected_clients(
    settings: config.Settings,
) -> None:
    """Verifica que los datos del universo conserven toda la historia.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    time_window_split = tests.builders.unit.sampling.build_time_window_split(
        tests.builders.unit.sampling.build_clean_data(), settings
    )
    split = time_window_split

    expected_months = {date(2024, month, 1) for month in range(1, 11)}
    assert split.universe_and_target_data.height == 18
    assert set(split.universe_and_target_data["client_id"].to_list()) == {1, 2}
    assert set(split.universe_and_target_data["Month"].to_list()) == expected_months


def test_should_keep_a_single_target_column_in_universe_data(
    settings: config.Settings,
) -> None:
    """Verifica que el target no se duplique al unir el universo con la historia.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    time_window_split = tests.builders.unit.sampling.build_time_window_split(
        tests.builders.unit.sampling.build_clean_data(), settings
    )
    split = time_window_split

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
    settings: config.Settings,
) -> None:
    """Verifica que el entrenamiento use solo la ventana de entrenamiento.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    time_window_split = tests.builders.unit.sampling.build_time_window_split(
        tests.builders.unit.sampling.build_clean_data(), settings
    )
    split = time_window_split

    assert split.training_data.height == 12
    assert split.training_data["client_id"].n_unique() == 2
    assert set(split.training_data["Month"].to_list()) == set(split.training_months)
    assert split.training_data["Month"].max() == split.last_training_month


def test_should_restrict_prediction_data_to_prediction_months(
    settings: config.Settings,
) -> None:
    """Verifica que la predicción use solo la ventana de predicción.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    time_window_split = tests.builders.unit.sampling.build_time_window_split(
        tests.builders.unit.sampling.build_clean_data(), settings
    )
    split = time_window_split

    assert split.prediction_data.height == 4
    assert split.prediction_data["client_id"].n_unique() == 2
    assert set(split.prediction_data["Month"].to_list()) == set(split.prediction_months)
    assert split.prediction_data["Month"].min() == split.first_prediction_month


def test_should_exclude_lead_month_from_training_and_prediction_data(
    settings: config.Settings,
) -> None:
    """Verifica que el mes intermedio no aparezca en ninguno de los conjuntos.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    time_window_split = tests.builders.unit.sampling.build_time_window_split(
        tests.builders.unit.sampling.build_clean_data(), settings
    )
    split = time_window_split

    lead_month = date(2024, 8, 1)
    assert lead_month not in split.training_months
    assert lead_month not in split.prediction_months
    assert lead_month not in split.training_data["Month"].to_list()
    assert lead_month not in split.prediction_data["Month"].to_list()


def test_should_take_universe_target_from_prediction_months(
    settings: config.Settings,
) -> None:
    """Verifica que el target del universo provenga de la ventana de predicción.

    El cliente 1 tiene target 0 en entrenamiento y 1 en predicción, por lo que
    el universo debe conservar el valor 1.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    clean_data = tests.builders.unit.sampling.build_clean_data()
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


def test_should_return_one_row_per_client_when_target_varies_across_prediction_months(
    settings: config.Settings,
) -> None:
    """Verifica que el universo agregue el target variable de predicción.

    Si el cliente tiene target 1 en al menos un mes de predicción, debe quedar
    una única fila con target 1. Si no tiene ningún target 1, debe quedar una
    única fila con target 0.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    clean_data = tests.builders.unit.sampling.build_clean_data()
    data = clean_data.with_columns(
        pl.when((pl.col("client_id") == 1) & (pl.col("Month") == date(2024, 10, 1)))
        .then(pl.lit(0, dtype=pl.Int64))
        .otherwise(pl.col("Target"))
        .alias("Target")
    )

    split = sampling.TimeWindowSplit.generate(data, 2, settings=settings)

    universe = split.universe_and_target

    assert universe.height == universe["client_id"].n_unique()
    assert universe.sort("client_id")["Target"].to_list() == [1, 0]


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
    settings: config.Settings,
) -> None:
    """Verifica el resultado vacío cuando ningún cliente cumple el historial.

    ## Args:
        settings: Configuración con los nombres de las columnas.
    """
    clean_data = tests.builders.unit.sampling.build_clean_data()
    data = clean_data.filter(pl.col("client_id") == 5)

    split = sampling.TimeWindowSplit.generate(data, 2, settings=settings)

    assert split.universe_and_target.height == 0
    assert split.universe_and_target_data.height == 0
    assert split.training_data.height == 0
    assert split.prediction_data.height == 0
    assert set(split.training_data.columns) == set(data.columns)


def test_should_use_custom_column_names_in_time_window_split(
    custom_dataset_settings: config.Settings,
) -> None:
    """Verifica que se respeten los nombres de columna configurados.

    ## Args:
        custom_dataset_settings: Configuración con los nombres alternativos.
    """
    renamed_clean_data = tests.builders.unit.sampling.build_renamed_clean_data(
        tests.builders.unit.sampling.build_clean_data()
    )
    split = sampling.TimeWindowSplit.generate(
        renamed_clean_data, 2, settings=custom_dataset_settings
    )

    assert split.universe_and_target.columns == ["customer_id", "churn"]
    assert set(split.universe_and_target["customer_id"].to_list()) == {1, 2}
    assert "churn" in split.training_data.columns
    assert "customer_id" in split.prediction_data.columns


def test_should_use_global_settings_when_settings_is_not_provided_for_time_windows(
    custom_dataset_settings: config.Settings,
    patch_global_settings: Callable[[config.Settings], None],
) -> None:
    """Verifica que se use la configuración global si no se recibe una explícita.

    ## Args:
        custom_dataset_settings: Configuración con los nombres alternativos.
        patch_global_settings: Helper para reemplazar la configuración global.
    """
    renamed_clean_data = tests.builders.unit.sampling.build_renamed_clean_data(
        tests.builders.unit.sampling.build_clean_data()
    )
    patch_global_settings(custom_dataset_settings)

    split = sampling.TimeWindowSplit.generate(renamed_clean_data, 2)

    assert split.universe_and_target.columns == ["customer_id", "churn"]
    assert set(split.universe_and_target["customer_id"].to_list()) == {1, 2}


@pytest.mark.parametrize("test_ratio", [0.0, 1.0, -0.1, 1.1])
def test_should_raise_value_error_if_test_ratio_is_out_of_open_interval(
    settings: config.Settings,
    test_ratio: float,
) -> None:
    """Verifica que se rechacen proporciones de prueba inválidas.

    ## Args:
        settings: Configuración con el nombre del target.
        test_ratio: Proporción de prueba inválida.
    """
    abt_data = tests.builders.unit.sampling.build_abt_data()
    with pytest.raises(ValueError, match="Invalid test_ratio"):
        sampling.stratified_train_test_split(abt_data, test_ratio, settings=settings)


def test_should_split_each_class_according_to_test_ratio(
    settings: config.Settings,
) -> None:
    """Verifica que la partición conserve la proporción de cada clase.

    ## Args:
        settings: Configuración con el nombre del target.
    """
    abt_data = tests.builders.unit.sampling.build_abt_data()
    train, test = sampling.stratified_train_test_split(abt_data, 0.3, settings=settings)

    assert train.height == 28
    assert test.height == 12
    assert train.filter(pl.col("Target") == 0).height == 14
    assert train.filter(pl.col("Target") == 1).height == 14
    assert test.filter(pl.col("Target") == 0).height == 6
    assert test.filter(pl.col("Target") == 1).height == 6
    assert test.height / abt_data.height == pytest.approx(0.3)


def test_should_preserve_all_rows_after_split(
    settings: config.Settings,
) -> None:
    """Verifica que la partición no pierda ni duplique filas.

    ## Args:
        settings: Configuración con el nombre del target.
    """
    abt_data = tests.builders.unit.sampling.build_abt_data()
    train, test = sampling.stratified_train_test_split(abt_data, 0.3, settings=settings)

    combined = pl.concat([train, test]).sort("client_id")
    expected = abt_data.sort("client_id")
    polars.testing.assert_frame_equal(combined, expected)


def test_should_return_only_original_columns_after_split(
    settings: config.Settings,
) -> None:
    """Verifica que se eliminen las columnas internas de la partición.

    ## Args:
        settings: Configuración con el nombre del target.
    """
    abt_data = tests.builders.unit.sampling.build_abt_data()
    train, test = sampling.stratified_train_test_split(abt_data, 0.3, settings=settings)

    assert train.columns == abt_data.columns
    assert test.columns == abt_data.columns


def test_should_produce_same_split_for_same_random_state(
    settings: config.Settings,
) -> None:
    """Verifica que la partición sea reproducible con la misma semilla.

    ## Args:
        settings: Configuración con el nombre del target y la semilla.
    """
    abt_data = tests.builders.unit.sampling.build_abt_data()
    first_train, first_test = sampling.stratified_train_test_split(
        abt_data, 0.3, settings=settings
    )
    second_train, second_test = sampling.stratified_train_test_split(
        abt_data, 0.3, settings=settings
    )

    polars.testing.assert_frame_equal(first_train, second_train)
    polars.testing.assert_frame_equal(first_test, second_test)


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
    custom_dataset_settings: config.Settings,
) -> None:
    """Verifica que se use el nombre de target configurado.

    ## Args:
        custom_dataset_settings: Configuración con el target alternativo.
    """
    abt_data = tests.builders.unit.sampling.build_abt_data()
    data = abt_data.rename({"client_id": "customer_id", "Target": "churn"})

    train, test = sampling.stratified_train_test_split(
        data, 0.3, settings=custom_dataset_settings
    )

    assert train.columns == data.columns
    assert test.height == 12
    assert test.filter(pl.col("churn") == 0).height == 6
    assert test.filter(pl.col("churn") == 1).height == 6


def test_should_use_global_settings_when_settings_is_not_provided_for_split(
    custom_dataset_settings: config.Settings,
    patch_global_settings: Callable[[config.Settings], None],
) -> None:
    """Verifica que la partición use la configuración global si no se indica.

    ## Args:
        custom_dataset_settings: Configuración con el target alternativo.
        patch_global_settings: Helper para reemplazar la configuración global.
    """
    abt_data = tests.builders.unit.sampling.build_abt_data()
    data = abt_data.rename({"client_id": "customer_id", "Target": "churn"})
    patch_global_settings(custom_dataset_settings)

    train, test = sampling.stratified_train_test_split(data, 0.3)

    assert train.height + test.height == data.height
    assert test.height == 12


@pytest.mark.parametrize("target_proportion", [0.0, 1.0, -0.1, 1.1])
def test_should_raise_value_error_if_target_proportion_is_out_of_open_interval(
    settings: config.Settings,
    target_proportion: float,
) -> None:
    """Verifica que se rechacen proporciones objetivo inválidas.

    ## Args:
        settings: Configuración con los nombres de columnas.
        target_proportion: Proporción objetivo inválida.
    """
    imbalanced_train = tests.builders.unit.sampling.build_imbalanced_train()
    with pytest.raises(ValueError, match="Invalid target_proportion"):
        sampling.oversample_with_unique_ids(
            imbalanced_train, target_proportion, settings=settings
        )


def test_should_keep_all_majority_rows_unchanged(
    settings: config.Settings,
) -> None:
    """Verifica que la clase mayoritaria se conserve intacta.

    ## Args:
        settings: Configuración con los nombres de columnas.
    """
    imbalanced_train = tests.builders.unit.sampling.build_imbalanced_train()
    balanced = sampling.oversample_with_unique_ids(
        imbalanced_train, 0.5, settings=settings
    )

    expected_majority = imbalanced_train.filter(pl.col("Target") == 0).sort("client_id")
    actual_majority = balanced.filter(pl.col("Target") == 0).sort("client_id")
    polars.testing.assert_frame_equal(actual_majority, expected_majority)


@pytest.mark.parametrize(
    ("target_proportion", "expected_minority_count"),
    [
        pytest.param(0.5, 30, id="half"),
        pytest.param(0.4, 20, id="forty-percent"),
        pytest.param(0.25, 10, id="quarter"),
    ],
)
def test_should_oversample_minority_class_to_requested_proportion(
    settings: config.Settings,
    target_proportion: float,
    expected_minority_count: int,
) -> None:
    """Verifica que la clase minoritaria alcance la proporción solicitada.

    ## Args:
        settings: Configuración con los nombres de columnas.
        target_proportion: Proporción objetivo de la clase minoritaria.
        expected_minority_count: Cantidad esperada de filas minoritarias.
    """
    imbalanced_train = tests.builders.unit.sampling.build_imbalanced_train()
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
    settings: config.Settings,
) -> None:
    """Verifica que el oversampling asigne identificadores únicos.

    ## Args:
        settings: Configuración con los nombres de columnas.
    """
    imbalanced_train = tests.builders.unit.sampling.build_imbalanced_train()
    balanced = sampling.oversample_with_unique_ids(
        imbalanced_train, 0.5, settings=settings
    )

    max_original_id = cast(int, imbalanced_train["client_id"].max())
    minority_ids = balanced.filter(pl.col("Target") == 1)["client_id"]
    assert balanced["client_id"].n_unique() == balanced.height
    assert cast(int, minority_ids.min()) > max_original_id


def test_should_preserve_columns_and_schema_after_oversampling(
    settings: config.Settings,
) -> None:
    """Verifica que el oversampling conserve columnas y tipos de datos.

    ## Args:
        settings: Configuración con los nombres de columnas.
    """
    imbalanced_train = tests.builders.unit.sampling.build_imbalanced_train()
    balanced = sampling.oversample_with_unique_ids(
        imbalanced_train, 0.5, settings=settings
    )

    assert balanced.columns == imbalanced_train.columns
    assert balanced.schema == imbalanced_train.schema


def test_should_produce_same_balanced_data_for_same_random_state(
    settings: config.Settings,
) -> None:
    """Verifica que el oversampling sea reproducible con la misma semilla.

    ## Args:
        settings: Configuración con los nombres de columnas y la semilla.
    """
    imbalanced_train = tests.builders.unit.sampling.build_imbalanced_train()
    first_balanced = sampling.oversample_with_unique_ids(
        imbalanced_train, 0.5, settings=settings
    )
    second_balanced = sampling.oversample_with_unique_ids(
        imbalanced_train, 0.5, settings=settings
    )

    polars.testing.assert_frame_equal(first_balanced, second_balanced)


def test_should_use_custom_column_names_in_oversampling(
    custom_dataset_settings: config.Settings,
) -> None:
    """Verifica que se usen los nombres de identificador y target configurados.

    ## Args:
        custom_dataset_settings: Configuración con los nombres alternativos.
    """
    imbalanced_train = tests.builders.unit.sampling.build_imbalanced_train()
    data = imbalanced_train.rename({"client_id": "customer_id", "Target": "churn"})

    balanced = sampling.oversample_with_unique_ids(
        data, 0.5, settings=custom_dataset_settings
    )

    assert balanced.columns == data.columns
    assert balanced.filter(pl.col("churn") == 1).height == 30
    assert balanced["customer_id"].n_unique() == balanced.height


def test_should_use_global_settings_when_settings_is_not_provided_for_oversampling(
    custom_dataset_settings: config.Settings,
    patch_global_settings: Callable[[config.Settings], None],
) -> None:
    """Verifica que el oversampling use la configuración global si no se indica.

    ## Args:
        custom_dataset_settings: Configuración con los nombres alternativos.
        patch_global_settings: Helper para reemplazar la configuración global.
    """
    imbalanced_train = tests.builders.unit.sampling.build_imbalanced_train()
    data = imbalanced_train.rename({"client_id": "customer_id", "Target": "churn"})
    patch_global_settings(custom_dataset_settings)

    balanced = sampling.oversample_with_unique_ids(data, 0.5)

    assert balanced.filter(pl.col("churn") == 1).height == 30
    assert balanced.filter(pl.col("churn") == 0).height == 30
