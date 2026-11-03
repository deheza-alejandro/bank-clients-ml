"""Tests unitarios para el módulo de entrenamiento de modelos LightGBM."""

import types
from typing import cast
from unittest import mock

import lightgbm as lgb
import numpy as np
import polars as pl
import polars.testing
import pytest
import pytest_mock
import sklearn.model_selection

from bank_clients_ml import binning, column_groups, config, training


class _StubSearcher:
    """Buscador de hiperparámetros simulado para los entrenadores."""

    def __init__(
        self,
        *,
        best_score: float = 0.75,
        n_splits: int = 3,
        cv_results: dict[str, list] | None = None,
    ) -> None:
        """Inicializa el buscador con los resultados simulados indicados.

        ## Args:
            best_score: Mejor puntuación de la validación cruzada.
            n_splits: Cantidad de particiones de la validación cruzada.
            cv_results: Resultados de la validación cruzada.
        """
        self.best_score_ = best_score
        self.n_splits_ = n_splits
        self.cv_results_ = cv_results if cv_results is not None else {}


@pytest.fixture
def importances() -> pl.DataFrame:
    """Devuelve una tabla de importancias ordenada de mayor a menor.

    ## Returns:
        Tabla con tres variables y sus importancias.
    """
    return pl.DataFrame(
        {
            "Feature": ["Feature_A", "Feature_B", "Feature_C"],
            "Importance": [0.5, 0.3, 0.2],
        }
    )


@pytest.fixture
def minimal_train() -> pl.DataFrame:
    """Devuelve un entrenamiento mínimo con target binario.

    ## Returns:
        DataFrame con una fila por clase del target.
    """
    return pl.DataFrame({"Target": [0, 1]})


def _build_trainer(
    settings: config.Settings,
    importances: pl.DataFrame,
    searcher: _StubSearcher | None = None,
) -> training.LGBMTrainer:
    """Construye un entrenador con un buscador simulado.

    ## Args:
        settings: Configuración global del proyecto.
        importances: Tabla de importancias ordenada de mayor a menor.
        searcher: Buscador simulado; si no se indica, se usa uno por defecto.

    ## Returns:
        Entrenador listo para consultar sus propiedades.
    """
    return training.LGBMTrainer(
        importances=importances,
        search_logs="captured logs",
        n_iter=2,
        settings=settings,
        _searcher=cast(
            sklearn.model_selection.RandomizedSearchCV,
            searcher if searcher is not None else _StubSearcher(),
        ),
    )


def test_should_build_expected_hyperparameter_distributions() -> None:
    """Verifica que el espacio de búsqueda incluya todos los hiperparámetros."""
    distributions = training._build_param_distributions(12_000, 2)

    assert set(distributions) == {
        "n_estimators",
        "max_depth",
        "num_leaves",
        "subsample",
        "learning_rate",
        "min_child_samples",
    }


@pytest.mark.parametrize(
    ("n_samples", "splits_cross_validation", "expected_max"),
    [
        pytest.param(2_000, 2, 333, id="small-dataset"),
        pytest.param(12_000, 2, 2_000, id="medium-dataset"),
        pytest.param(100_000, 2, 2_900, id="large-dataset-capped"),
    ],
)
def test_should_adapt_min_child_samples_to_fold_size(
    n_samples: int,
    splits_cross_validation: int,
    expected_max: int,
) -> None:
    """Verifica que el rango de hojas mínimas sea factible para los pliegues.

    ## Args:
        n_samples: Cantidad de filas del conjunto de entrenamiento.
        splits_cross_validation: Cantidad de particiones de la validación cruzada.
        expected_max: Valor máximo esperado para `min_child_samples`.
    """
    distributions = training._build_param_distributions(
        n_samples, splits_cross_validation
    )
    min_child_samples = np.asarray(distributions["min_child_samples"])

    fold_size = n_samples * (splits_cross_validation - 1) // splits_cross_validation
    assert min_child_samples.size > 0
    assert int(min_child_samples.max()) == expected_max
    assert int(min_child_samples.max()) * 2 <= fold_size


@pytest.mark.parametrize(
    "splits_cross_validation",
    [pytest.param(2, id="two-folds"), pytest.param(5, id="five-folds")],
)
def test_should_build_cross_validation_with_requested_splits_and_seed(
    settings: config.Settings,
    splits_cross_validation: int,
) -> None:
    """Verifica la configuración de la validación cruzada estratificada.

    ## Args:
        settings: Configuración con la semilla del proyecto.
        splits_cross_validation: Cantidad de particiones solicitada.
    """
    cross_validation = training._build_cross_validation(
        splits_cross_validation, settings
    )

    expected_cv = sklearn.model_selection.StratifiedKFold(
        n_splits=splits_cross_validation,
        shuffle=True,
        random_state=settings.random_state,
    )
    features = np.zeros((2 * splits_cross_validation, 1))
    target = np.repeat([0, 1], splits_cross_validation)

    assert [
        (train.tolist(), test.tolist())
        for train, test in cross_validation.split(features, target)
    ] == [
        (train.tolist(), test.tolist())
        for train, test in expected_cv.split(features, target)
    ]
    assert cross_validation.get_n_splits() == splits_cross_validation


def test_should_build_searcher_with_configured_parallelism() -> None:
    """Verifica la configuración del buscador y su paralelismo."""
    custom_settings = config.Settings(n_jobs=1)

    searcher = training._build_searcher(
        n_iter=2,
        splits_cross_validation=3,
        n_samples=2_000,
        settings=custom_settings,
    )

    assert isinstance(searcher, sklearn.model_selection.RandomizedSearchCV)
    params = searcher.get_params()
    assert params["n_iter"] == 2
    assert params["n_jobs"] == 1
    assert params["random_state"] == custom_settings.random_state
    estimator = params["estimator"]
    assert isinstance(estimator, lgb.LGBMClassifier)
    assert estimator.get_params()["n_jobs"] == 1


def test_should_expose_search_metrics(
    settings: config.Settings,
    importances: pl.DataFrame,
) -> None:
    """Verifica las propiedades públicas de resultados de la búsqueda.

    ## Args:
        settings: Configuración global del proyecto.
        importances: Tabla de importancias ordenada de mayor a menor.
    """
    trainer = _build_trainer(
        settings,
        importances,
        _StubSearcher(best_score=0.82, n_splits=4),
    )

    assert trainer.best_score == 0.82
    assert trainer.splits_cross_validation == 4
    assert trainer.random_state == settings.random_state


@pytest.mark.parametrize(
    "n_iter",
    [pytest.param(0, id="zero"), pytest.param(-1, id="negative")],
)
def test_should_raise_value_error_when_n_iter_is_less_than_one(
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
    minimal_train: pl.DataFrame,
    n_iter: int,
) -> None:
    """Verifica que no se entrene con una cantidad inválida de combinaciones.

    ## Args:
        mocker: Utilidad para comprobar que no se ejecuta la búsqueda.
        settings: Configuración global del proyecto.
        minimal_train: Entrenamiento mínimo con target binario.
        n_iter: Cantidad inválida de combinaciones a evaluar.
    """
    search_mock = mocker.patch.object(training, "_fit_lgbm_random_search")

    with pytest.raises(ValueError, match="Invalid n_iter"):
        training.LGBMTrainer.fit_and_build(minimal_train, [], settings, n_iter=n_iter)

    search_mock.assert_not_called()


@pytest.mark.parametrize(
    "splits_cross_validation",
    [pytest.param(1, id="one-fold"), pytest.param(0, id="zero-folds")],
)
def test_should_raise_value_error_when_splits_cross_validation_is_less_than_two(
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
    minimal_train: pl.DataFrame,
    splits_cross_validation: int,
) -> None:
    """Verifica que no se entrene con una validación cruzada insuficiente.

    ## Args:
        mocker: Utilidad para comprobar que no se ejecuta la búsqueda.
        settings: Configuración global del proyecto.
        minimal_train: Entrenamiento mínimo con target binario.
        splits_cross_validation: Cantidad inválida de particiones.
    """
    search_mock = mocker.patch.object(training, "_fit_lgbm_random_search")

    with pytest.raises(ValueError, match="Invalid splits_cross_validation"):
        training.LGBMTrainer.fit_and_build(
            minimal_train,
            [],
            settings,
            splits_cross_validation=splits_cross_validation,
        )

    search_mock.assert_not_called()


def test_should_return_cv_results_sorted_by_rank_without_params(
    settings: config.Settings,
    importances: pl.DataFrame,
) -> None:
    """Verifica el orden y las columnas de los resultados de validación cruzada.

    ## Args:
        settings: Configuración global del proyecto.
        importances: Tabla de importancias ordenada de mayor a menor.
    """
    cv_results = {
        "rank_test_score": [2, 1],
        "mean_test_score": [0.7, 0.9],
        "std_test_score": [0.01, 0.02],
        "params": [{"max_depth": 4}, {"max_depth": 5}],
    }
    trainer = _build_trainer(
        settings,
        importances,
        _StubSearcher(cv_results=cv_results),
    )

    result = trainer.cv_results

    assert result.columns == ["rank_test_score", "mean_test_score", "std_test_score"]
    assert "params" not in result.columns
    expected = pl.DataFrame(
        {
            "rank_test_score": [1, 2],
            "mean_test_score": [0.9, 0.7],
            "std_test_score": [0.02, 0.01],
        }
    )
    polars.testing.assert_frame_equal(result, expected)


@pytest.mark.parametrize(
    ("top_n", "expected"),
    [
        pytest.param(0, (), id="zero-features"),
        pytest.param(2, ("Feature_A", "Feature_B"), id="two-features"),
        pytest.param(
            3,
            ("Feature_A", "Feature_B", "Feature_C"),
            id="all-features",
        ),
    ],
)
def test_should_return_top_ranked_features_in_importance_order(
    settings: config.Settings,
    importances: pl.DataFrame,
    top_n: int,
    expected: tuple[str, ...],
) -> None:
    """Verifica la selección de las variables mejor posicionadas.

    ## Args:
        settings: Configuración global del proyecto.
        importances: Tabla de importancias ordenada de mayor a menor.
        top_n: Cantidad de variables solicitada.
        expected: Variables esperadas en orden de importancia.
    """
    trainer = _build_trainer(settings, importances)

    result = trainer.get_top_ranked_features(top_n)

    assert result == expected


@pytest.mark.parametrize(
    "top_n",
    [pytest.param(-1, id="negative"), pytest.param(4, id="greater-than-features")],
)
def test_should_raise_value_error_when_top_n_is_out_of_range(
    settings: config.Settings,
    importances: pl.DataFrame,
    top_n: int,
) -> None:
    """Verifica que se rechace una cantidad inválida de variables a seleccionar.

    ## Args:
        settings: Configuración global del proyecto.
        importances: Tabla de importancias ordenada de mayor a menor.
        top_n: Cantidad de variables fuera del rango permitido.
    """
    trainer = _build_trainer(settings, importances)

    with pytest.raises(ValueError, match=f"Invalid top_n {top_n}"):
        trainer.get_top_ranked_features(top_n)


def test_should_plot_importances_without_name_mapping(
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
    importances: pl.DataFrame,
) -> None:
    """Verifica el gráfico de importancias cuando no se indica un mapeo.

    ## Args:
        mocker: Utilidad para sustituir la generación del gráfico.
        settings: Configuración global del proyecto.
        importances: Tabla de importancias ordenada de mayor a menor.
    """
    plot_mock = mocker.patch.object(training.visualization, "plot_top_features")
    trainer = _build_trainer(
        settings,
        importances,
        _StubSearcher(best_score=0.81),
    )

    trainer.plot_top_features("baseline")

    plotted_importances, plot_name = plot_mock.call_args.args
    polars.testing.assert_frame_equal(plotted_importances, importances)
    assert plot_name == "baseline"
    assert plot_mock.call_args.kwargs["roc_auc"] == 0.81
    assert plot_mock.call_args.kwargs["settings"] is settings


def test_should_plot_importances_with_descriptive_feature_names(
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
    importances: pl.DataFrame,
) -> None:
    """Verifica que el mapeo de nombres se aplique solo al gráfico.

    ## Args:
        mocker: Utilidad para sustituir la generación del gráfico.
        settings: Configuración global del proyecto.
        importances: Tabla de importancias ordenada de mayor a menor.
    """
    plot_mock = mocker.patch.object(training.visualization, "plot_top_features")
    trainer = _build_trainer(
        settings,
        importances,
        _StubSearcher(best_score=0.83),
    )
    name_mapping = binning.NameMapping.from_dict(
        {"Feature_A": "Feature A", "Feature_C": "Feature C"}
    )

    trainer.plot_top_features("mapped", name_mapping)

    plotted_importances = plot_mock.call_args.args[0]
    expected = pl.DataFrame(
        {
            "Feature": ["Feature A", "Feature_B", "Feature C"],
            "Importance": [0.5, 0.3, 0.2],
        }
    )
    polars.testing.assert_frame_equal(plotted_importances, expected)
    assert importances.get_column("Feature").to_list() == [
        "Feature_A",
        "Feature_B",
        "Feature_C",
    ]


def test_should_train_one_model_for_each_column_group(
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
) -> None:
    """Verifica que se entrene un modelo distinto por cada grupo de columnas.

    ## Args:
        mocker: Utilidad para sustituir el agrupamiento y el ajuste.
        settings: Configuración global del proyecto.
    """
    groups = {
        column_groups.ValidColumnGroup.SAVING_ACCOUNT_MONETARY: (
            "SavingAccount_Balance_ARG",
        ),
        column_groups.ValidColumnGroup.OTHERS: ("Region", "Email"),
    }
    generated_groups = mocker.Mock()
    generated_groups.items.return_value = tuple(groups.items())
    group_mock = mocker.patch.object(
        training.column_groups,
        "group_columns_by_source",
        return_value=generated_groups,
    )
    trainers_by_columns = {columns: mocker.Mock() for columns in groups.values()}
    fit_mock = mocker.patch.object(
        training.LGBMTrainer,
        "fit_and_build",
        side_effect=(
            lambda _train, columns, _settings, *, n_iter, splits_cross_validation: (
                trainers_by_columns[tuple(columns)]
            )
        ),
    )
    train = pl.DataFrame(
        {
            "SavingAccount_Balance_ARG": [10.0],
            "Region": [1],
            "Email": [1],
            "Target": [0],
        }
    )

    result = training._generate_trainers(
        train,
        n_iter=2,
        splits_cross_validation=3,
        settings=settings,
    )

    assert isinstance(result, types.MappingProxyType)
    expected_trainers = {
        group: trainers_by_columns[columns] for group, columns in groups.items()
    }
    assert dict(result) == expected_trainers
    group_mock.assert_called_once_with(train, settings)
    expected_calls = [
        mocker.call(
            train,
            columns,
            settings,
            n_iter=2,
            splits_cross_validation=3,
        )
        for columns in groups.values()
    ]
    assert fit_mock.call_args_list == expected_calls


def test_should_build_groups_trainer_with_generated_trainers(
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
) -> None:
    """Verifica la construcción del entrenador agrupado y sus atributos.

    ## Args:
        mocker: Utilidad para sustituir la generación de entrenadores.
        settings: Configuración global del proyecto.
    """
    generated_trainers = types.MappingProxyType(
        {column_groups.ValidColumnGroup.OTHERS: mocker.Mock()}
    )
    generate_mock = mocker.patch.object(
        training,
        "_generate_trainers",
        return_value=generated_trainers,
    )
    train = pl.DataFrame({"Feature_A": [1.0], "Target": [0]})

    result = training.GroupsLGBMTrainer.fit_and_build(
        train,
        settings,
        n_iter=4,
        splits_cross_validation=5,
    )

    assert isinstance(result, training.GroupsLGBMTrainer)
    assert result.trainers is generated_trainers
    assert result.n_iter == 4
    assert result.splits_cross_validation == 5
    assert result.settings is settings
    generate_mock.assert_called_once_with(
        train,
        n_iter=4,
        splits_cross_validation=5,
        settings=settings,
    )


def test_should_describe_group_lengths_with_pretty_names(
    settings: config.Settings,
) -> None:
    """Verifica los textos descriptivos con la cantidad de variables por grupo.

    ## Args:
        settings: Configuración global del proyecto.
    """
    trainers = {
        column_groups.ValidColumnGroup.ALL: mock.Mock(number_of_features=10),
        column_groups.ValidColumnGroup.OTHERS: mock.Mock(number_of_features=4),
    }
    groups_trainer = training.GroupsLGBMTrainer(
        types.MappingProxyType(trainers),
        2,
        3,
        settings,
    )

    result = groups_trainer.groups_lengths

    assert result == ("All Columns: 10", "Others: 4")


def test_should_map_every_group_to_its_descriptive_names(
    settings: config.Settings,
) -> None:
    """Verifica el mapeo entre nombres internos y descriptivos de los grupos.

    ## Args:
        settings: Configuración global del proyecto.
    """
    groups_trainer = training.GroupsLGBMTrainer(
        types.MappingProxyType({}),
        2,
        3,
        settings,
    )
    members = column_groups.ValidColumnGroup.members()

    name_mapping = groups_trainer.groups_name_mapping

    assert name_mapping.snake_case_names == tuple(
        member.snake_case_name for member in members
    )
    assert name_mapping.pretty_names == tuple(member.pretty_name for member in members)


def test_should_plot_top_features_for_each_group(
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
) -> None:
    """Verifica que se genere un gráfico por grupo con su nombre interno.

    ## Args:
        mocker: Utilidad para crear los entrenadores simulados.
        settings: Configuración global del proyecto.
    """
    groups = (
        column_groups.ValidColumnGroup.SAVING_ACCOUNT_MONETARY,
        column_groups.ValidColumnGroup.OTHERS,
    )
    trainers = {group: mocker.Mock() for group in groups}
    groups_trainer = training.GroupsLGBMTrainer(
        types.MappingProxyType(trainers),
        2,
        3,
        settings,
    )

    groups_trainer.plot_top_features()

    for group, trainer in trainers.items():
        trainer.plot_top_features.assert_called_once_with(group.snake_case_name)


def test_should_collect_top_ranked_features_for_each_group(
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
) -> None:
    """Verifica la combinación de variables seleccionadas por cada grupo.

    ## Args:
        mocker: Utilidad para crear los entrenadores simulados.
        settings: Configuración global del proyecto.
    """
    group_features = {
        column_groups.ValidColumnGroup.ALL: ("Feature_A",),
        column_groups.ValidColumnGroup.SAVING_ACCOUNT_MONETARY: (
            "SavingAccount_Balance_ARG",
            "SavingAccount_Total_Amount",
        ),
        column_groups.ValidColumnGroup.OTHERS: ("Region", "Client_Age_grp", "Email"),
    }
    trainers = {}
    for group, features in group_features.items():
        trainer = mocker.Mock()
        trainer.get_top_ranked_features.side_effect = lambda top_n, available=features: (
            available[:top_n]
        )
        trainers[group] = trainer
    groups_trainer = training.GroupsLGBMTrainer(
        types.MappingProxyType(trainers),
        2,
        3,
        settings,
    )

    result = groups_trainer.get_top_grouped_features()

    assert result == (
        "SavingAccount_Balance_ARG",
        "Region",
        "Client_Age_grp",
        "Email",
    )
    for group, trainer in trainers.items():
        trainer.get_top_ranked_features.assert_called_once_with(group.top_n)
