"""Tests unitarios para el módulo de entrenamiento de modelos LightGBM."""

from typing import cast

import numpy as np
import polars as pl
import polars.testing
import pytest
import pytest_mock
import sklearn.model_selection

from bank_clients_ml import binning, config, training


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
    "n_iter",
    [pytest.param(0, id="zero"), pytest.param(-1, id="negative")],
)
def test_should_raise_value_error_when_n_iter_is_less_than_one(
    settings: config.Settings,
    minimal_train: pl.DataFrame,
    n_iter: int,
) -> None:
    """Verifica que no se entrene con una cantidad inválida de combinaciones.

    ## Args:
        settings: Configuración global del proyecto.
        minimal_train: Entrenamiento mínimo con target binario.
        n_iter: Cantidad inválida de combinaciones a evaluar.
    """
    with pytest.raises(ValueError, match="Invalid n_iter"):
        training.LGBMTrainer.fit_and_build(minimal_train, [], settings, n_iter=n_iter)


@pytest.mark.parametrize(
    "splits_cross_validation",
    [pytest.param(1, id="one-fold"), pytest.param(0, id="zero-folds")],
)
def test_should_raise_value_error_when_splits_cross_validation_is_less_than_two(
    settings: config.Settings,
    minimal_train: pl.DataFrame,
    splits_cross_validation: int,
) -> None:
    """Verifica que no se entrene con una validación cruzada insuficiente.

    ## Args:
        settings: Configuración global del proyecto.
        minimal_train: Entrenamiento mínimo con target binario.
        splits_cross_validation: Cantidad inválida de particiones.
    """
    with pytest.raises(ValueError, match="Invalid splits_cross_validation"):
        training.LGBMTrainer.fit_and_build(
            minimal_train,
            [],
            settings,
            splits_cross_validation=splits_cross_validation,
        )


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
