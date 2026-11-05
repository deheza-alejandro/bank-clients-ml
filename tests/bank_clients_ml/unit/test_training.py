"""Tests unitarios para el módulo de entrenamiento de modelos LightGBM."""

from typing import Any

import numpy as np
import polars as pl
import polars.testing
import pytest
import pytest_mock

from bank_clients_ml import binning, config, training


class _StubClassifier:
    """Clasificador simulado con importancias configurables."""

    def __init__(self, feature_importances: list[float]) -> None:
        """Inicializa el clasificador con las importancias indicadas.

        ## Args:
            feature_importances: Importancias asociadas a las columnas entrenadas.
        """
        self.feature_importances_ = feature_importances


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
        self.best_estimator_ = _StubClassifier([])

    def fit(self, _x: object, _y: object) -> "_StubSearcher":
        """Simula el ajuste del buscador sin ejecutar un entrenamiento real."""
        return self


def _train_frame(
    columns: list[str],
    target: str,
    height: int = 4,
) -> pl.DataFrame:
    """Construye un DataFrame mínimo con las columnas y el target indicados."""
    return pl.DataFrame(
        {
            **{column: [float(index) for index in range(height)] for column in columns},
            target: [index % 2 for index in range(height)],
        }
    )


def _patch_searcher(
    mocker: pytest_mock.MockerFixture,
    searcher: _StubSearcher,
) -> Any:
    """Reemplaza el buscador de scikit-learn por un doble controlado."""
    return mocker.patch.object(
        training.sklearn.model_selection,
        "RandomizedSearchCV",
        return_value=searcher,
    )


def _build_trainer(
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
    importances: pl.DataFrame,
    searcher: _StubSearcher | None = None,
) -> training.LGBMTrainer:
    """Construye un entrenador mediante la API pública y un buscador simulado.

    ## Args:
        mocker: Utilidad para sustituir el buscador de scikit-learn.
        settings: Configuración global del proyecto.
        importances: Tabla de importancias ordenada de mayor a menor.
        searcher: Buscador simulado; si no se indica, se usa uno por defecto.

    ## Returns:
        Entrenador listo para consultar sus propiedades.
    """
    searcher = searcher if searcher is not None else _StubSearcher()
    searcher.best_estimator_ = _StubClassifier(
        importances.get_column(settings.analysis.importance).to_list()
    )
    columns = importances.get_column(settings.analysis.feature).to_list()
    _patch_searcher(mocker, searcher)

    return training.LGBMTrainer.fit_and_build(
        _train_frame(columns, settings.dataset.target),
        columns,
        settings,
        n_iter=2,
        splits_cross_validation=3,
    )


def _search_param_distributions(
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
    n_samples: int,
    splits_cross_validation: int,
) -> dict[str, Any]:
    """Obtiene las distribuciones pasadas al buscador mediante la API pública."""
    searcher = _StubSearcher()
    searcher.best_estimator_ = _StubClassifier([1.0])
    search_factory = _patch_searcher(mocker, searcher)

    training.LGBMTrainer.fit_and_build(
        _train_frame(["Feature"], settings.dataset.target, height=n_samples),
        ["Feature"],
        settings,
        n_iter=1,
        splits_cross_validation=splits_cross_validation,
    )

    search_call = search_factory.call_args
    assert search_call is not None
    return search_call.kwargs["param_distributions"]


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


def test_should_build_expected_hyperparameter_distributions(
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
) -> None:
    """Verifica que el espacio de búsqueda incluya todos los hiperparámetros.

    ## Args:
        mocker: Utilidad para inspeccionar el buscador construido.
        settings: Configuración global del proyecto.
    """
    distributions = _search_param_distributions(mocker, settings, 12_000, 2)

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
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
    n_samples: int,
    splits_cross_validation: int,
    expected_max: int,
) -> None:
    """Verifica que el rango de hojas mínimas sea factible para los pliegues.

    ## Args:
        mocker: Utilidad para inspeccionar el buscador construido.
        settings: Configuración global del proyecto.
        n_samples: Cantidad de filas del conjunto de entrenamiento.
        splits_cross_validation: Cantidad de particiones de la validación cruzada.
        expected_max: Valor máximo esperado para `min_child_samples`.
    """
    distributions = _search_param_distributions(
        mocker, settings, n_samples, splits_cross_validation
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
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
    importances: pl.DataFrame,
) -> None:
    """Verifica el orden y las columnas de los resultados de validación cruzada.

    ## Args:
        mocker: Utilidad para sustituir el buscador de scikit-learn.
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
        mocker,
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
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
    importances: pl.DataFrame,
    top_n: int,
    expected: tuple[str, ...],
) -> None:
    """Verifica la selección de las variables mejor posicionadas.

    ## Args:
        mocker: Utilidad para sustituir el buscador de scikit-learn.
        settings: Configuración global del proyecto.
        importances: Tabla de importancias ordenada de mayor a menor.
        top_n: Cantidad de variables solicitada.
        expected: Variables esperadas en orden de importancia.
    """
    trainer = _build_trainer(mocker, settings, importances)

    result = trainer.get_top_ranked_features(top_n)

    assert result == expected


@pytest.mark.parametrize(
    "top_n",
    [pytest.param(-1, id="negative"), pytest.param(4, id="greater-than-features")],
)
def test_should_raise_value_error_when_top_n_is_out_of_range(
    mocker: pytest_mock.MockerFixture,
    settings: config.Settings,
    importances: pl.DataFrame,
    top_n: int,
) -> None:
    """Verifica que se rechace una cantidad inválida de variables a seleccionar.

    ## Args:
        mocker: Utilidad para sustituir el buscador de scikit-learn.
        settings: Configuración global del proyecto.
        importances: Tabla de importancias ordenada de mayor a menor.
        top_n: Cantidad de variables fuera del rango permitido.
    """
    trainer = _build_trainer(mocker, settings, importances)

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
        mocker,
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
        mocker,
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
