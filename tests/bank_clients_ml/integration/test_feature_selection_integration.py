"""Tests de integración de la selección de features."""

import types

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

from bank_clients_ml import (
    binning,
    bivariate_analysis,
    bivariate_table,
    column_groups,
    config,
    redundant_column_filter,
    training,
)


@pytest.fixture
def column_filter(
    train_test_abt: tuple[pl.DataFrame, pl.DataFrame],
    integration_settings: config.Settings,
) -> redundant_column_filter.RedundantColumnFilter:
    """Devuelve el filtro de columnas generado sobre la ABT de integración."""
    train, test = train_test_abt
    return redundant_column_filter.RedundantColumnFilter.generate(
        train=train,
        test=test,
        imbalanced_binary_threshold=0.10,
        correlation_threshold=0.80,
        settings=integration_settings,
    )


@pytest.fixture
def enhanced_train_test(
    train_test_abt: tuple[pl.DataFrame, pl.DataFrame],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Agrega columnas constantes, desbalanceadas y perfectamente correlacionadas."""
    train, test = train_test_abt

    def _augment(frame: pl.DataFrame, offset: int) -> pl.DataFrame:
        return frame.with_columns(
            pl.lit(1.0).alias("integration_constant_feature"),
            pl.when(pl.int_range(0, frame.height) == 0)
            .then(1)
            .otherwise(0)
            .cast(pl.Int8)
            .alias("integration_rare_flag"),
            (pl.int_range(0, frame.height) + offset)
            .cast(pl.Float64)
            .alias("integration_correlated_a"),
        ).with_columns(
            (pl.col("integration_correlated_a") * 2.0 + 1.0).alias(
                "integration_correlated_b"
            )
        )

    return _augment(train, 0), _augment(test, 100)


@pytest.fixture
def enhanced_column_filter(
    enhanced_train_test: tuple[pl.DataFrame, pl.DataFrame],
    integration_settings: config.Settings,
) -> redundant_column_filter.RedundantColumnFilter:
    """Devuelve el filtro generado sobre la ABT con columnas redundantes extra."""
    train, test = enhanced_train_test
    return redundant_column_filter.RedundantColumnFilter.generate(
        train=train,
        test=test,
        imbalanced_binary_threshold=0.10,
        correlation_threshold=0.80,
        settings=integration_settings,
    )


@pytest.fixture
def grouped_train_test(
    column_filter: redundant_column_filter.RedundantColumnFilter,
    integration_settings: config.Settings,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Reduce los conjuntos conservando hasta cuatro features por grupo."""
    grouped = dict(
        column_groups.group_columns_by_source(
            column_filter.correlated_train,
            integration_settings,
        ).items()
    )
    selected_columns = [
        column
        for group, columns in grouped.items()
        if group is not column_groups.ValidColumnGroup.ALL
        for column in columns[:4]
    ]
    required_columns = [
        integration_settings.dataset.id,
        integration_settings.dataset.target,
        *selected_columns,
    ]
    return (
        column_filter.correlated_train.select(required_columns),
        column_filter.correlated_test.select(required_columns),
    )


def test_should_reduce_redundant_columns_and_cover_all_groups(
    enhanced_column_filter: redundant_column_filter.RedundantColumnFilter,
    integration_settings: config.Settings,
) -> None:
    """Verifica la reducción secuencial y la cobertura de grupos.

    ## Args:
        enhanced_column_filter: Filtro generado sobre una ABT con columnas extra.
        integration_settings: Configuración con identificador y target.
    """
    column_filter = enhanced_column_filter

    assert "integration_constant_feature" in column_filter.constant_cols
    assert "integration_rare_flag" in column_filter.imbalanced_binary_columns
    assert "integration_correlated_b" in column_filter.correlated_columns

    assert "integration_constant_feature" not in column_filter.reduced_train.columns
    assert "integration_constant_feature" not in column_filter.correlated_test.columns
    assert "integration_rare_flag" not in column_filter.correlated_train.columns
    assert "integration_rare_flag" not in column_filter.correlated_test.columns
    assert "integration_correlated_b" not in column_filter.uncorrelated_train.columns
    assert "integration_correlated_b" in column_filter.correlated_train.columns
    assert "integration_correlated_b" in column_filter.correlated_test.columns

    assert set(column_filter.uncorrelated_train.columns).issubset(
        set(column_filter.correlated_train.columns)
    )

    grouped = dict(
        column_groups.group_columns_by_source(
            column_filter.uncorrelated_train, integration_settings
        ).items()
    )
    all_columns = grouped[column_groups.ValidColumnGroup.ALL]
    grouped_columns = [
        column
        for group, columns in grouped.items()
        if group is not column_groups.ValidColumnGroup.ALL
        for column in columns
    ]
    assert set(grouped_columns) == set(all_columns)
    assert len(grouped_columns) == len(set(grouped_columns)) == len(all_columns)

    correlation_matrix = (
        column_filter.uncorrelated_train.drop(
            integration_settings.dataset.id,
            integration_settings.dataset.target,
        )
        .corr()
        .to_numpy()
    )
    np.fill_diagonal(correlation_matrix, 0.0)
    assert np.nanmax(np.abs(correlation_matrix)) <= column_filter.correlation_threshold


@pytest.mark.integration_lgbm
def test_should_train_grouped_models_and_select_top_features(
    grouped_train_test: tuple[pl.DataFrame, pl.DataFrame],
    integration_settings: config.Settings,
) -> None:
    """Verifica el entrenamiento por grupos y la selección de sus mejores features.

    ## Args:
        grouped_train_test: Conjuntos reducidos con columnas de todos los grupos.
        integration_settings: Configuración del entrenamiento.
    """
    train, _ = grouped_train_test
    grouped = dict(
        column_groups.group_columns_by_source(train, integration_settings).items()
    )

    result = training.GroupsLGBMTrainer.fit_and_build(
        train,
        integration_settings,
        n_iter=1,
        splits_cross_validation=2,
    )

    assert isinstance(result.trainers, types.MappingProxyType)
    assert set(result.trainers) == set(column_groups.ValidColumnGroup)
    assert result.n_iter == 1
    assert result.splits_cross_validation == 2
    for group, trainer in result.items():
        assert isinstance(trainer, training.LGBMTrainer)
        assert isinstance(trainer.best_model, lgb.LGBMClassifier)
        assert trainer.n_iter == 1
        assert set(trainer.features).issubset(set(grouped[group]))
        assert trainer.number_of_features == len(grouped[group])
        assert trainer.splits_cross_validation == 2
        assert 0.0 <= trainer.best_score <= 1.0

    selected = result.get_top_grouped_features()
    expected_selected = tuple(
        feature
        for group, trainer in result.items()
        for feature in trainer.get_top_ranked_features(group.top_n)
    )
    expected_count = sum(
        group.top_n
        for group in column_groups.ValidColumnGroup
        if group is not column_groups.ValidColumnGroup.ALL and grouped[group]
    )
    assert selected == expected_selected
    assert len(selected) == expected_count
    assert len(set(selected)) == len(selected)

    for group, trainer in result.items():
        top_features = trainer.get_top_ranked_features(group.top_n)
        assert set(top_features).issubset(set(grouped[group]))
        if group is column_groups.ValidColumnGroup.ALL:
            assert top_features == ()


def _build_binners(
    train: pl.DataFrame,
    settings: config.Settings,
    count: int = 2,
) -> tuple[binning.Binner, ...]:
    """Construye binners con al menos dos grupos de bines sobre features reales."""
    candidate_columns = [
        column
        for column in train.columns
        if column not in {settings.dataset.id, settings.dataset.target}
        and train.schema[column].is_numeric()
    ]

    binners: list[binning.Binner] = []
    for column in candidate_columns:
        pretty_name = column.replace("_", " ")
        name_mapping = binning.NameMapping.from_dict({column: pretty_name})
        table = bivariate_table.build_bivariate_tables(
            train,
            name_mapping,
            5,
            settings,
        )[0].table
        if table.height < 2:
            continue

        split = max(1, table.height // 2)
        intervals = [(1, split)]
        if split < table.height:
            intervals.append((split + 1, table.height))
        if len(intervals) < 2:
            continue

        group_percentages = []
        for start, end in intervals:
            group = table.filter(
                pl.col(settings.analysis.bin).is_between(start, end, closed="both")
            )
            clients = group.get_column(settings.analysis.clients).sum()
            positives = group.get_column(settings.dataset.target).sum()
            group_percentages.append(
                0.0 if clients == 0 else float(positives) / float(clients) * 100.0
            )
        if len({round(value, 6) for value in group_percentages}) < 2:
            continue

        binner = binning.Binner(column, pretty_name)
        for start, end in intervals:
            binner.merge_bins(start, end)
        binners.append(binner)
        if len(binners) == count:
            break

    assert len(binners) == count
    return tuple(binners)


@pytest.mark.integration_lgbm
def test_should_bin_selected_features_and_keep_final_columns(
    grouped_train_test: tuple[pl.DataFrame, pl.DataFrame],
    integration_settings: config.Settings,
) -> None:
    """Verifica la codificación por bines y las columnas finales del modelo.

    ## Args:
        grouped_train_test: Conjuntos reducidos con features reales.
        integration_settings: Configuración del proyecto.
    """
    train, test = grouped_train_test
    binners = _build_binners(train, integration_settings)
    analyzer = bivariate_analysis.BivariateAnalyzer(
        correlated_train=train,
        correlated_test=test,
        settings=integration_settings,
    )

    result = analyzer.apply_binning(binners, max_bins_quantity=5)

    expected_columns = [
        integration_settings.dataset.id,
        integration_settings.dataset.target,
        *(binner.snake_case_name for binner in binners),
    ]
    assert result.train.columns == expected_columns
    assert result.test.columns == expected_columns
    assert result.train.height == train.height
    assert result.test.height == test.height

    for frame in (result.train, result.test):
        for binner in binners:
            series = frame.get_column(binner.snake_case_name)
            assert series.null_count() == 0
            assert series.dtype in (pl.Float32, pl.Float64)
            assert series.is_finite().all()

    assert result.name_mapping.snake_case_names == tuple(
        binner.snake_case_name for binner in binners
    )
    assert result.name_mapping.pretty_names == tuple(
        binner.pretty_name for binner in binners
    )
    assert isinstance(result.trainer, training.LGBMTrainer)
    assert result.n_iter == 20


@pytest.mark.integration_lgbm
def test_should_train_features_selection_with_mapped_features_and_explicit_iterations(
    synthetic_train: pl.DataFrame,
    integration_settings: config.Settings,
) -> None:
    """Verifica el entrenamiento de la selección con iteraciones explícitas.

    ## Args:
        synthetic_train: Conjunto sintético con variables y target.
        integration_settings: Configuración del proyecto.
    """
    features = tuple(
        column for column in synthetic_train.columns if column.startswith("Feature_")
    )
    name_mapping = binning.NameMapping.from_dict(
        {feature: feature.replace("_", " ") for feature in features}
    )
    columns = [
        integration_settings.dataset.target,
        *name_mapping.snake_case_names,
    ]
    train = synthetic_train.head(1_500).select(columns)
    test = synthetic_train.tail(500).select(columns)

    result = bivariate_analysis.FeaturesSelection.fit_and_build(
        train=train,
        test=test,
        name_mapping=name_mapping,
        settings=integration_settings,
        n_iter=7,
    )

    assert result.n_iter == 7
    assert result.trainer.n_iter == 7
    assert isinstance(result.trainer.best_model, lgb.LGBMClassifier)
    assert set(result.trainer.features) == set(name_mapping.snake_case_names)
    assert result.snake_case_names == name_mapping.snake_case_names
    assert result.train is train
    assert result.test is test
    assert result.name_mapping is name_mapping
    assert result.settings is integration_settings


@pytest.mark.integration_lgbm
def test_should_run_heavy_operations_and_return_correlations(
    mocker,
    grouped_train_test: tuple[pl.DataFrame, pl.DataFrame],
    integration_settings: config.Settings,
) -> None:
    """Verifica el entrenamiento jerárquico y las correlaciones de las mejores features.

    ## Args:
        mocker: Utilidad para sustituir la generación de gráficos.
        grouped_train_test: Conjuntos reducidos con features reales.
        integration_settings: Configuración del proyecto.
    """
    train, test = grouped_train_test
    correlation_threshold = 0.8
    column_filter = redundant_column_filter.RedundantColumnFilter(
        reduced_train=train,
        correlated_train=train,
        correlated_test=test,
        uncorrelated_train=train,
        constant_cols=(),
        imbalanced_binary_columns=(),
        correlated_columns=(),
        imbalanced_binary_threshold=0.1,
        correlation_threshold=correlation_threshold,
        settings=integration_settings,
        _corr_df=train.drop(
            integration_settings.dataset.id,
            integration_settings.dataset.target,
        ).corr(),
    )
    plot_mock = mocker.patch.object(training.visualization, "plot_top_features")

    group_trainer, trainer, correlations = (
        column_filter.heavy_operations_and_correlations("integration_top_features")
    )

    assert isinstance(group_trainer, training.GroupsLGBMTrainer)
    assert isinstance(trainer, training.LGBMTrainer)
    top_features = trainer.get_top_ranked_features()
    assert set(correlations) == set(top_features)
    for feature, correlation_frame in correlations.items():
        assert correlation_frame.columns == ["feature_name", "correlation"]
        assert (
            correlation_frame.get_column("correlation").abs() > correlation_threshold
        ).all()
        assert feature not in correlation_frame.get_column("feature_name").to_list()

    plotted_names = {call.args[1] for call in plot_mock.call_args_list}
    expected_plot_names = {
        group.snake_case_name for group in column_groups.ValidColumnGroup
    } | {"integration_top_features"}
    assert plotted_names == expected_plot_names
