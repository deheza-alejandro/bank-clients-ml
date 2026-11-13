"""Constructores de datos usados por la selección de features."""

import polars as pl

from bank_clients_ml import column_groups, config, redundant_column_filter


def build_column_filter(
    train_test_abt: tuple[pl.DataFrame, pl.DataFrame],
    settings: config.Settings,
) -> redundant_column_filter.RedundantColumnFilter:
    """Devuelve el filtro de columnas generado sobre la ABT de integración.

    ## Args:
        train_test_abt: Conjuntos de entrenamiento y prueba de la ABT.
        settings: Configuración con los nombres de las columnas.

    ## Returns:
        Filtro con las columnas redundantes detectadas.
    """
    train, test = train_test_abt
    return redundant_column_filter.RedundantColumnFilter.generate(
        train=train,
        test=test,
        imbalanced_binary_threshold=0.10,
        correlation_threshold=0.80,
        settings=settings,
    )


def build_enhanced_train_test(
    train_test_abt: tuple[pl.DataFrame, pl.DataFrame],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Agrega columnas constantes, desbalanceadas y perfectamente correlacionadas.

    ## Args:
        train_test_abt: Conjuntos de entrenamiento y prueba de la ABT.

    ## Returns:
        Conjuntos de entrenamiento y prueba con columnas redundantes extra.
    """
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


def build_enhanced_column_filter(
    enhanced_train_test: tuple[pl.DataFrame, pl.DataFrame],
    settings: config.Settings,
) -> redundant_column_filter.RedundantColumnFilter:
    """Devuelve el filtro generado sobre la ABT con columnas redundantes extra.

    ## Args:
        enhanced_train_test: Conjuntos con columnas redundantes adicionales.
        settings: Configuración con los nombres de las columnas.

    ## Returns:
        Filtro con las columnas redundantes detectadas.
    """
    train, test = enhanced_train_test
    return redundant_column_filter.RedundantColumnFilter.generate(
        train=train,
        test=test,
        imbalanced_binary_threshold=0.10,
        correlation_threshold=0.80,
        settings=settings,
    )


def build_grouped_train_test(
    column_filter: redundant_column_filter.RedundantColumnFilter,
    settings: config.Settings,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Reduce los conjuntos conservando hasta cuatro features por grupo.

    ## Args:
        column_filter: Filtro con las columnas redundantes ya detectadas.
        settings: Configuración con los nombres de las columnas.

    ## Returns:
        Conjuntos reducidos de entrenamiento y prueba.
    """
    grouped = dict(
        column_groups.group_columns_by_source(
            column_filter.correlated_train,
            settings,
        ).items()
    )
    selected_columns = [
        column
        for group, columns in grouped.items()
        if group is not column_groups.ValidColumnGroup.ALL
        for column in columns[:4]
    ]
    required_columns = [
        settings.dataset.id,
        settings.dataset.target,
        *selected_columns,
    ]
    return (
        column_filter.correlated_train.select(required_columns),
        column_filter.correlated_test.select(required_columns),
    )
