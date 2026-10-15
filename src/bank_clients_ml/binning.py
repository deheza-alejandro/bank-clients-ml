"""Agrupación de bines para codificación por porcentaje de target.

Reúne utilidades para combinar los bines del análisis bivariado en grupos
con un mismo porcentaje de target y generar expresiones de Polars que
reemplazan los valores originales por dicho porcentaje.

## Clases exportadas:
    Binner: Asociación entre una feature y los intervalos de bines a agrupar.
    NameMapping: Mapeo entre nombres internos y nombres descriptivos.
"""

import dataclasses
import types
from collections.abc import ItemsView, Iterable, Mapping
from typing import NamedTuple, Protocol, Self

import polars as pl

from bank_clients_ml import config


class _GroupedBins(NamedTuple):
    """Intervalo agregado junto con su porcentaje de target.

    ## Attributes:
        interval_expr: Expresión que identifica las filas pertenecientes al intervalo.
        target_pct: Porcentaje de target calculado para el grupo.
    """

    interval_expr: pl.Expr
    target_pct: float


def _build_interval_expr(
    feature: str, group: pl.DataFrame, settings: config.Settings
) -> pl.Expr:
    """Construye la condición de pertenencia al intervalo del grupo.

    Calcula el mínimo y el máximo agregados de los bines del grupo y genera
    una expresión que verifica si el valor de la columna se encuentra dentro
    del intervalo cerrado resultante.

    ## Args:
        feature: Nombre de la feature sobre la que se construye la condición.
        group: Subconjunto de la tabla de análisis bivariado con los bines del grupo.
        settings: Configuración con los nombres de las columnas de análisis.

    ## Returns:
        Expresión de pertenencia al intervalo con ambos extremos incluidos,
        o el literal `False` cuando el grupo no define límites (por ejemplo, si solo
        agrupa bines de nulos).
    """
    min = group.get_column(settings.analysis.min).min()
    max = group.get_column(settings.analysis.max).max()

    if min is None or max is None:
        return pl.lit(False)
    return pl.col(feature).is_between(min, max, closed="both")


def _calculate_target_pct(
    bivariate_table: pl.DataFrame, settings: config.Settings
) -> float:
    """Calcula el porcentaje de target de un grupo de bines.

    Divide el total de casos positivos entre el total de clientes del grupo
    y expresa el resultado en porcentaje.

    ## Args:
        bivariate_table: Tabla con un grupo de bines y las columnas de
            `settings.analysis.clients` y `settings.dataset.target`.
        settings: Configuración con los nombres de las columnas de análisis.

    ## Returns:
        Porcentaje de target del grupo, o `0.0` si el grupo está vacío.
    """
    clients = float(bivariate_table.get_column(settings.analysis.clients).sum())
    target = float(bivariate_table.get_column(settings.dataset.target).sum())
    return target / clients * 100.0 if clients > 0.0 else 0.0


@dataclasses.dataclass(frozen=True, slots=True)
class _BinInterval:
    """Intervalo inclusivo de bines que se agrupan en un mismo valor.

    Representa un intervalo cerrado sobre la numeración de bines generada por
    el análisis bivariado. Los bines comprendidos entre ambos extremos reciben
    el mismo porcentaje de target durante la transformación.

    ## Attributes:
        start: Número de bin inicial del intervalo.
        end: Número de bin final del intervalo.

    ## Raises:
        ValueError: Si `start` o `end` son menores a 1, o si `start` es mayor
            que `end`.
    """

    start: int
    end: int

    def __post_init__(self):
        """Valida los extremos del intervalo."""
        if self.start < 1 or self.end < 1:
            msg = f"Both 'start' and 'end' must be >= 1. start: {self.start}, end: {self.end}"
            raise ValueError(msg)
        if self.start > self.end:
            msg = f"'start' must be <= 'end'. start: {self.start}, end: {self.end}"
            raise ValueError(msg)

    @property
    def range(self) -> range:
        """Números de bin incluidos en el intervalo."""
        return range(self.start, self.end + 1)

    def exceeds(self, value: int) -> bool:
        """Indica si el intervalo supera la cantidad indicada.

        ## Args:
            value: Cantidad a evaluar.

        ## Returns:
            `True` cuando el extremo final excede `value`, `False` en caso contrario.
        """
        return self.end > value

    def to_expr(self, settings: config.Settings) -> pl.Expr:
        """Construye una expresión que filtra los bines del intervalo.

        ## Args:
            settings: Configuración con el nombre de la columna de bin.

        ## Returns:
            Expresión que verifica si el número de bin pertenece al intervalo.
        """
        return pl.col(settings.analysis.bin).is_between(
            self.start, self.end, closed="both"
        )

    def group(
        self, feature: str, bivariate_table: pl.DataFrame, settings: config.Settings
    ) -> _GroupedBins:
        """Resume las estadísticas de los bines comprendidos en el intervalo.

        Filtra la tabla de análisis bivariado por número de bin y resume el
        intervalo con sus límites agregados y el porcentaje de target del grupo.

        ## Args:
            feature: Nombre de la feature analizada.
            bivariate_table: Tabla de análisis bivariado de la feature.
            settings: Configuración con los nombres de las columnas.

        ## Returns:
            Intervalo agregado junto con el porcentaje de target del grupo.

        ## Raises:
            TypeError: Si las columnas de mínimo y máximo no son numéricas.
        """
        if (
            not bivariate_table.schema[settings.analysis.min].is_numeric()
            or not bivariate_table.schema[settings.analysis.max].is_numeric()
        ):
            msg = (
                f"Columns 'Min' and 'Max' must be numeric. "
                f"Found types: "
                f"Min={bivariate_table.schema[settings.analysis.min]}, "
                f"Max={bivariate_table.schema[settings.analysis.max]}"
            )
            raise TypeError(msg)

        group = bivariate_table.filter(
            pl.col(settings.analysis.bin).is_between(
                self.start, self.end, closed="both"
            )
        )

        target_pct = _calculate_target_pct(group, settings)
        interval_expr = _build_interval_expr(feature, group, settings)

        return _GroupedBins(interval_expr, target_pct)


@dataclasses.dataclass(frozen=True, slots=True)
class Binner:
    """Asociación entre una feature y los intervalos de bines a agrupar.

    Define cómo deben combinarse los bines de una variable analizada para
    reemplazar sus valores originales por el porcentaje de target de cada grupo.

    ## Attributes:
        snake_case_name: Nombre interno de la feature a transformar.
        pretty_name: Nombre descriptivo de la feature para presentación.
        bin_intervals: Intervalos de bines que conforman cada grupo de agregación.
    """

    snake_case_name: str
    pretty_name: str
    bin_intervals: list[_BinInterval] = dataclasses.field(
        default_factory=list[_BinInterval]
    )

    def merge_bins(self, start: int, end: int | None = None) -> Self:
        """Agrega un intervalo de bines al grupo de transformación.

        ## Args:
            start: Número de bin inicial del intervalo.
            end: Número de bin final del intervalo. Si se omite, se agrupa un único bin.

        ## Returns:
            La misma instancia para permitir llamadas encadenadas.
        """
        end = end or start
        self.bin_intervals.append(_BinInterval(start, end))
        return self

    def _validate_against(self, bivariate_table: pl.DataFrame) -> None:
        """Valida los intervalos contra la tabla de análisis bivariado.

        Comprueba que existan intervalos definidos, que ningún intervalo exceda la
        cantidad de bines de la tabla y que no haya bines solapados entre intervalos.

        ## Args:
            bivariate_table: Tabla de análisis bivariado usada como referencia,
                donde cada fila corresponde a un bin.

        ## Raises:
            ValueError: Si no hay intervalos definidos, si algún intervalo excede la
                cantidad de bines de la tabla o si existen intervalos solapados.
        """
        if not self.bin_intervals:
            msg = (
                f"Feature {self.snake_case_name!r}: 'bin_intervals' must not be empty."
            )
            raise ValueError(msg)

        seen: set[int] = set()
        overlapping: set[int] = set()
        for bin_interval in self.bin_intervals:
            for bin_number in bin_interval.range:
                if bin_number in seen:
                    overlapping.add(bin_number)
                seen.add(bin_number)
        if overlapping:
            msg = f"Feature {self.snake_case_name!r}: overlapping bins {sorted(overlapping)}"
            raise ValueError(msg)

        n_bins = bivariate_table.height
        for bin_interval in self.bin_intervals:
            if bin_interval.exceeds(n_bins):
                msg = (
                    f"Feature {self.snake_case_name!r}: "
                    f"invalid interval {bin_interval}: "
                    f"bivariate_table has {n_bins} bins."
                )
                raise ValueError(msg)

    def _calculate_residual_target_pct(
        self, bivariate_table: pl.DataFrame, settings: config.Settings
    ) -> float:
        """Calcula el porcentaje de target del grupo residual.

        Identifica los bines de la tabla que no pertenecen a ningún intervalo
        definido y calcula su porcentaje de target agregado. Este valor se
        utiliza como valor por defecto para las filas cuyos valores quedan fuera
        de todos los intervalos definidos, incluidos los nulos y los huecos entre grupos.

        ## Args:
            bivariate_table: Tabla de análisis bivariado de la feature.
            settings: Configuración con los nombres de las columnas.

        ## Returns:
            Porcentaje de target del grupo residual, o `0.0` si no quedan bines
            sin cubrir.
        """
        bin_intervals_exprs = [
            bin_interval.to_expr(settings) for bin_interval in self.bin_intervals
        ]
        residual_group = bivariate_table.filter(~pl.any_horizontal(bin_intervals_exprs))
        return _calculate_target_pct(residual_group, settings)

    def build_encoding_expr(
        self,
        bivariate_table: pl.DataFrame,
        settings: config.Settings,
    ) -> pl.Expr:
        """Construye la expresión de transformación por intervalos de bines.

        Agrupa los bines de la tabla de análisis bivariado según los intervalos
        definidos y calcula el porcentaje de target de cada grupo. Cada grupo cubre
        el intervalo agregado entre su mínimo y su máximo, con ambos extremos
        incluidos. Los huecos entre bines del mismo grupo quedan incluidos en el
        intervalo. Los valores fuera de todos los intervalos, así como los nulos,
        reciben el porcentaje del grupo residual.

        Si un valor coincide con la frontera compartida por dos grupos, prevalece
        el primer intervalo según el orden de la cadena `when/then`.

        La columna debe ser numérica. Los intervalos sin límites (`None`)
        definidos (por ejemplo, los que solo agrupan bines de nulos) nunca
        coinciden y sus valores se asignan al grupo residual.

        ## Args:
            bivariate_table: Tabla de análisis bivariado de la feature, con las
                columnas de bin, mínimo, máximo, clientes y target.
            settings: Configuración con los nombres de las columnas.

        ## Returns:
            Expresión de Polars que asigna a cada fila el porcentaje de target
            del intervalo al que pertenece.

        ## Raises:
            ValueError: Si no hay intervalos definidos, si algún intervalo
                excede la cantidad de bines de la tabla o si existen intervalos
                solapados.
        """
        self._validate_against(bivariate_table)

        groups = [
            bin_interval.group(self.snake_case_name, bivariate_table, settings)
            for bin_interval in self.bin_intervals
        ]

        expr = pl.when(groups[0].interval_expr).then(groups[0].target_pct)
        for group in groups[1:]:
            expr = expr.when(group.interval_expr).then(group.target_pct)

        default_target_pct = self._calculate_residual_target_pct(
            bivariate_table, settings
        )
        return expr.otherwise(default_target_pct).alias(self.snake_case_name)


class DualNamed(Protocol):
    """Protocolo para objetos con doble denominación.

    Define el contrato de los objetos que exponen un nombre interno y un
    nombre descriptivo para presentación.
    """

    @property
    def snake_case_name(self) -> str:
        """Nombre interno."""
        ...

    @property
    def pretty_name(self) -> str:
        """Nombre descriptivo para presentación."""
        ...


@dataclasses.dataclass(frozen=True, slots=True)
class NameMapping:
    """Mapeo entre nombres internos y descriptivos.

    Mantiene una asociación inmutable entre el nombre interno de cada columna
    y su nombre descriptivo para presentación o renombre.
    """

    _mapping: types.MappingProxyType[str, str]

    @classmethod
    def from_dict(cls, mapping: Mapping[str, str]) -> Self:
        """Crea un mapeo a partir de un diccionario.

        ## Args:
            mapping: Asociación entre nombres internos y descriptivos.

        ## Returns:
            Nuevo mapeo inmutable con el contenido indicado.
        """
        return cls(types.MappingProxyType(mapping))

    @classmethod
    def from_dual_named(cls, dual_named_iterable: Iterable[DualNamed]) -> Self:
        """Crea un mapeo a partir de objetos con doble nombre.

        ## Args:
            dual_named_iterable: Objetos que exponen nombre interno y descriptivo.

        ## Returns:
            Nuevo mapeo inmutable con ambos nombres de cada objeto.
        """
        return cls.from_dict(
            {
                dual_named.snake_case_name: dual_named.pretty_name
                for dual_named in dual_named_iterable
            }
        )

    @property
    def snake_case_names(self) -> tuple[str, ...]:
        """Nombres internos."""
        return tuple(self._mapping.keys())

    @property
    def pretty_names(self) -> tuple[str, ...]:
        """Nombres descriptivos."""
        return tuple(self._mapping.values())

    def get(self, snake_case_name: str) -> str:
        """Obtiene el nombre descriptivo asociado a un nombre interno.

        ## Args:
            snake_case_name: Nombre interno a buscar.

        ## Returns:
            Nombre descriptivo correspondiente.

        ## Raises:
            KeyError: Si el nombre interno no existe en el mapeo.
        """
        return self._mapping[snake_case_name]

    def items(self) -> ItemsView[str, str]:
        """Devuelve los pares de nombres del mapeo.

        ## Returns:
            Vista con los pares de nombre interno y descriptivo.
        """
        return self._mapping.items()

    def to_expr(self, column: str) -> pl.Expr:
        """Construye una expresión que aplica los nombres descriptivos.

        Reemplaza los valores de la columna según el mapeo y conserva
        el valor original cuando no existe coincidencia.

        ## Args:
            column: Nombre de la columna cuyos valores se reemplazan.

        ## Returns:
            Expresión que devuelve los valores renombrados.
        """
        return (
            pl.col(column)
            .replace_strict(self._mapping, default=pl.col(column))
            .alias(column)
        )
