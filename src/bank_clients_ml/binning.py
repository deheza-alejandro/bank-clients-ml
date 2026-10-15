import dataclasses
import types
from collections.abc import ItemsView, Iterable, Mapping
from typing import NamedTuple, Protocol, Self

import polars as pl

from bank_clients_ml import config


class _BinGroup(NamedTuple):
    range_condition: pl.Expr
    target_pct: float


def _get_range_condition(
    column: str, group: pl.DataFrame, settings: config.Settings
) -> pl.Expr:
    """Construye la condición de pertenencia al intervalo del grupo.

    ## Args:
        column: Columna sobre la que se construye la condición.

    ## Returns:
        Expresión de pertenencia al intervalo `[low, high]` con ambos extremos
        incluidos, o `False` si el grupo no tiene límites (por ejemplo, si solo
        agrupa bines de nulos).
    """
    min = group.get_column(settings.analysis.min).min()
    max = group.get_column(settings.analysis.max).max()

    if min is None or max is None:
        return pl.lit(False)
    return pl.col(column).is_between(min, max, closed="both")


def _calculate_target_pct(table: pl.DataFrame, settings: config.Settings) -> float:
    """Calcula el porcentaje de target de un grupo de bines.

    ## Args:
        table: Tabla con un grupo de bines con las columnas
            `settings.analysis.clients` y `settings.dataset.target`.

    ## Returns:
        Porcentaje de target del grupo, o `0.0` si el grupo está vacío.
    """
    clients = float(table.get_column(settings.analysis.clients).sum())
    target = float(table.get_column(settings.dataset.target).sum())
    return target / clients * 100.0 if clients > 0.0 else 0.0


@dataclasses.dataclass(frozen=True, slots=True)
class BinRange:
    """Rango inclusivo de bines que se agrupan en un mismo valor.

    Representa un intervalo cerrado sobre la numeración de bines generada por
    el análisis bivariado. Los bines comprendidos entre ambos extremos reciben
    el mismo porcentaje de target durante la transformación.

    ## Attributes:
        start: Número de bin inicial del intervalo.
        end: Número de bin final del intervalo.

    ## Raises:
        ValueError: Si `start` o `end` son menores a 1, o si `start` es mayor que `end`.
    """

    start: int
    end: int

    def __post_init__(self):
        """Valida los extremos del rango."""
        if self.start < 1 or self.end < 1:
            msg = f"Both 'start' and 'end' must be >= 1. start: {self.start}, end: {self.end}"
            raise ValueError(msg)
        if self.start > self.end:
            msg = f"'start' must be <= 'end'. start: {self.start}, end: {self.end}"
            raise ValueError(msg)

    @property
    def range(self) -> range:
        return range(self.start, self.end + 1)

    def exceeds(self, n_bins: int) -> bool:
        return self.end > n_bins

    def to_expr(self, settings: config.Settings) -> pl.Expr:
        return pl.col(settings.analysis.bin).is_between(
            self.start, self.end, closed="both"
        )

    def to_bin_group(
        self, column: str, table: pl.DataFrame, settings: config.Settings
    ) -> _BinGroup:
        """Agrega las estadísticas de un rango de bines.

        Filtra la tabla de análisis bivariado por número de bin y resume el
        intervalo con los límites `[min, max]` con el porcentaje de target del
        grupo.

        ## Args:
            table: Tabla de análisis bivariado de la columna.
            bin_range: Rango de bines a agregar.
            settings: Configuración con los nombres de las columnas.
        """
        if (
            not table.schema[settings.analysis.min].is_numeric()
            or not table.schema[settings.analysis.max].is_numeric()
        ):
            msg = (
                f"Columns 'Min' and 'Max' must be numeric. "
                f"Found types: "
                f"Min={table.schema[settings.analysis.min]}, "
                f"Max={table.schema[settings.analysis.max]}"
            )
            raise TypeError(msg)

        group = table.filter(
            pl.col(settings.analysis.bin).is_between(
                self.start, self.end, closed="both"
            )
        )

        target_pct = _calculate_target_pct(group, settings)
        range_condition = _get_range_condition(column, group, settings)

        return _BinGroup(range_condition, target_pct)


@dataclasses.dataclass(frozen=True, slots=True)
class BinTransformation:
    """Asociación entre una columna y sus rangos de bines.

    Define cómo deben combinarse los bines de una variable analizada para
    reemplazar sus valores originales por el porcentaje de target de cada grupo.

    ## Attributes:
        column: Nombre de la columna a transformar.
        bin_ranges: Rangos de bines que conforman cada grupo.
    """

    snake_case_name: str
    pretty_name: str
    bin_ranges: list[BinRange] = dataclasses.field(default_factory=list[BinRange])

    def add_range(self, start: int, end: int | None = None) -> Self:
        end = end or start
        self.bin_ranges.append(BinRange(start, end))
        return self

    def validate(self, bivariate_table: pl.DataFrame) -> None:
        """Valida la transformación de bines.

        ## Args:
            bin_transformation: Asociación entre la columna a transformar y los
                rangos de bines a validar.
            n_bins: Cantidad de bines de la tabla.

        ## Raises:
            ValueError: Si algún rango excede `n_bins`, si `bin_ranges` está
            vacía o si se solapan bines.
        """
        if not self.bin_ranges:
            msg = f"Column {self.snake_case_name!r}: 'bin_ranges' must not be empty."
            raise ValueError(msg)

        seen: set[int] = set()
        overlapping: set[int] = set()
        for bin_range in self.bin_ranges:
            for bin_number in bin_range.range:
                if bin_number in seen:
                    overlapping.add(bin_number)
                seen.add(bin_number)
        if overlapping:
            msg = f"Column {self.snake_case_name!r}: overlapping bins {sorted(overlapping)}"
            raise ValueError(msg)

        n_bins = bivariate_table.height
        for bin_range in self.bin_ranges:
            if bin_range.exceeds(n_bins):
                msg = (
                    f"Column {self.snake_case_name!r}: "
                    f"invalid range {bin_range}: "
                    f"table has {n_bins} bins."
                )
                raise ValueError(msg)

    def _get_default_target_pct(
        self, bivariate_table: pl.DataFrame, settings: config.Settings
    ) -> float:
        """Calcula el porcentaje de target del grupo residual.

        Identifica los bines de la tabla que no pertenecen a ningún rango de la
        transformación y calcula su porcentaje de target agregado. Este valor se
        utiliza como valor por defecto para las filas cuyos valores quedan fuera
        de todos los intervalos definidos, incluidos los nulos y los huecos entre grupos.

        ## Args:
            table: Tabla de análisis bivariado de la columna.
            bin_transformation: Asociación entre la columna a transformar y los
                rangos de bines que definen cada grupo de agregación.
            settings: Configuración con los nombres de las columnas.

        ## Returns:
            Porcentaje de target del grupo residual, o `0.0` si no quedan bines
            sin cubrir.
        """
        bin_match = [bin_range.to_expr(settings) for bin_range in self.bin_ranges]
        residual_group = bivariate_table.filter(~pl.any_horizontal(bin_match))
        return _calculate_target_pct(residual_group, settings)

    def group_bins_by_ranges(
        self,
        bivariate_table: pl.DataFrame,
        settings: config.Settings,
    ) -> pl.Expr:
        """Construye una expresión de transformación por rangos de bines.

        Agrupa las estadísticas de la tabla de análisis bivariado según los rangos
        indicados y calcula el porcentaje de target de cada grupo. Cada grupo cubre
        el intervalo `[min, max]` agregado de sus bines, con ambos extremos
        incluidos. Los huecos entre bines del mismo grupo quedan incluidos en el
        intervalo. Los valores fuera de todos los rangos, así como los nulos,
        reciben el porcentaje del grupo residual.

        Si un valor cae exactamente en la frontera compartida por dos grupos
        (`max` de uno igual al `min` del siguiente), gana el primer rango de la
        lista, según el orden de la cadena `when/then`.

        La columna debe ser numérica. Los rangos con límites `None` (por ejemplo,
        los que solo agrupan bines de nulos) nunca hacen match y esos valores van
        al grupo residual.

        ## Args:
            bin_transformation: Asociación entre una columna a transformar y los
                rangos de bines que definen cada grupo de agregación.
            table: Tabla de análisis bivariado de la columna, con las columnas de
                bin, mínimo, máximo, clientes y target.
            settings: Configuración con el nombre de la columna target. Si no se
                indica, se obtiene la configuración global.

        ## Returns:
            Expresión de Polars que asigna a cada fila el porcentaje de target
            del rango al que pertenece.

        ## Raises:
            ValueError: Si algún rango excede la cantidad de bines de la tabla.
        """
        self.validate(bivariate_table)

        groups = [
            bin_range.to_bin_group(self.snake_case_name, bivariate_table, settings)
            for bin_range in self.bin_ranges
        ]

        expr = pl.when(groups[0].range_condition).then(groups[0].target_pct)
        for group in groups[1:]:
            expr = expr.when(group.range_condition).then(group.target_pct)

        default_target_pct = self._get_default_target_pct(bivariate_table, settings)
        return expr.otherwise(default_target_pct).alias(self.snake_case_name)


class Nameable(Protocol):
    @property
    def snake_case_name(self) -> str: ...

    @property
    def pretty_name(self) -> str: ...


@dataclasses.dataclass(frozen=True, slots=True)
class Renames:
    _mapping: types.MappingProxyType[str, str]

    @classmethod
    def from_dict(cls, mapping: Mapping[str, str]) -> Self:
        return cls(types.MappingProxyType(mapping))

    @classmethod
    def from_nameable(cls, nameable_iterable: Iterable[Nameable]) -> Self:
        return cls.from_dict(
            {
                nameable.snake_case_name: nameable.pretty_name
                for nameable in nameable_iterable
            }
        )

    @property
    def snake_case_names(self) -> tuple[str, ...]:
        return tuple(self._mapping.keys())

    @property
    def pretty_names(self) -> tuple[str, ...]:
        return tuple(self._mapping.values())

    def get(self, snake_case_name: str) -> str:
        return self._mapping[snake_case_name]

    def items(self) -> ItemsView[str, str]:
        return self._mapping.items()

    def to_expr(self, column: str) -> pl.Expr:
        """Aplica nombres distintos.

        Reemplaza los valores de la columna feature según el diccionario de
        renombres y conserva el valor original cuando no existe correspondencia.

        ## Args:
            column: nombre de la columna.

        ## Returns:
            .
        """
        return (
            pl.col(column)
            .replace_strict(self._mapping, default=pl.col(column))
            .alias(column)
        )
