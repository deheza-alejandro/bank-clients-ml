"""Escenario temporal compartido por los tests de integración."""

import datetime

MONTHS: tuple[datetime.date, ...] = tuple(
    datetime.date(2024, month, 1) for month in range(1, 11)
)
TRAINING_MONTHS: tuple[datetime.date, ...] = MONTHS[:7]
LEAD_MONTH: datetime.date = MONTHS[7]
PREDICTION_MONTHS: tuple[datetime.date, ...] = MONTHS[8:]
LAST_TRAINING_MONTH: datetime.date = TRAINING_MONTHS[-1]
FIRST_PREDICTION_MONTH: datetime.date = PREDICTION_MONTHS[0]
NULL_TARGET_COUNT = 3
