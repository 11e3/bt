"""Constants for financial calculations."""

from decimal import Decimal

# Basic numeric constants
ZERO = Decimal("0")
ONE = Decimal("1")
HUNDRED = Decimal("100")

# Precision and safety buffers
SAFETY_BUFFER = Decimal("0.999")
PRECISION_BUFFER = Decimal("0.99999")

# Trading thresholds
MIN_TRADE_AMOUNT = Decimal("0.0005")
MAX_POSITION_SIZE = Decimal("10000000")

# Metric constants
METRIC_PRECISION = Decimal("999999")
