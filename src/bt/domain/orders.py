"""Order abstraction for trading operations.

Implements Open/Closed Principle - easy to add new order types without modifying existing code.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum

from bt.domain.types import Amount, Fee, Percentage, Price, Quantity
from bt.utils.constants import ONE
from bt.utils.decimal_cache import get_decimal


class OrderSide(Enum):
    """Order side enumeration."""

    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    """Order type enumeration."""

    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    STOP_LIMIT = "stop_limit"


class Order(ABC):
    """Abstract base class for all order types.

    Implements Strategy Pattern for different order execution strategies.
    Follows Open/Closed Principle - new order types extend this class.
    """

    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Quantity,
        timestamp: datetime,
    ):
        """Initialize order.

        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            quantity: Quantity to trade
            timestamp: Order creation timestamp
        """
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.timestamp = timestamp
        self.executed = False
        self.execution_price: Price | None = None
        self.execution_time: datetime | None = None

    @abstractmethod
    def calculate_execution_price(self, market_price: Price, slippage: Percentage) -> Price:
        """Calculate execution price for this order type.

        Args:
            market_price: Current market price
            slippage: Slippage percentage

        Returns:
            Execution price
        """
        pass

    @abstractmethod
    def can_execute(self, market_price: Price) -> bool:
        """Check if order can be executed at current market price.

        Args:
            market_price: Current market price

        Returns:
            True if order can execute
        """
        pass

    @abstractmethod
    def get_order_type(self) -> OrderType:
        """Get the order type.

        Returns:
            OrderType enum value
        """
        pass

    def calculate_cost(self, execution_price: Price, quantity: Quantity, fee: Fee) -> Amount:
        """Calculate total cost including fees.

        Args:
            execution_price: Price at execution
            quantity: Quantity traded
            fee: Trading fee

        Returns:
            Total cost
        """
        return Amount(
            get_decimal(execution_price) * get_decimal(quantity) * (ONE + get_decimal(fee))
        )

    def calculate_proceeds(self, execution_price: Price, quantity: Quantity, fee: Fee) -> Amount:
        """Calculate proceeds after fees.

        Args:
            execution_price: Price at execution
            quantity: Quantity traded
            fee: Trading fee

        Returns:
            Proceeds after fees
        """
        return Amount(
            get_decimal(execution_price) * get_decimal(quantity) * (ONE - get_decimal(fee))
        )

    def mark_executed(self, execution_price: Price, execution_time: datetime) -> None:
        """Mark order as executed.

        Args:
            execution_price: Price at which order was executed
            execution_time: Time of execution
        """
        self.executed = True
        self.execution_price = execution_price
        self.execution_time = execution_time


class MarketOrder(Order):
    """Market order - executes immediately at market price with slippage.

    Most common order type for backtesting.
    """

    def get_order_type(self) -> OrderType:
        """Get order type."""
        return OrderType.MARKET

    def calculate_execution_price(self, market_price: Price, slippage: Percentage) -> Price:
        """Calculate execution price with slippage.

        For buys: price increases (unfavorable)
        For sells: price decreases (unfavorable)

        Args:
            market_price: Current market price
            slippage: Slippage percentage

        Returns:
            Execution price with slippage applied
        """
        if self.side == OrderSide.BUY:
            # Buy at higher price (slippage increases cost)
            return Price(get_decimal(market_price) * (ONE + get_decimal(slippage)))
        # Sell at lower price (slippage decreases proceeds)
        return Price(get_decimal(market_price) * (ONE - get_decimal(slippage)))

    def can_execute(self, market_price: Price) -> bool:  # noqa: ARG002
        """Market orders always execute immediately.

        Args:
            market_price: Current market price (unused)

        Returns:
            Always True
        """
        return True


class LimitOrder(Order):
    """Limit order - only executes if price is favorable.

    Buy: only if market price <= limit price
    Sell: only if market price >= limit price
    """

    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Quantity,
        limit_price: Price,
        timestamp: datetime,
    ):
        """Initialize limit order.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Quantity to trade
            limit_price: Maximum buy price or minimum sell price
            timestamp: Order creation timestamp
        """
        super().__init__(symbol, side, quantity, timestamp)
        self.limit_price = limit_price

    def get_order_type(self) -> OrderType:
        """Get order type."""
        return OrderType.LIMIT

    def calculate_execution_price(self, market_price: Price, slippage: Percentage) -> Price:
        """Calculate execution price (at limit or better).

        Args:
            market_price: Current market price
            slippage: Slippage percentage (applied to market price)

        Returns:
            Execution price (limit price or better)
        """
        market_with_slippage = MarketOrder(
            self.symbol, self.side, self.quantity, self.timestamp
        ).calculate_execution_price(market_price, slippage)

        if self.side == OrderSide.BUY:
            # Execute at the lower of limit price and market price
            return Price(min(get_decimal(self.limit_price), get_decimal(market_with_slippage)))
        # Execute at the higher of limit price and market price
        return Price(max(get_decimal(self.limit_price), get_decimal(market_with_slippage)))

    def can_execute(self, market_price: Price) -> bool:
        """Check if limit order can execute.

        Args:
            market_price: Current market price

        Returns:
            True if price condition is met
        """
        if self.side == OrderSide.BUY:
            # Buy only if market price <= limit price
            return get_decimal(market_price) <= get_decimal(self.limit_price)
        # Sell only if market price >= limit price
        return get_decimal(market_price) >= get_decimal(self.limit_price)


class StopLossOrder(Order):
    """Stop loss order - triggers when price crosses stop price.

    Becomes market order when triggered.
    Buy: triggers when price >= stop price (stop buy)
    Sell: triggers when price <= stop price (stop loss)
    """

    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Quantity,
        stop_price: Price,
        timestamp: datetime,
    ):
        """Initialize stop loss order.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Quantity to trade
            stop_price: Trigger price
            timestamp: Order creation timestamp
        """
        super().__init__(symbol, side, quantity, timestamp)
        self.stop_price = stop_price
        self.triggered = False

    def get_order_type(self) -> OrderType:
        """Get order type."""
        return OrderType.STOP_LOSS

    def calculate_execution_price(self, market_price: Price, slippage: Percentage) -> Price:
        """Calculate execution price (market order after trigger).

        Args:
            market_price: Current market price
            slippage: Slippage percentage

        Returns:
            Market price with slippage
        """
        # Behaves like market order once triggered
        market_order = MarketOrder(self.symbol, self.side, self.quantity, self.timestamp)
        return market_order.calculate_execution_price(market_price, slippage)

    def can_execute(self, market_price: Price) -> bool:
        """Check if stop loss is triggered.

        Args:
            market_price: Current market price

        Returns:
            True if stop price is crossed
        """
        if self.side == OrderSide.SELL:
            # Stop loss sell triggers when price falls to/below stop price
            if get_decimal(market_price) <= get_decimal(self.stop_price):
                self.triggered = True
                return True
        else:
            # Stop buy triggers when price rises to/above stop price
            if get_decimal(market_price) >= get_decimal(self.stop_price):
                self.triggered = True
                return True

        return False


class StopLimitOrder(Order):
    """Stop limit order - limit order that activates after stop price is hit.

    More conservative than stop loss (may not fill if price gaps).
    """

    def __init__(
        self,
        symbol: str,
        side: OrderSide,
        quantity: Quantity,
        stop_price: Price,
        limit_price: Price,
        timestamp: datetime,
    ):
        """Initialize stop limit order.

        Args:
            symbol: Trading symbol
            side: Order side
            quantity: Quantity to trade
            stop_price: Trigger price
            limit_price: Maximum buy / minimum sell price after trigger
            timestamp: Order creation timestamp
        """
        super().__init__(symbol, side, quantity, timestamp)
        self.stop_price = stop_price
        self.limit_price = limit_price
        self.triggered = False

    def get_order_type(self) -> OrderType:
        """Get order type."""
        return OrderType.STOP_LIMIT

    def calculate_execution_price(self, market_price: Price, slippage: Percentage) -> Price:
        """Calculate execution price (limit order after trigger).

        Args:
            market_price: Current market price
            slippage: Slippage percentage

        Returns:
            Limit price or better
        """
        # Behaves like limit order once triggered
        limit_order = LimitOrder(
            self.symbol, self.side, self.quantity, self.limit_price, self.timestamp
        )
        return limit_order.calculate_execution_price(market_price, slippage)

    def can_execute(self, market_price: Price) -> bool:
        """Check if order can execute.

        First checks if triggered, then checks limit condition.

        Args:
            market_price: Current market price

        Returns:
            True if triggered and limit condition met
        """
        # Check if stop is triggered
        if not self.triggered:
            if self.side == OrderSide.SELL:
                if get_decimal(market_price) <= get_decimal(self.stop_price):
                    self.triggered = True
            else:
                if get_decimal(market_price) >= get_decimal(self.stop_price):
                    self.triggered = True

        # If triggered, check limit condition
        if self.triggered:
            limit_order = LimitOrder(
                self.symbol, self.side, self.quantity, self.limit_price, self.timestamp
            )
            return limit_order.can_execute(market_price)

        return False
