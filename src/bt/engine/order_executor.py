"""Order execution service.

Responsible for executing orders with proper validation and fee calculation.
Follows Single Responsibility Principle.
"""

from datetime import datetime

from bt.domain.models import Position
from bt.domain.orders import MarketOrder, Order, OrderSide
from bt.domain.types import Amount, Fee, Percentage, Price, Quantity
from bt.utils.logging import get_logger

logger = get_logger(__name__)


class OrderExecutor:
    """Executes trading orders with fees and slippage.

    Responsibilities:
    - Execute orders
    - Apply fees and slippage
    - Validate orders
    - Calculate costs and proceeds

    Does NOT handle:
    - Position tracking (Portfolio)
    - Trade recording (TradeRecorder)
    - Cash management (Portfolio)
    """

    def __init__(self, fee: Fee, slippage: Percentage):
        """Initialize order executor.

        Args:
            fee: Trading fee as decimal (e.g., 0.0005 for 0.05%)
            slippage: Slippage as decimal
        """
        self.fee = fee
        self.slippage = slippage

    def execute_order(
        self, order: Order, market_price: Price, current_cash: Amount
    ) -> tuple[bool, Price | None, Amount]:
        """Execute an order if conditions are met.

        Args:
            order: Order to execute
            market_price: Current market price
            current_cash: Available cash

        Returns:
            Tuple of (success, execution_price, cost_or_proceeds)
        """
        # Check if order can execute
        if not order.can_execute(market_price):
            logger.debug(
                f"Order cannot execute: {order.symbol} {order.side.value}",
                extra={"market_price": float(market_price)},
            )
            return False, None, Amount(0)

        # Calculate execution price
        execution_price = order.calculate_execution_price(market_price, self.slippage)

        # Calculate cost/proceeds
        if order.side == OrderSide.BUY:
            cost = order.calculate_cost(execution_price, order.quantity, self.fee)

            # Check if we have enough cash
            if cost > current_cash:
                logger.warning(
                    "Insufficient funds for buy order",
                    extra={
                        "symbol": order.symbol,
                        "cost": float(cost),
                        "available_cash": float(current_cash),
                    },
                )
                return False, None, Amount(0)

            # Mark order as executed
            order.mark_executed(execution_price, order.timestamp)

            logger.debug(
                "Buy order executed",
                extra={
                    "symbol": order.symbol,
                    "quantity": float(order.quantity),
                    "price": float(execution_price),
                    "cost": float(cost),
                },
            )

            return True, execution_price, cost

        # SELL
        proceeds = order.calculate_proceeds(execution_price, order.quantity, self.fee)

        # Mark order as executed
        order.mark_executed(execution_price, order.timestamp)

        logger.debug(
            "Sell order executed",
            extra={
                "symbol": order.symbol,
                "quantity": float(order.quantity),
                "price": float(execution_price),
                "proceeds": float(proceeds),
            },
        )

        return True, execution_price, proceeds

    def create_market_buy_order(
        self, symbol: str, quantity: Quantity, timestamp: datetime
    ) -> MarketOrder:
        """Create a market buy order.

        Args:
            symbol: Trading symbol
            quantity: Quantity to buy
            timestamp: Order timestamp

        Returns:
            MarketOrder instance
        """
        return MarketOrder(symbol, OrderSide.BUY, quantity, timestamp)

    def create_market_sell_order(
        self, symbol: str, quantity: Quantity, timestamp: datetime
    ) -> MarketOrder:
        """Create a market sell order.

        Args:
            symbol: Trading symbol
            quantity: Quantity to sell
            timestamp: Order timestamp

        Returns:
            MarketOrder instance
        """
        return MarketOrder(symbol, OrderSide.SELL, quantity, timestamp)

    def calculate_max_quantity(self, price: Price, available_cash: Amount) -> Quantity:
        """Calculate maximum quantity that can be purchased.

        Args:
            price: Purchase price
            available_cash: Available cash

        Returns:
            Maximum quantity affordable
        """
        from bt.utils.constants import PRECISION_BUFFER
        from bt.utils.decimal_cache import get_decimal

        # Calculate unit cost (price + slippage + fee)
        execution_price = get_decimal(price) * (get_decimal("1") + get_decimal(self.slippage))
        unit_cost = execution_price * (get_decimal("1") + get_decimal(self.fee))

        if unit_cost <= 0:
            return Quantity(0)

        # Calculate max quantity with precision buffer
        max_qty = get_decimal(available_cash) / unit_cost
        return Quantity(max_qty * get_decimal(PRECISION_BUFFER))

    def update_position_on_buy(
        self,
        position: Position,
        execution_price: Price,
        quantity: Quantity,
        entry_date: datetime,
    ) -> Position:
        """Update position after buy execution.

        Args:
            position: Current position
            execution_price: Price at which order was executed
            quantity: Quantity bought
            entry_date: Entry date

        Returns:
            Updated position
        """

        if position.is_open:
            # Add to existing position (weighted average pricing)
            total_cost = position.quantity * position.entry_price
            new_cost = quantity * execution_price
            total_quantity = position.quantity + quantity

            avg_price = (total_cost + new_cost) / total_quantity
            position.entry_price = Price(avg_price)
            position.quantity = Quantity(total_quantity)
        else:
            # Create new position
            position.quantity = Quantity(quantity)
            position.entry_price = Price(execution_price)
            position.entry_date = entry_date

        return position

    def calculate_pnl(
        self,
        position: Position,
        exit_price: Price,
    ) -> tuple[Amount, Percentage]:
        """Calculate P&L for position exit.

        Args:
            position: Position being closed
            exit_price: Exit price

        Returns:
            Tuple of (pnl, return_percentage)
        """
        from decimal import Decimal

        from bt.utils.decimal_cache import get_decimal

        # Calculate actual entry cost (with slippage and fee)
        entry_cost = (
            position.entry_price
            * position.quantity
            * (Decimal("1") + Decimal(self.slippage))
            * (Decimal("1") + Decimal(self.fee))
        )

        # Calculate actual exit proceeds (with slippage and fee)
        exit_proceeds = (
            get_decimal(exit_price)
            * get_decimal(position.quantity)
            * (Decimal("1") - Decimal(self.slippage))
            * (Decimal("1") - Decimal(self.fee))
        )

        # P&L is proceeds minus cost
        pnl = Amount(exit_proceeds - entry_cost)

        # Return percentage
        return_pct = Percentage(
            (get_decimal(exit_price) / get_decimal(position.entry_price) - Decimal("1"))
            * Decimal("100")
        )

        return pnl, return_pct
