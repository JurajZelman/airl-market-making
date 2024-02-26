"""Commission models for computation of transaction costs."""

from abc import ABC, abstractmethod


class CommissionModel(ABC):
    """Abstract class for commission models."""

    @abstractmethod
    def maker_fee(self, quantity: float, price: float) -> float:
        """
        Compute the maker fee.

        Args:
            quantity: Quantity of the asset that is being purchased (positive)
                or sold (negative).
            price: Price at which the asset is being purchased or sold.
        """
        raise NotImplementedError("'maker_fee' method is not implemented.")

    @abstractmethod
    def taker_fee(self, quantity: float, price: float) -> float:
        """
        Compute the taker fee.

        Args:
            quantity: Quantity of the asset that is being purchased (positive)
                or sold (negative).
            price: Price at which the asset is being purchased or sold.
        """
        raise NotImplementedError("'taker_fee' method is not implemented.")


class NoCommissionModel(CommissionModel):
    """Commission model with no transaction costs."""

    def maker_fee(self, quantity: float, price: float) -> float:
        """
        Compute the maker fee.

        Args:
            quantity: Quantity of the asset that is being purchased (positive)
                or sold (negative).
            price: Price at which the asset is being purchased or sold.

        Returns:
            Maker fee to be paid to the trading platform.
        """
        return 0

    def taker_fee(self, quantity: float, price: float) -> float:
        """
        Compute the taker fee.

        Args:
            quantity: Quantity of the asset that is being purchased (positive)
                or sold (negative).
            price: Price at which the asset is being purchased or sold.

        Returns:
            Taker fee to be paid to the trading platform.
        """
        return 0


class BinanceCommissions(CommissionModel):
    """Binance commission model for spot and margin trading."""

    def __init__(self, tier: int, bnb: bool = False) -> None:
        """
        Initialize the commission model.

        Args:
            tier: Binance tier of the user. Goes from `0` to `14` indicating the
                regular user, VIP 1, VIP 2, ..., VIP 9 tiers and values `10-14`
                indicating the Binance Liquidity Program tiers with rebates.
            bnb: Indicates whether to use Binance `BNB` 25% discount when paying
                with the `BNB` token. Defaults to False.
        """
        self.tier = tier
        self.bnb = bnb

    def maker_fee(self, quantity: float, price: float) -> float:
        """
        Compute the maker fee.

        Args:
            quantity: Quantity of the asset that is being purchased (positive)
                or sold (negative).
            price: Price at which the asset is being purchased or sold.

        Returns:
            Maker fee to be paid to the trading platform. Negative fee indicates
            a rebate.
        """
        quantity = abs(quantity)
        size = quantity * price
        if self.bnb and self.tier < 10:
            size = size * 0.75

        match self.tier:
            case 0:
                return 0.001 * size
            case 1:
                return 0.0009 * size
            case 2:
                return 0.0008 * size
            case 3:
                return 0.00042 * size
            case 4:
                return 0.00042 * size
            case 5:
                return 0.00036 * size
            case 6:
                return 0.0003 * size
            case 7:
                return 0.00024 * size
            case 8:
                return 0.00018 * size
            case 9:
                return 0.00012 * size
            case 10:
                return 0.0000 * size
            case 11:
                return -0.00004 * size
            case 12:
                return -0.00006 * size
            case 13:
                return -0.00008 * size
            case 14:
                return -0.0001 * size
            case _:
                raise ValueError(f"Invalid tier: {self.tier}")

    def taker_fee(self, quantity: float, price: float) -> float:
        """
        Compute the taker fee.

        Args:
            quantity: Quantity of the asset that is being purchased (positive)
                or sold (negative).
            price: Price at which the asset is being purchased or sold.

        Returns:
            Taker fee to be paid to the trading platform.
        """
        quantity = abs(quantity)
        size = quantity * price
        if self.bnb:
            size = size * 0.75
        match self.tier:
            case 0:
                return 0.001 * size
            case 1:
                return 0.001 * size
            case 2:
                return 0.001 * size
            case 3:
                return 0.0006 * size
            case 4:
                return 0.00054 * size
            case 5:
                return 0.00048 * size
            case 6:
                return 0.00042 * size
            case 7:
                return 0.00036 * size
            case 8:
                return 0.0003 * size
            case 9:
                return 0.00024 * size
            case 10:
                return 0.001 * size
            case 11:
                return 0.001 * size
            case 12:
                return 0.001 * size
            case 13:
                return 0.001 * size
            case 14:
                return 0.001 * size
            case _:
                raise ValueError(f"Invalid tier: {self.tier}")


class BitCommissions(CommissionModel):
    """BIT exchange commission model for spot and margin trading."""

    def __init__(self, tier: int) -> None:
        """
        Initialize the commission model.

        Args:
            tier: BIT.com tier of the user. Goes from `1` to `9` indicating the
                regular VIP 1, VIP 2, ..., VIP 9 tiers.
        """
        self.tier = tier

    def maker_fee(self, quantity: float, price: float) -> float:
        """
        Compute the maker fee.

        Args:
            quantity: Quantity of the asset that is being purchased (positive)
                or sold (negative).
            price: Price at which the asset is being purchased or sold.

        Returns:
            Maker fee to be paid to the trading platform. Negative fee indicates
            a rebate.
        """
        quantity = abs(quantity)
        size = quantity * price

        match self.tier:
            case 1:
                return 0.0008 * size
            case 2:
                return 0.0007 * size
            case 3:
                return 0.0006 * size
            case 4:
                return 0.0005 * size
            case 5:
                return 0.0004 * size
            case 6:
                return 0.0003 * size
            case 7:
                return 0.0002 * size
            case 8:
                return 0.0001 * size
            case 9:
                return 0 * size
            case _:
                raise ValueError(f"Invalid tier: {self.tier}")

    def taker_fee(self, quantity: float, price: float) -> float:
        """
        Compute the taker fee.

        Args:
            quantity: Quantity of the asset that is being purchased (positive)
                or sold (negative).
            price: Price at which the asset is being purchased or sold.

        Returns:
            Taker fee to be paid to the trading platform.
        """
        quantity = abs(quantity)
        size = quantity * price

        match self.tier:
            case 1:
                return 0.001 * size
            case 2:
                return 0.0009 * size
            case 3:
                return 0.0008 * size
            case 4:
                return 0.0007 * size
            case 5:
                return 0.0006 * size
            case 6:
                return 0.0005 * size
            case 7:
                return 0.0004 * size
            case 8:
                return 0.00035 * size
            case 9:
                return 0.0003 * size
            case _:
                raise ValueError(f"Invalid tier: {self.tier}")
