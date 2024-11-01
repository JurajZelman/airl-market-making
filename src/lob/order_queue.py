"""Price order queue for orders of the same price."""

from pyllist import dllist, dllistnode

from src.lob.orders import Order
from src.lob.utils import round_to_lot


class OrderQueue:
    """
    Price order queue for orders of the same price. The queue is implemented
    as a double-linked list and is sorted by the entry time of the orders
    (price-time priority rule).
    """

    def __init__(self, lot_size: float) -> None:
        """
        Initialize a price order queue.

        Args:
            lot_size: Lot size of the orders in the queue.
        """
        self.lot_size = lot_size  # Lot size of the orders in the queue
        self.queue = dllist()  # Double-linked list of orders
        self.num_orders = 0  # Number of orders in the queue
        self.volume = 0  # Cumulative volume of the orders in the queue

    @property
    def first_order(self) -> Order:
        """Return the first order in the queue."""
        return self._first_order

    @first_order.getter
    def first_order(self) -> Order:
        """Return the first order in the queue."""
        return self.queue.first.value

    @property
    def first_order_node(self) -> dllistnode:
        """Return the node of the first order in the queue."""
        return self._first_order_node

    @first_order_node.getter
    def first_order_node(self) -> dllistnode:
        """Return the node of the first order in the queue."""
        return self.queue.first

    @property
    def last_order(self) -> Order:
        """Return the last order in the queue."""
        return self._last_order

    @last_order.getter
    def last_order(self) -> Order:
        """Return the last order in the queue."""
        return self.queue.last.value

    @property
    def last_order_node(self) -> dllistnode:
        """Return the node of the last order in the queue."""
        return self._last_order

    @last_order_node.getter
    def last_order_node(self) -> dllistnode:
        """Return the node of the last order in the queue."""
        return self.queue.last

    def add_order(self, order: Order) -> dllistnode:
        """
        Add an order to the price order queue.

        Args:
            order: Order to add.

        Returns:
            Node of the added order in the double-linked list.
        """
        # Add order to the last position in the queue
        if (
            self.num_orders == 0
            or order.entry_time >= self.last_order.entry_time
        ):
            self.num_orders += 1
            self.volume += order.volume
            self.volume = round_to_lot(self.volume, self.lot_size)
            return self.queue.append(order)

        # Find the position where to place the order in the queue
        else:
            temp = self.last_order_node
            while temp is not None and order.entry_time < temp.value.entry_time:
                temp = temp.prev
            if temp is None:
                self.num_orders += 1
                self.volume += order.volume
                self.volume = round_to_lot(self.volume, self.lot_size)
                return self.queue.appendleft(order)
            else:
                self.num_orders += 1
                self.volume += order.volume
                self.volume = round_to_lot(self.volume, self.lot_size)
                self.queue.insert(order, after=temp)

    def remove_order(self, order: Order, order_node: dllistnode) -> None:
        """
        Remove an order from the price order queue.

        Args:
            order: Order to remove.
            order_node: Node of the order in the double-linked list.
        """
        self.volume -= order.volume
        self.volume = round_to_lot(self.volume, self.lot_size)
        self.queue.remove(order_node)
        self.num_orders -= 1

    def update_order_volume(self, order: Order, volume: float) -> None:
        """
        Update the volume of an order in the price order queue. This is not
        meant to be used by the agent, it just serves as an helper function for
        the exchange for updating the volume of a partially matched order.

        Args:
            order: Order to update.
            volume: New volume of the order.
        """
        if volume <= 0:
            raise ValueError("Volume must be positive.")
        if volume == order.volume:
            return
        self.volume = self.volume - order.volume + volume
        self.volume = round_to_lot(self.volume, self.lot_size)
        order.volume = volume

    def __repr__(self) -> str:
        """
        Return a string representation of the order queue. The string is a
        concatenation of the string representations of the orders in the queue,
        with the last line being the number of orders and the total volume
        of the orders in the queue.

        Returns:
            repr: String representation of the order queue.
        """
        repr, temp = "", self.first_order_node
        while temp:
            repr += f"{temp.__repr__()} \n"
            temp = temp.next
        repr += f"Num orders: {self.num_orders}, Volume: {self.volume}"
        return repr
