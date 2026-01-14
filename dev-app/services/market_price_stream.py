"""
Market Price Streaming Service for Virtual Stop Loss

Uses Lightstreamer to receive real-time BID/OFFER prices
for monitored scalp positions. This enables sub-second
reaction to price movements for virtual stop loss triggers.

Usage:
    manager = MarketPriceStreamManager(auth_headers)
    await manager.connect()
    manager.subscribe('CS.D.EURUSD.CEEM.IP', on_price_update)
"""

import asyncio
import logging
from typing import Dict, Callable, Optional, List
from dataclasses import dataclass
from datetime import datetime
from threading import Lock

try:
    from lightstreamer.client import LightstreamerClient, Subscription, SubscriptionListener
    LIGHTSTREAMER_AVAILABLE = True
except ImportError:
    LIGHTSTREAMER_AVAILABLE = False
    LightstreamerClient = None
    Subscription = None
    SubscriptionListener = object

from config_virtual_stop import (
    get_lightstreamer_url,
    LIGHTSTREAMER_CONNECT_TIMEOUT,
    LIGHTSTREAMER_AUTO_RECONNECT,
    LIGHTSTREAMER_RECONNECT_DELAY_SECONDS,
    LIGHTSTREAMER_MAX_RECONNECT_ATTEMPTS,
)

logger = logging.getLogger(__name__)


@dataclass
class MarketPrice:
    """Current market price data from streaming."""
    epic: str
    bid: float
    offer: float
    mid: float
    spread: float
    timestamp: datetime
    market_state: str


class MarketPriceListener(SubscriptionListener):
    """
    Handles MARKET subscription updates from Lightstreamer.

    Invokes callback with MarketPrice on each update.
    """

    def __init__(self, epic: str, callback: Callable[[MarketPrice], None]):
        self.epic = epic
        self.callback = callback
        self.last_update: Optional[MarketPrice] = None
        self.update_count = 0

    def onItemUpdate(self, update):
        """Process incoming market price update."""
        try:
            # Get BID and OFFER values
            bid_str = update.getValue('BID')
            offer_str = update.getValue('OFFER')

            if bid_str is None or offer_str is None:
                return

            bid = float(bid_str)
            offer = float(offer_str)
            mid = (bid + offer) / 2
            spread = offer - bid

            # Get market state (TRADEABLE, CLOSED, etc.)
            market_state = update.getValue('MARKET_STATE') or 'TRADEABLE'

            price = MarketPrice(
                epic=self.epic,
                bid=bid,
                offer=offer,
                mid=mid,
                spread=spread,
                timestamp=datetime.utcnow(),
                market_state=market_state
            )

            self.last_update = price
            self.update_count += 1

            # Log every 100th update for monitoring
            if self.update_count % 100 == 0:
                logger.debug(f"[MARKET STREAM] {self.epic}: BID={bid:.5f} OFFER={offer:.5f} "
                           f"(updates: {self.update_count})")

            # Invoke callback with new price
            self.callback(price)

        except Exception as e:
            logger.error(f"[MARKET STREAM] Error processing update for {self.epic}: {e}")

    def onSubscription(self):
        """Called when subscription is established."""
        logger.info(f"[MARKET STREAM] ✅ Subscribed to MARKET:{self.epic}")

    def onSubscriptionError(self, code, message):
        """Called on subscription error."""
        logger.error(f"[MARKET STREAM] ❌ Subscription error {self.epic}: {code} - {message}")

    def onUnsubscription(self):
        """Called when unsubscribed."""
        logger.info(f"[MARKET STREAM] Unsubscribed from MARKET:{self.epic}")


class MarketPriceStreamManager:
    """
    Manages Lightstreamer connections for real-time market price streaming.

    Handles:
    - Connection establishment and authentication
    - MARKET:{epic} subscriptions for BID/OFFER prices
    - Automatic reconnection on connection loss
    - Thread-safe subscription management
    """

    def __init__(self, auth_headers: dict):
        """
        Initialize stream manager.

        Args:
            auth_headers: IG API auth headers with CST, X-SECURITY-TOKEN, accountId
        """
        if not LIGHTSTREAMER_AVAILABLE:
            logger.warning("[MARKET STREAM] Lightstreamer library not installed. "
                         "Install with: pip install lightstreamer-client-lib")

        self.auth_headers = auth_headers
        self.client: Optional[LightstreamerClient] = None
        self.subscriptions: Dict[str, Subscription] = {}
        self.listeners: Dict[str, MarketPriceListener] = {}
        self.is_connected = False
        self._lock = Lock()
        self._reconnect_task: Optional[asyncio.Task] = None
        self._monitor_task: Optional[asyncio.Task] = None
        self._reconnect_attempts = 0

        # Get URL from config
        self.lightstreamer_url = get_lightstreamer_url()

    async def connect(self) -> bool:
        """
        Establish Lightstreamer connection.

        Returns:
            True if connected successfully, False otherwise
        """
        if not LIGHTSTREAMER_AVAILABLE:
            logger.error("[MARKET STREAM] Cannot connect - Lightstreamer library not installed")
            return False

        try:
            logger.info(f"[MARKET STREAM] Connecting to {self.lightstreamer_url}...")

            self.client = LightstreamerClient(self.lightstreamer_url, "DEFAULT")

            # Set authentication
            account_id = self.auth_headers.get('accountId')
            cst = self.auth_headers.get('CST')
            xst = self.auth_headers.get('X-SECURITY-TOKEN')

            if not all([account_id, cst, xst]):
                logger.error("[MARKET STREAM] Missing authentication headers")
                return False

            password = f"CST-{cst}|XST-{xst}"
            self.client.connectionDetails.setUser(account_id)
            self.client.connectionDetails.setPassword(password)

            # Add connection listener
            self.client.addListener(self._create_connection_listener())

            # Connect
            self.client.connect()

            # Wait for connection to establish
            for _ in range(LIGHTSTREAMER_CONNECT_TIMEOUT):
                if self.is_connected:
                    logger.info("[MARKET STREAM] ✅ Connected to Lightstreamer")
                    self._reconnect_attempts = 0

                    # Start connection monitor if auto-reconnect enabled
                    if LIGHTSTREAMER_AUTO_RECONNECT:
                        self._monitor_task = asyncio.create_task(self._monitor_connection())

                    return True
                await asyncio.sleep(1)

            logger.error("[MARKET STREAM] ❌ Connection timeout")
            return False

        except Exception as e:
            logger.error(f"[MARKET STREAM] ❌ Connection failed: {e}")
            return False

    def subscribe(self, epic: str, callback: Callable[[MarketPrice], None]) -> bool:
        """
        Subscribe to market prices for an epic.

        Args:
            epic: Market epic (e.g., 'CS.D.EURUSD.CEEM.IP')
            callback: Function to call with MarketPrice on each update

        Returns:
            True if subscription successful
        """
        if not LIGHTSTREAMER_AVAILABLE or not self.client:
            logger.error(f"[MARKET STREAM] Cannot subscribe - not connected")
            return False

        with self._lock:
            if epic in self.subscriptions:
                logger.debug(f"[MARKET STREAM] Already subscribed to {epic}")
                return True

            try:
                # Create MARKET subscription
                item = f"MARKET:{epic}"
                fields = ["BID", "OFFER", "UPDATE_TIME", "MARKET_STATE"]

                subscription = Subscription(
                    mode="MERGE",
                    items=[item],
                    fields=fields
                )

                # Create listener with callback
                listener = MarketPriceListener(epic, callback)
                subscription.addListener(listener)

                # Subscribe
                self.client.subscribe(subscription)

                self.subscriptions[epic] = subscription
                self.listeners[epic] = listener

                logger.info(f"[MARKET STREAM] 📊 Subscribing to MARKET:{epic}")
                return True

            except Exception as e:
                logger.error(f"[MARKET STREAM] ❌ Failed to subscribe to {epic}: {e}")
                return False

    def unsubscribe(self, epic: str) -> bool:
        """
        Unsubscribe from market prices for an epic.

        Args:
            epic: Market epic

        Returns:
            True if unsubscribed successfully
        """
        with self._lock:
            if epic not in self.subscriptions:
                return True

            try:
                subscription = self.subscriptions.pop(epic)
                self.listeners.pop(epic, None)

                if self.client:
                    self.client.unsubscribe(subscription)

                logger.info(f"[MARKET STREAM] Unsubscribed from {epic}")
                return True

            except Exception as e:
                logger.error(f"[MARKET STREAM] Error unsubscribing from {epic}: {e}")
                return False

    def unsubscribe_all(self):
        """Unsubscribe from all epics."""
        with self._lock:
            epics = list(self.subscriptions.keys())

        for epic in epics:
            self.unsubscribe(epic)

    def disconnect(self):
        """Disconnect from Lightstreamer and cleanup."""
        logger.info("[MARKET STREAM] Disconnecting...")

        # Cancel monitor task
        if self._monitor_task:
            self._monitor_task.cancel()

        # Unsubscribe all
        self.unsubscribe_all()

        # Disconnect client
        with self._lock:
            if self.client:
                try:
                    self.client.disconnect()
                except Exception as e:
                    logger.warning(f"[MARKET STREAM] Error during disconnect: {e}")
                finally:
                    self.client = None

            self.is_connected = False
            self.subscriptions.clear()
            self.listeners.clear()

        logger.info("[MARKET STREAM] Disconnected")

    def get_last_price(self, epic: str) -> Optional[MarketPrice]:
        """
        Get the last known price for an epic.

        Args:
            epic: Market epic

        Returns:
            Last MarketPrice or None if not available
        """
        with self._lock:
            listener = self.listeners.get(epic)
            return listener.last_update if listener else None

    def get_subscribed_epics(self) -> List[str]:
        """Get list of currently subscribed epics."""
        with self._lock:
            return list(self.subscriptions.keys())

    def get_status(self) -> dict:
        """Get stream manager status for monitoring."""
        with self._lock:
            return {
                "connected": self.is_connected,
                "lightstreamer_url": self.lightstreamer_url,
                "subscriptions_count": len(self.subscriptions),
                "subscribed_epics": list(self.subscriptions.keys()),
                "reconnect_attempts": self._reconnect_attempts,
            }

    def _create_connection_listener(self):
        """Create Lightstreamer connection status listener."""
        manager = self

        class ConnectionStatusListener:
            def onStatusChange(self, status):
                logger.info(f"[MARKET STREAM] Connection status: {status}")
                manager.is_connected = status in [
                    "CONNECTED:STREAM-SENSING",
                    "CONNECTED:WS-STREAMING",
                    "CONNECTED:HTTP-STREAMING"
                ]

            def onServerError(self, code, message):
                logger.error(f"[MARKET STREAM] Server error: {code} - {message}")
                manager.is_connected = False

        return ConnectionStatusListener()

    async def _monitor_connection(self):
        """Monitor connection health and trigger reconnection if needed."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds

                if not self.is_connected:
                    logger.warning("[MARKET STREAM] ⚠️ Connection lost, attempting reconnection...")
                    await self._reconnect()

            except asyncio.CancelledError:
                logger.info("[MARKET STREAM] Connection monitor stopped")
                break
            except Exception as e:
                logger.error(f"[MARKET STREAM] Error in connection monitor: {e}")

    async def _reconnect(self):
        """Attempt to reconnect with exponential backoff."""
        if self._reconnect_attempts >= LIGHTSTREAMER_MAX_RECONNECT_ATTEMPTS:
            logger.error("[MARKET STREAM] ❌ Max reconnection attempts reached")
            return

        self._reconnect_attempts += 1
        wait_time = min(300, LIGHTSTREAMER_RECONNECT_DELAY_SECONDS * (2 ** self._reconnect_attempts))

        logger.info(f"[MARKET STREAM] 🔄 Reconnection attempt {self._reconnect_attempts} in {wait_time}s...")
        await asyncio.sleep(wait_time)

        try:
            # Store current subscriptions to restore after reconnect
            with self._lock:
                epics_to_restore = list(self.subscriptions.keys())
                callbacks_to_restore = {
                    epic: listener.callback
                    for epic, listener in self.listeners.items()
                }

            # Disconnect old client
            if self.client:
                try:
                    self.client.disconnect()
                except:
                    pass
                self.client = None

            self.subscriptions.clear()
            self.listeners.clear()

            # Reconnect
            connected = await self.connect()

            if connected:
                # Restore subscriptions
                for epic in epics_to_restore:
                    callback = callbacks_to_restore.get(epic)
                    if callback:
                        self.subscribe(epic, callback)

                logger.info(f"[MARKET STREAM] ✅ Reconnected, restored {len(epics_to_restore)} subscriptions")
            else:
                logger.error("[MARKET STREAM] ❌ Reconnection failed")

        except Exception as e:
            logger.error(f"[MARKET STREAM] ❌ Reconnection error: {e}")
