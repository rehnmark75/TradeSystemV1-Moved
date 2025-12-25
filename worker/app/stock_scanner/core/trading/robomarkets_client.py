"""
RoboMarkets StocksTrader API Client

API Documentation: https://api-doc.stockstrader.com/

This client handles:
- Authentication via Bearer token
- Fetching available instruments/tickers
- Getting real-time quotes
- Placing, modifying, and cancelling orders
- Managing positions (deals)

Note: RoboMarkets API does NOT provide historical OHLCV data.
Use yfinance for historical data.
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    ACTIVE = "active"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Instrument:
    """Stock instrument from RoboMarkets"""
    ticker: str
    name: str
    contract_size: float
    min_quantity: float
    max_quantity: float
    quantity_step: float
    currency: str
    exchange: Optional[str] = None
    is_tradeable: bool = True
    metadata: Optional[Dict] = None


@dataclass
class Quote:
    """Real-time quote from RoboMarkets"""
    ticker: str
    ask: float
    bid: float
    last: float
    timestamp: datetime
    spread: float = 0.0


@dataclass
class Order:
    """Order representation"""
    order_id: str
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    status: OrderStatus
    created_at: datetime
    filled_price: Optional[float] = None
    filled_quantity: Optional[float] = None


@dataclass
class Position:
    """Open position (deal) representation"""
    deal_id: str
    ticker: str
    side: str  # 'long' or 'short'
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    opened_at: datetime


@dataclass
class AccountBalance:
    """Account balance information"""
    total_value: float  # my_portfolio - total account value
    invested: float  # investments - value in positions
    available: float  # available_to_invest - free cash
    timestamp: datetime


class RoboMarketsError(Exception):
    """Base exception for RoboMarkets API errors"""
    def __init__(self, message: str, code: str = None, response: Dict = None):
        self.message = message
        self.code = code
        self.response = response
        super().__init__(self.message)


class RoboMarketsAuthError(RoboMarketsError):
    """Authentication error"""
    pass


class RoboMarketsOrderError(RoboMarketsError):
    """Order-related error"""
    pass


class RoboMarketsClient:
    """
    Async client for RoboMarkets StocksTrader API

    Usage:
        client = RoboMarketsClient(api_key="...", account_id="...")
        async with client:
            instruments = await client.get_instruments()
            quote = await client.get_quote("AAPL")
            order = await client.place_order("AAPL", "buy", 10)
    """

    BASE_URL = "https://api.stockstrader.com/api/v1"

    def __init__(
        self,
        api_key: str,
        account_id: str,
        base_url: str = None,
        timeout: int = 30,
        max_retries: int = 3
    ):
        """
        Initialize RoboMarkets client

        Args:
            api_key: API key for authentication
            account_id: Trading account ID
            base_url: Optional custom API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts for failed requests
        """
        self.api_key = api_key
        self.account_id = account_id
        self.base_url = base_url or self.BASE_URL
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None
        self._instruments_cache: Dict[str, Instrument] = {}
        self._cache_timestamp: Optional[datetime] = None
        self._cache_ttl = 300  # 5 minutes

    @property
    def headers(self) -> Dict[str, str]:
        """Get request headers with authentication"""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json"
        }

    async def __aenter__(self):
        """Async context manager entry"""
        await self._create_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()

    async def _create_session(self):
        """Create aiohttp session"""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=self.timeout,
                headers=self.headers
            )

    async def close(self):
        """Close the HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Dict = None,
        params: Dict = None,
        retry_count: int = 0
    ) -> Dict:
        """
        Make an API request with error handling and retries

        Args:
            method: HTTP method (GET, POST, PUT, DELETE)
            endpoint: API endpoint path
            data: Request body data
            params: Query parameters
            retry_count: Current retry attempt

        Returns:
            API response data

        Raises:
            RoboMarketsError: On API errors
            RoboMarketsAuthError: On authentication errors
        """
        await self._create_session()

        url = f"{self.base_url}{endpoint}"

        try:
            async with self._session.request(
                method=method,
                url=url,
                data=data,
                params=params
            ) as response:
                response_data = await response.json()

                # Check for API errors
                if response.status == 401:
                    raise RoboMarketsAuthError(
                        "Authentication failed. Check API key.",
                        code="auth_error",
                        response=response_data
                    )

                if response.status >= 400:
                    error_msg = response_data.get("msg", "Unknown error")
                    raise RoboMarketsError(
                        f"API error: {error_msg}",
                        code=response_data.get("code"),
                        response=response_data
                    )

                # Check response code
                if response_data.get("code") == "error":
                    raise RoboMarketsError(
                        response_data.get("msg", "Unknown error"),
                        code="error",
                        response=response_data
                    )

                return response_data.get("data", response_data)

        except aiohttp.ClientError as e:
            if retry_count < self.max_retries:
                await asyncio.sleep(2 ** retry_count)  # Exponential backoff
                return await self._request(
                    method, endpoint, data, params, retry_count + 1
                )
            raise RoboMarketsError(f"Connection error: {str(e)}")

    # =========================================================================
    # INSTRUMENTS
    # =========================================================================

    async def get_instruments(self, use_cache: bool = True) -> List[Instrument]:
        """
        Get all available trading instruments

        Args:
            use_cache: Whether to use cached instruments

        Returns:
            List of Instrument objects
        """
        # Check cache
        if use_cache and self._instruments_cache:
            if self._cache_timestamp:
                age = (datetime.now() - self._cache_timestamp).seconds
                if age < self._cache_ttl:
                    return list(self._instruments_cache.values())

        endpoint = f"/accounts/{self.account_id}/instruments"

        try:
            data = await self._request("GET", endpoint)
            instruments = []

            for item in data.get("instruments", []):
                instrument = Instrument(
                    ticker=item.get("ticker", ""),
                    name=item.get("description", item.get("ticker", "")),
                    contract_size=float(item.get("contractSize", 1)),
                    min_quantity=float(item.get("minOrderQuantity", 1)),
                    max_quantity=float(item.get("maxOrderQuantity", 10000)),
                    quantity_step=float(item.get("quantityStep", 1)),
                    currency=item.get("currency", "USD"),
                    exchange=item.get("exchange"),
                    is_tradeable=item.get("tradeable", True),
                    metadata=item
                )
                instruments.append(instrument)
                self._instruments_cache[instrument.ticker] = instrument

            self._cache_timestamp = datetime.now()
            logger.info(f"Fetched {len(instruments)} instruments from RoboMarkets")
            return instruments

        except RoboMarketsError as e:
            logger.error(f"Failed to fetch instruments: {e.message}")
            raise

    async def get_instrument(self, ticker: str) -> Optional[Instrument]:
        """
        Get a specific instrument by ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            Instrument object or None if not found
        """
        if ticker in self._instruments_cache:
            return self._instruments_cache[ticker]

        # Fetch all instruments to populate cache
        await self.get_instruments()
        return self._instruments_cache.get(ticker)

    # =========================================================================
    # QUOTES
    # =========================================================================

    async def get_quote(self, ticker: str) -> Quote:
        """
        Get real-time quote for a ticker

        Args:
            ticker: Stock ticker symbol

        Returns:
            Quote object with current prices
        """
        endpoint = f"/accounts/{self.account_id}/instruments/{ticker}/quote"

        try:
            data = await self._request("GET", endpoint)

            ask = float(data.get("ask", 0))
            bid = float(data.get("bid", 0))
            last = float(data.get("last", 0))

            return Quote(
                ticker=ticker,
                ask=ask,
                bid=bid,
                last=last,
                timestamp=datetime.utcnow(),
                spread=ask - bid if ask and bid else 0
            )

        except RoboMarketsError as e:
            logger.error(f"Failed to get quote for {ticker}: {e.message}")
            raise

    async def get_quotes(self, tickers: List[str]) -> Dict[str, Quote]:
        """
        Get quotes for multiple tickers

        Args:
            tickers: List of ticker symbols

        Returns:
            Dict mapping ticker to Quote
        """
        quotes = {}
        tasks = [self.get_quote(ticker) for ticker in tickers]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for ticker, result in zip(tickers, results):
            if isinstance(result, Quote):
                quotes[ticker] = result
            else:
                logger.warning(f"Failed to get quote for {ticker}: {result}")

        return quotes

    # =========================================================================
    # ORDERS
    # =========================================================================

    async def place_order(
        self,
        ticker: str,
        side: str,  # 'buy' or 'sell'
        quantity: float,
        order_type: str = "market",
        price: float = None,
        stop_loss: float = None,
        take_profit: float = None
    ) -> Order:
        """
        Place a trading order

        Args:
            ticker: Stock ticker symbol
            side: Order side ('buy' or 'sell')
            quantity: Number of shares
            order_type: Order type ('market', 'limit', 'stop')
            price: Limit/stop price (required for limit/stop orders)
            stop_loss: Stop loss price
            take_profit: Take profit price

        Returns:
            Order object with order details
        """
        endpoint = f"/accounts/{self.account_id}/orders"

        data = {
            "ticker": ticker,
            "side": side.lower(),
            "type": order_type.lower(),
            "quantity": str(quantity)
        }

        if price and order_type.lower() in ("limit", "stop"):
            data["price"] = str(price)

        if stop_loss:
            data["stopLoss"] = str(stop_loss)

        if take_profit:
            data["takeProfit"] = str(take_profit)

        try:
            result = await self._request("POST", endpoint, data=data)

            order = Order(
                order_id=str(result.get("orderId", "")),
                ticker=ticker,
                side=OrderSide(side.lower()),
                order_type=OrderType(order_type.lower()),
                quantity=quantity,
                price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                status=OrderStatus.PENDING,
                created_at=datetime.utcnow()
            )

            logger.info(f"Placed {side} order for {quantity} {ticker}: {order.order_id}")
            return order

        except RoboMarketsError as e:
            logger.error(f"Failed to place order for {ticker}: {e.message}")
            raise RoboMarketsOrderError(
                f"Order failed: {e.message}",
                code=e.code,
                response=e.response
            )

    async def modify_order(
        self,
        order_id: str,
        price: float = None,
        stop_loss: float = None,
        take_profit: float = None
    ) -> Dict:
        """
        Modify an existing limit/stop order

        Args:
            order_id: Order ID to modify
            price: New limit/stop price
            stop_loss: New stop loss price
            take_profit: New take profit price

        Returns:
            Updated order data
        """
        endpoint = f"/accounts/{self.account_id}/orders/{order_id}"

        data = {}
        if price:
            data["price"] = str(price)
        if stop_loss:
            data["stopLoss"] = str(stop_loss)
        if take_profit:
            data["takeProfit"] = str(take_profit)

        try:
            result = await self._request("PUT", endpoint, data=data)
            logger.info(f"Modified order {order_id}")
            return result

        except RoboMarketsError as e:
            logger.error(f"Failed to modify order {order_id}: {e.message}")
            raise

    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an unfulfilled order

        Args:
            order_id: Order ID to cancel

        Returns:
            True if cancelled successfully
        """
        endpoint = f"/accounts/{self.account_id}/orders/{order_id}"

        try:
            await self._request("DELETE", endpoint)
            logger.info(f"Cancelled order {order_id}")
            return True

        except RoboMarketsError as e:
            logger.error(f"Failed to cancel order {order_id}: {e.message}")
            raise

    async def get_orders(
        self,
        status: str = None,
        start_date: datetime = None,
        end_date: datetime = None,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get orders with optional filtering

        Args:
            status: Filter by status ('active', 'filled', 'cancelled')
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum orders to return

        Returns:
            List of order data
        """
        endpoint = f"/accounts/{self.account_id}/orders"

        params = {"limit": limit}
        if status:
            params["status"] = status
        if start_date:
            params["startDate"] = int(start_date.timestamp())
        if end_date:
            params["endDate"] = int(end_date.timestamp())

        try:
            data = await self._request("GET", endpoint, params=params)
            # Handle both list response and dict with "orders" key
            if isinstance(data, list):
                return data
            return data.get("orders", [])

        except RoboMarketsError as e:
            logger.error(f"Failed to get orders: {e.message}")
            raise

    # =========================================================================
    # POSITIONS (DEALS)
    # =========================================================================

    async def get_positions(self) -> List[Position]:
        """
        Get all open positions

        Returns:
            List of Position objects
        """
        endpoint = f"/accounts/{self.account_id}/deals"

        try:
            data = await self._request("GET", endpoint)
            positions = []

            # Handle both list response and dict with "deals" key
            if isinstance(data, list):
                deals = data
            else:
                deals = data.get("deals", [])

            for deal in deals:
                # API uses snake_case: id, volume, open_price, open_time, close_price
                position = Position(
                    deal_id=str(deal.get("id", deal.get("dealId", ""))),
                    ticker=deal.get("ticker", ""),
                    side="long" if deal.get("side") == "buy" else "short",
                    quantity=float(deal.get("volume", deal.get("quantity", 0))),
                    entry_price=float(deal.get("open_price", deal.get("openPrice", 0))),
                    current_price=float(deal.get("close_price", deal.get("currentPrice", 0))),
                    unrealized_pnl=float(deal.get("profit", 0)),
                    stop_loss=float(deal.get("stop_loss", deal.get("stopLoss", 0))) if deal.get("stop_loss") or deal.get("stopLoss") else None,
                    take_profit=float(deal.get("take_profit", deal.get("takeProfit", 0))) if deal.get("take_profit") or deal.get("takeProfit") else None,
                    opened_at=datetime.utcfromtimestamp(deal.get("open_time", deal.get("openTime", 0)))
                )
                positions.append(position)

            return positions

        except RoboMarketsError as e:
            logger.error(f"Failed to get positions: {e.message}")
            raise

    async def modify_position(
        self,
        deal_id: str,
        stop_loss: float = None,
        take_profit: float = None
    ) -> Dict:
        """
        Modify stop loss / take profit on an open position

        Args:
            deal_id: Deal ID to modify
            stop_loss: New stop loss price
            take_profit: New take profit price

        Returns:
            Updated deal data
        """
        endpoint = f"/accounts/{self.account_id}/deals/{deal_id}"

        data = {}
        if stop_loss:
            data["stopLoss"] = str(stop_loss)
        if take_profit:
            data["takeProfit"] = str(take_profit)

        try:
            result = await self._request("PUT", endpoint, data=data)
            logger.info(f"Modified position {deal_id}")
            return result

        except RoboMarketsError as e:
            logger.error(f"Failed to modify position {deal_id}: {e.message}")
            raise

    async def close_position(self, deal_id: str) -> bool:
        """
        Close an open position

        Args:
            deal_id: Deal ID to close

        Returns:
            True if closed successfully
        """
        endpoint = f"/accounts/{self.account_id}/deals/{deal_id}"

        try:
            await self._request("DELETE", endpoint)
            logger.info(f"Closed position {deal_id}")
            return True

        except RoboMarketsError as e:
            logger.error(f"Failed to close position {deal_id}: {e.message}")
            raise

    async def get_deal_history(
        self,
        history_from: datetime = None,
        history_to: datetime = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict]:
        """
        Get closed deals history for statistics.

        Args:
            history_from: Start date for history (default: 30 days ago)
            history_to: End date for history (default: now)
            skip: Number of records to skip for pagination
            limit: Maximum records to return (max 100)

        Returns:
            List of closed deal dictionaries with P&L data
        """
        endpoint = f"/accounts/{self.account_id}/deals"

        # Default to last 30 days if no dates specified
        if history_from is None:
            history_from = datetime.utcnow() - timedelta(days=30)
        if history_to is None:
            history_to = datetime.utcnow()

        params = {
            "history_from": int(history_from.timestamp()),
            "history_to": int(history_to.timestamp()),
            "skip": skip,
            "limit": min(limit, 100)  # API max is 100
        }

        try:
            data = await self._request("GET", endpoint, params=params)

            # Handle both list response and dict with "deals" key
            if isinstance(data, list):
                deals = data
            else:
                deals = data.get("deals", [])

            # Parse and enrich deal data
            parsed_deals = []
            for deal in deals:
                # API uses snake_case: id, volume, open_price, open_time, close_price, close_time
                parsed_deal = {
                    "deal_id": str(deal.get("id", deal.get("dealId", ""))),
                    "ticker": deal.get("ticker", ""),
                    "side": "long" if deal.get("side") == "buy" else "short",
                    "quantity": float(deal.get("volume", deal.get("quantity", 0))),
                    "open_price": float(deal.get("open_price", deal.get("openPrice", 0))),
                    "close_price": float(deal.get("close_price", deal.get("closePrice", 0))) if deal.get("close_price") or deal.get("closePrice") else None,
                    "open_time": datetime.utcfromtimestamp(deal.get("open_time", deal.get("openTime", 0))) if deal.get("open_time") or deal.get("openTime") else None,
                    "close_time": datetime.utcfromtimestamp(deal.get("close_time", deal.get("closeTime", 0))) if deal.get("close_time") or deal.get("closeTime") else None,
                    "profit": float(deal.get("profit", 0)),
                    "profit_pct": 0.0,
                    "status": deal.get("status", "unknown"),
                    "stop_loss": float(deal.get("stop_loss", deal.get("stopLoss", 0))) if deal.get("stop_loss") or deal.get("stopLoss") else None,
                    "take_profit": float(deal.get("take_profit", deal.get("takeProfit", 0))) if deal.get("take_profit") or deal.get("takeProfit") else None,
                    "raw": deal
                }

                # Calculate profit percentage
                if parsed_deal["open_price"] and parsed_deal["open_price"] > 0:
                    if parsed_deal["close_price"]:
                        if parsed_deal["side"] == "long":
                            parsed_deal["profit_pct"] = (
                                (parsed_deal["close_price"] - parsed_deal["open_price"]) /
                                parsed_deal["open_price"]
                            ) * 100
                        else:
                            parsed_deal["profit_pct"] = (
                                (parsed_deal["open_price"] - parsed_deal["close_price"]) /
                                parsed_deal["open_price"]
                            ) * 100

                parsed_deals.append(parsed_deal)

            logger.info(f"Fetched {len(parsed_deals)} closed deals from history")
            return parsed_deals

        except RoboMarketsError as e:
            logger.error(f"Failed to get deal history: {e.message}")
            raise

    async def get_all_deal_history(
        self,
        history_from: datetime = None,
        history_to: datetime = None
    ) -> List[Dict]:
        """
        Get all closed deals history with pagination handling.

        Args:
            history_from: Start date for history
            history_to: End date for history

        Returns:
            Complete list of closed deals
        """
        all_deals = []
        skip = 0
        limit = 100

        while True:
            deals = await self.get_deal_history(
                history_from=history_from,
                history_to=history_to,
                skip=skip,
                limit=limit
            )

            if not deals:
                break

            all_deals.extend(deals)

            if len(deals) < limit:
                break

            skip += limit

        logger.info(f"Fetched total of {len(all_deals)} closed deals")
        return all_deals

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    async def test_connection(self) -> bool:
        """
        Test API connection and authentication

        Returns:
            True if connection successful
        """
        try:
            await self.get_instruments(use_cache=False)
            logger.info("RoboMarkets API connection successful")
            return True
        except RoboMarketsError as e:
            logger.error(f"RoboMarkets API connection failed: {e.message}")
            return False

    def get_cached_instrument(self, ticker: str) -> Optional[Instrument]:
        """
        Get instrument from cache (synchronous)

        Args:
            ticker: Stock ticker symbol

        Returns:
            Cached Instrument or None
        """
        return self._instruments_cache.get(ticker)

    async def validate_ticker(self, ticker: str) -> bool:
        """
        Check if a ticker is valid and tradeable

        Args:
            ticker: Stock ticker symbol

        Returns:
            True if ticker is valid and tradeable
        """
        instrument = await self.get_instrument(ticker)
        return instrument is not None and instrument.is_tradeable

    # =========================================================================
    # ACCOUNT
    # =========================================================================

    async def get_account_balance(self) -> AccountBalance:
        """
        Get account balance information

        Returns:
            AccountBalance object with total value, invested, and available cash
        """
        endpoint = f"/accounts/{self.account_id}"

        try:
            data = await self._request("GET", endpoint)

            # API returns: {'cash': {'my_portfolio': X, 'investments': Y, 'available_to_invest': Z}}
            cash = data.get("cash", {})

            return AccountBalance(
                total_value=float(cash.get("my_portfolio", 0)),
                invested=float(cash.get("investments", 0)),
                available=float(cash.get("available_to_invest", 0)),
                timestamp=datetime.utcnow()
            )

        except RoboMarketsError as e:
            logger.error(f"Failed to get account balance: {e.message}")
            raise
