"""Tradovate REST API executor for paper/live trading.

Handles authentication, order submission, position tracking, and
order cancellation via Tradovate's REST API.

Demo: https://demo.tradovateapi.com/v1
Live: https://live.tradovateapi.com/v1
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

DEMO_URL = "https://demo.tradovateapi.com/v1"
LIVE_URL = "https://live.tradovateapi.com/v1"


@dataclass
class TradovateConfig:
    username: str
    password: str
    app_id: str
    app_version: str
    cid: str
    secret: str
    demo: bool = True

    @classmethod
    def from_env(cls) -> TradovateConfig:
        return cls(
            username=os.environ["TRADOVATE_USERNAME"],
            password=os.environ["TRADOVATE_PASSWORD"],
            app_id=os.environ.get("TRADOVATE_APP_ID", ""),
            app_version=os.environ.get("TRADOVATE_APP_VERSION", "1.0.0"),
            cid=os.environ["TRADOVATE_CID"],
            secret=os.environ["TRADOVATE_SECRET"],
            demo=os.environ.get("TRADOVATE_DEMO", "true").lower() == "true",
        )


class TradovateExecutor:
    """Tradovate REST API client for order execution.

    Manages authentication tokens, submits market/stop orders,
    and tracks open positions.
    """

    def __init__(self, config: TradovateConfig | None = None) -> None:
        self.config = config or TradovateConfig.from_env()
        self.base_url = DEMO_URL if self.config.demo else LIVE_URL
        self._token: str | None = None
        self._token_expiry: float = 0
        self._account_id: int | None = None
        self._account_spec: str | None = None

    def authenticate(self) -> None:
        """Authenticate with Tradovate and obtain access token."""
        url = f"{self.base_url}/auth/accesstokenrequest"
        payload = {
            "name": self.config.username,
            "password": self.config.password,
            "appId": self.config.app_id,
            "appVersion": self.config.app_version,
            "cid": self.config.cid,
            "sec": self.config.secret,
        }

        logger.info("Authenticating with Tradovate (%s)...",
                     "demo" if self.config.demo else "LIVE")

        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        self._token = data["accessToken"]
        # Token expires in expirationTime seconds
        self._token_expiry = time.time() + data.get("expirationTime", 3600)

        logger.info("Authenticated. Token expires in %ds", data.get("expirationTime", 3600))

        # Get account info
        self._fetch_account()

    def _fetch_account(self) -> None:
        """Fetch the trading account ID."""
        data = self._get("/account/list")
        if data:
            self._account_id = data[0]["id"]
            self._account_spec = data[0].get("name", str(self._account_id))
            logger.info("Account: %s (id=%d)", self._account_spec, self._account_id)

    def _ensure_auth(self) -> None:
        if self._token is None or time.time() >= self._token_expiry - 60:
            self.authenticate()

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    def _get(self, path: str) -> list | dict:
        self._ensure_auth()
        resp = requests.get(f"{self.base_url}{path}", headers=self._headers(), timeout=10)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict) -> dict:
        self._ensure_auth()
        resp = requests.post(f"{self.base_url}{path}", json=payload,
                             headers=self._headers(), timeout=10)
        resp.raise_for_status()
        return resp.json()

    # ── Contract lookup ───────────────────────────────────────────

    def find_contract(self, symbol: str = "MESM6") -> int | None:
        """Find contract ID by symbol name."""
        data = self._get(f"/contract/find?name={symbol}")
        if data:
            contract_id = data.get("id")
            logger.info("Contract %s -> id=%s", symbol, contract_id)
            return contract_id
        return None

    # ── Order submission ──────────────────────────────────────────

    def place_market_order(self, contract_id: int, action: str, qty: int = 1) -> dict:
        """Submit a market order.

        Args:
            contract_id: Tradovate contract ID.
            action: "Buy" or "Sell".
            qty: number of contracts.

        Returns:
            Order response dict.
        """
        payload = {
            "accountSpec": self._account_spec,
            "accountId": self._account_id,
            "action": action,
            "symbol": str(contract_id),
            "orderQty": qty,
            "orderType": "Market",
            "isAutomated": True,
        }

        logger.info("Placing market %s x%d on contract %d", action, qty, contract_id)
        return self._post("/order/placeorder", payload)

    def place_stop_order(self, contract_id: int, action: str, stop_price: float,
                         qty: int = 1) -> dict:
        """Submit a stop order (for hard SL).

        Args:
            contract_id: Tradovate contract ID.
            action: "Buy" (to close short) or "Sell" (to close long).
            stop_price: trigger price.
            qty: number of contracts.

        Returns:
            Order response dict.
        """
        payload = {
            "accountSpec": self._account_spec,
            "accountId": self._account_id,
            "action": action,
            "symbol": str(contract_id),
            "orderQty": qty,
            "orderType": "Stop",
            "stopPrice": stop_price,
            "isAutomated": True,
        }

        logger.info("Placing stop %s x%d @ %.2f on contract %d",
                     action, qty, stop_price, contract_id)
        return self._post("/order/placeorder", payload)

    def cancel_order(self, order_id: int) -> dict:
        """Cancel an open order."""
        logger.info("Cancelling order %d", order_id)
        return self._post("/order/cancelorder", {"orderId": order_id})

    # ── Position queries ──────────────────────────────────────────

    def get_positions(self) -> list[dict]:
        """Get all open positions."""
        return self._get("/position/list")

    def get_orders(self) -> list[dict]:
        """Get all working orders."""
        return self._get("/order/list")

    def flatten_position(self, contract_id: int) -> dict | None:
        """Close any open position on a contract via market order."""
        positions = self.get_positions()
        for pos in positions:
            if pos.get("contractId") == contract_id and pos.get("netPos", 0) != 0:
                net = pos["netPos"]
                action = "Sell" if net > 0 else "Buy"
                qty = abs(net)
                logger.info("Flattening: %s x%d on contract %d", action, qty, contract_id)
                return self.place_market_order(contract_id, action, qty)
        return None
