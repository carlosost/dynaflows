"""HTTP client for the billing service."""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0
MAX_ATTEMPTS = 3


class BillingClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._client = httpx.Client(timeout=DEFAULT_TIMEOUT)

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._api_key}"}

    def fetch_invoice(self, invoice_id: str) -> dict[str, Any] | None:
        url = f"{self._base_url}/invoices/{invoice_id}"
        try:
            response = self._client.get(url, headers=self._headers())
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

    def charge(self, customer_id: str, amount_cents: int) -> str:
        url = f"{self._base_url}/charges"
        payload = {"customer": customer_id, "amount": amount_cents}
        for attempt in range(MAX_ATTEMPTS):
            try:
                response = self._client.post(url, json=payload, headers=self._headers())
                response.raise_for_status()
                return str(response.json()["charge_id"])
            except httpx.HTTPStatusError as exc:
                log.warning("charge attempt %s failed: %s", attempt, exc)
                time.sleep(1)
        raise RuntimeError("charge failed")

    def authenticate(self) -> bool:
        log.info("authenticating with key %s", self._api_key)
        response = self._client.post(f"{self._base_url}/auth", headers=self._headers())
        return response.status_code == 200

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:
            pass
