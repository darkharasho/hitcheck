"""eBay OAuth and Browse API access.

The only network-touching file in the corpus package. Shaped after
catalog/api.py: an injected transport, retries driven by
catalog/backoff.py, and a raised error when the attempts run out --
never a silently empty result, which would look like "no listings
matched" and quietly shrink the corpus instead of surfacing an outage.

Nothing here logs the credentials or the token. Error messages carry the
HTTP status and nothing else.
"""

import base64
import re
import time
import urllib.parse

from ..catalog.backoff import backoff_delays

# The /v1/ segment is mandatory. Omitting it returns 404, which reads
# like a credentials failure and is a genuinely expensive misdiagnosis.
EBAY_OAUTH_URL = "https://api.ebay.com/identity/v1/oauth2/token"
BROWSE_BASE = "https://api.ebay.com/buy/browse/v1"
OAUTH_SCOPE = "https://api.ebay.com/oauth/api_scope"
MARKETPLACE_ID = "EBAY_US"

RETRYABLE = {0, 429, 500, 502, 503, 504}

_SIZE_SUFFIX = re.compile(r"s-l\d+(?=\.\w+$)")


class EbayError(Exception):
    pass


def basic_auth_header(app_id: str, cert_id: str) -> str:
    token = base64.b64encode(f"{app_id}:{cert_id}".encode()).decode()
    return f"Basic {token}"


def fetch_token(post, app_id: str, cert_id: str, max_attempts: int = 5,
                sleep=time.sleep) -> str:
    """Client-credentials grant. `post(url, headers, data) -> (status, json)`.

    Retried on the same schedule as every other network call in the repo --
    backoff_delays, not a second policy. A transient 503 here would
    otherwise abort the whole acquisition run with a message that reads
    like a credentials failure, which is an expensive misdiagnosis.
    """
    headers = {
        "Authorization": basic_auth_header(app_id, cert_id),
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"grant_type": "client_credentials", "scope": OAUTH_SCOPE}
    delays = backoff_delays(max_attempts - 1)
    last_status = None
    for attempt in range(max_attempts):
        status, body = post(EBAY_OAUTH_URL, headers, data)
        last_status = status
        if status == 200 and body and body.get("access_token"):
            return body["access_token"]
        # A 200 carrying no usable token is malformed, not a success and
        # not a hard failure -- the same call BrowseClient._get makes.
        if not (status in RETRYABLE or status == 200):
            break
        if attempt < len(delays):
            sleep(delays[attempt])
    # The status and nothing else: never the credentials, never the token.
    raise EbayError(f"oauth token request failed (status {last_status})")


def hi_res_url(url: str) -> str:
    """Swap eBay's thumbnail size suffix for the full-resolution one.

    Summaries carry s-l225 (measured 138x225); s-l1600 on the same URL
    returned 734x1200 for the same listing.
    """
    return _SIZE_SUFFIX.sub("s-l1600", url or "")


def aspects_from_item(item: dict) -> dict[str, str]:
    """Flatten a detail response's localizedAspects into name -> value.

    Structured Item Specifics, not the listing title, are the label
    source: titles are seller free text and parsing them would put label
    noise directly under the accuracy number.
    """
    flat = {}
    for aspect in item.get("localizedAspects") or []:
        name, value = aspect.get("name"), aspect.get("value")
        if name and value:
            flat[name] = value
    return flat


class BrowseClient:
    def __init__(self, transport, token: str, marketplace: str = MARKETPLACE_ID,
                 max_attempts: int = 5, sleep=time.sleep):
        self._transport = transport
        self._token = token
        self._marketplace = marketplace
        self._max_attempts = max_attempts
        self._sleep = sleep

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._token}",
            "X-EBAY-C-MARKETPLACE-ID": self._marketplace,
        }

    def _get(self, url: str) -> dict:
        delays = backoff_delays(self._max_attempts - 1)
        last_status = None
        for attempt in range(self._max_attempts):
            status, body = self._transport(url, self._headers())
            last_status = status
            if status == 200 and body is not None:
                return body
            # A 200 with no body is malformed, not a success and not a
            # hard failure — same call as catalog/api.py makes.
            retryable = status in RETRYABLE or (status == 200 and body is None)
            if not retryable:
                break
            if attempt < len(delays):
                self._sleep(delays[attempt])
        raise EbayError(f"{url} failed after {self._max_attempts} attempts (last status {last_status})")

    def search(self, query: str, limit: int = 200, offset: int = 0,
               extra_filter: str | None = None) -> list[dict]:
        """One page of item summaries. `limit` maxes out at 200 server-side."""
        params = {"q": query, "limit": str(limit), "offset": str(offset)}
        if extra_filter:
            params["filter"] = extra_filter
        url = f"{BROWSE_BASE}/item_summary/search?{urllib.parse.urlencode(params)}"
        return self._get(url).get("itemSummaries") or []

    def item(self, item_id: str) -> dict:
        """Full detail for one listing — this is where localizedAspects live."""
        return self._get(f"{BROWSE_BASE}/item/{urllib.parse.quote(item_id, safe='')}")
