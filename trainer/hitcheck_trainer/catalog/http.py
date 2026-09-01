"""Real-network adapters matching the transport contracts used by
`CatalogApi` and `download_images`. Connection errors surface as status
0, which both retry loops treat as retryable."""

import httpx


def httpx_transport(timeout: float = 30.0):
    client = httpx.Client(timeout=timeout, follow_redirects=True)

    def transport(url: str, headers: dict):
        try:
            response = client.get(url, headers=headers)
        except httpx.HTTPError:
            return 0, None
        if response.status_code != 200:
            return response.status_code, None
        try:
            return 200, response.json()
        except ValueError:
            return 0, None

    return transport


def httpx_fetch(timeout: float = 30.0):
    client = httpx.Client(timeout=timeout, follow_redirects=True)

    def fetch(url: str):
        try:
            response = client.get(url)
        except httpx.HTTPError:
            return 0, None
        return response.status_code, response.content if response.status_code == 200 else None

    return fetch


def httpx_post_form(timeout: float = 30.0):
    """Form-encoded POST returning (status, json). For eBay's OAuth grant.

    The GET transports above cover every other call in this repo; the
    token endpoint is the one place a POST is needed.
    """
    client = httpx.Client(timeout=timeout, follow_redirects=True)

    def post(url: str, headers: dict, data: dict):
        try:
            response = client.post(url, headers=headers, data=data)
        except httpx.HTTPError:
            return 0, None
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, None

    return post
