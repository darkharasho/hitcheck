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
