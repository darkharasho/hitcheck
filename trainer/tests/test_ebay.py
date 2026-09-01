import base64

import pytest

from hitcheck_trainer.corpus.ebay import (
    BROWSE_BASE,
    EBAY_OAUTH_URL,
    BrowseClient,
    EbayError,
    aspects_from_item,
    basic_auth_header,
    fetch_token,
    hi_res_url,
)


def test_oauth_url_keeps_the_mandatory_v1_segment():
    # Dropping /v1/ returns 404, which reads like a credentials problem
    # and cost real debugging time once already.
    assert EBAY_OAUTH_URL == "https://api.ebay.com/identity/v1/oauth2/token"


def test_basic_auth_header_is_base64_of_appid_colon_certid():
    header = basic_auth_header("APP-123", "CERT-456")
    assert header.startswith("Basic ")
    decoded = base64.b64decode(header.split(" ", 1)[1]).decode()
    assert decoded == "APP-123:CERT-456"


def test_fetch_token_posts_client_credentials_and_returns_the_token():
    seen = {}

    def post(url, headers, data):
        seen.update(url=url, headers=headers, data=data)
        return 200, {"access_token": "v^1.1#tok", "expires_in": 7200}

    assert fetch_token(post, "APP-123", "CERT-456") == "v^1.1#tok"
    assert seen["url"] == EBAY_OAUTH_URL
    assert seen["data"]["grant_type"] == "client_credentials"
    assert seen["data"]["scope"] == "https://api.ebay.com/oauth/api_scope"
    assert seen["headers"]["Content-Type"] == "application/x-www-form-urlencoded"
    assert seen["headers"]["Authorization"].startswith("Basic ")


def test_fetch_token_raises_without_leaking_the_credentials():
    with pytest.raises(EbayError) as exc:
        fetch_token(lambda url, headers, data: (401, None), "APP-SECRET", "CERT-SECRET")
    message = str(exc.value)
    assert "401" in message
    assert "APP-SECRET" not in message
    assert "CERT-SECRET" not in message


def test_hi_res_url_swaps_the_size_suffix():
    assert hi_res_url("https://i.ebayimg.com/images/g/abc/s-l225.jpg") == (
        "https://i.ebayimg.com/images/g/abc/s-l1600.jpg"
    )
    assert hi_res_url("https://i.ebayimg.com/images/g/abc/s-l500.jpg").endswith("s-l1600.jpg")


def test_hi_res_url_leaves_a_url_with_no_size_suffix_alone():
    assert hi_res_url("https://example.com/photo.jpg") == "https://example.com/photo.jpg"


def test_aspects_from_item_flattens_localized_aspects():
    item = {"localizedAspects": [
        {"type": "STRING", "name": "Card Name", "value": "Charizard ex"},
        {"type": "STRING", "name": "Language", "value": "English"},
    ]}
    assert aspects_from_item(item) == {"Card Name": "Charizard ex", "Language": "English"}


def test_aspects_from_item_of_a_listing_with_no_specifics_is_empty_not_a_crash():
    assert aspects_from_item({}) == {}
    assert aspects_from_item({"localizedAspects": None}) == {}


def test_aspects_from_item_skips_entries_missing_a_name_or_value():
    item = {"localizedAspects": [
        {"name": "Set"},
        {"value": "orphan"},
        {"name": "Set", "value": "151"},
    ]}
    assert aspects_from_item(item) == {"Set": "151"}


def fake_transport(responses):
    """responses: list of (status, body). Records every url and header seen."""
    calls = []

    def transport(url, headers):
        calls.append((url, headers))
        return responses[min(len(calls) - 1, len(responses) - 1)]

    return transport, calls


def test_search_sends_the_bearer_token_and_marketplace_header():
    transport, calls = fake_transport([(200, {"itemSummaries": [{"itemId": "v1|1|0"}]})])
    client = BrowseClient(transport, token="v^1.1#tok")
    summaries = client.search("charizard psa 10")
    assert summaries == [{"itemId": "v1|1|0"}]
    url, headers = calls[0]
    assert url.startswith(f"{BROWSE_BASE}/item_summary/search")
    assert headers["Authorization"] == "Bearer v^1.1#tok"
    assert headers["X-EBAY-C-MARKETPLACE-ID"] == "EBAY_US"


def test_search_url_encodes_the_query_and_carries_limit_and_offset():
    transport, calls = fake_transport([(200, {"itemSummaries": []})])
    BrowseClient(transport, token="t").search("charizard & pikachu", limit=50, offset=100)
    url = calls[0][0]
    assert "q=charizard+%26+pikachu" in url
    assert "limit=50" in url
    assert "offset=100" in url


def test_search_of_a_page_with_no_results_returns_an_empty_list():
    transport, _ = fake_transport([(200, {})])
    assert BrowseClient(transport, token="t").search("nothing") == []


def test_item_fetches_the_detail_endpoint_for_one_listing():
    transport, calls = fake_transport([(200, {"itemId": "v1|1|0", "localizedAspects": []})])
    client = BrowseClient(transport, token="t")
    assert client.item("v1|1|0")["itemId"] == "v1|1|0"
    assert calls[0][0] == f"{BROWSE_BASE}/item/v1%7C1%7C0"


def test_a_retryable_status_is_retried_with_backoff_then_succeeds():
    transport, calls = fake_transport([(503, None), (429, None), (200, {"itemSummaries": []})])
    slept = []
    client = BrowseClient(transport, token="t", sleep=slept.append)
    client.search("x")
    assert len(calls) == 3
    assert len(slept) == 2
    assert slept == sorted(slept)  # backoff grows


def test_a_non_retryable_status_fails_immediately():
    transport, calls = fake_transport([(400, None)])
    with pytest.raises(EbayError):
        BrowseClient(transport, token="t", sleep=lambda s: None).search("x")
    assert len(calls) == 1


def test_exhausted_retries_raise_rather_than_returning_empty():
    # An empty list would look like "no listings matched" and silently
    # shrink the corpus instead of surfacing an outage.
    transport, _ = fake_transport([(503, None)])
    with pytest.raises(EbayError) as exc:
        BrowseClient(transport, token="t", max_attempts=3, sleep=lambda s: None).search("x")
    assert "503" in str(exc.value)


def test_a_200_with_no_body_is_treated_as_retryable_not_as_success():
    # Matches catalog/api.py: a 200 with junk body is a malformed
    # response, not a real success and not a hard failure.
    transport, calls = fake_transport([(200, None), (200, {"itemSummaries": []})])
    BrowseClient(transport, token="t", sleep=lambda s: None).search("x")
    assert len(calls) == 2
