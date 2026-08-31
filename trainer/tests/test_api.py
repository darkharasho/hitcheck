import pytest

from hitcheck_trainer.catalog.api import CatalogApi, CatalogApiError


def make_page(n=2, page=1, total=20479):
    return {
        "data": [{"id": f"set-{i}", "name": f"Card {i}"} for i in range(n)],
        "page": page,
        "pageSize": n,
        "count": n,
        "totalCount": total,
    }


class FakeTransport:
    """Replays a scripted sequence of (status, body) responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers):
        self.calls.append((url, headers))
        if not self.responses:
            raise AssertionError("transport called more times than scripted")
        return self.responses.pop(0)


def test_returns_cards_on_first_success():
    api = CatalogApi(FakeTransport([(200, make_page())]), sleep=lambda _: None)
    page = api.fetch_page(1)
    assert len(page.cards) == 2
    assert page.total_count == 20479


def test_retries_through_500s_then_succeeds():
    transport = FakeTransport([(500, None), (500, None), (200, make_page())])
    api = CatalogApi(transport, sleep=lambda _: None)
    assert len(api.fetch_page(1).cards) == 2
    assert len(transport.calls) == 3


def test_retries_on_connection_failure_signalled_as_status_zero():
    transport = FakeTransport([(0, None), (200, make_page())])
    api = CatalogApi(transport, sleep=lambda _: None)
    assert len(api.fetch_page(1).cards) == 2


def test_raises_after_exhausting_attempts():
    transport = FakeTransport([(500, None)] * 3)
    api = CatalogApi(transport, max_attempts=3, sleep=lambda _: None)
    with pytest.raises(CatalogApiError):
        api.fetch_page(1)


def test_sleeps_between_attempts_with_growing_delays():
    slept = []
    transport = FakeTransport([(500, None), (500, None), (200, make_page())])
    api = CatalogApi(transport, sleep=slept.append)
    api.fetch_page(1)
    assert slept == [0.5, 1.0]


def test_does_not_retry_a_404():
    transport = FakeTransport([(404, None)])
    api = CatalogApi(transport, sleep=lambda _: None)
    with pytest.raises(CatalogApiError):
        api.fetch_page(1)
    assert len(transport.calls) == 1


def test_sends_api_key_header_when_provided():
    transport = FakeTransport([(200, make_page())])
    CatalogApi(transport, api_key="abc123", sleep=lambda _: None).fetch_page(1)
    assert transport.calls[0][1]["X-Api-Key"] == "abc123"


def test_omits_api_key_header_when_absent():
    transport = FakeTransport([(200, make_page())])
    CatalogApi(transport, sleep=lambda _: None).fetch_page(1)
    assert "X-Api-Key" not in transport.calls[0][1]


def test_requests_the_expected_page_and_size():
    transport = FakeTransport([(200, make_page())])
    CatalogApi(transport, sleep=lambda _: None).fetch_page(7, page_size=250)
    url = transport.calls[0][0]
    assert "page=7" in url and "pageSize=250" in url


def test_retries_a_200_with_no_body_then_succeeds():
    transport = FakeTransport([(200, None), (200, make_page())])
    api = CatalogApi(transport, sleep=lambda _: None)
    page = api.fetch_page(1)
    assert len(page.cards) == 2
    assert len(transport.calls) == 2


def test_raises_after_exhausting_attempts_on_repeated_200_with_no_body():
    transport = FakeTransport([(200, None)] * 3)
    api = CatalogApi(transport, max_attempts=3, sleep=lambda _: None)
    with pytest.raises(CatalogApiError):
        api.fetch_page(1)
    assert len(transport.calls) == 3


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_retries_each_retryable_server_error_then_succeeds(status):
    transport = FakeTransport([(status, None), (200, make_page())])
    api = CatalogApi(transport, sleep=lambda _: None)
    page = api.fetch_page(1)
    assert len(page.cards) == 2
    assert len(transport.calls) == 2
