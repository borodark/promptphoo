import pytest


@pytest.fixture(autouse=True)
def _reset_blob_storage_provider():
    """The default blob provider caches its base path (`<configDir>/blobs`) on first use, as
    upstream's module singleton does. Tests switch PROMPTFOO_CONFIG_DIR, so drop it after each."""
    yield
    from promptphoo.blobs import reset_blob_storage_provider

    reset_blob_storage_provider()
