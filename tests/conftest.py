import pytest

# Env vars that point promptphoo at a directory under the user's home (`~/.promptfoo` by
# default). A test that needs one of them sets it itself.
_STATE_DIR_ENV = ("PROMPTFOO_CACHE_PATH", "PROMPTFOO_LOG_DIR")


@pytest.fixture(autouse=True)
def _isolate_promptfoo_state(tmp_path_factory: pytest.TempPathFactory, monkeypatch):
    """Keep the suite away from the real `~/.promptfoo`: the config directory (database,
    cache, logs, blobs, promptfoo.yaml) is a fresh temp directory for every test."""
    from promptphoo import paths

    config_dir = tmp_path_factory.mktemp("promptfoo-config")
    monkeypatch.setenv("PROMPTFOO_CONFIG_DIR", str(config_dir))
    for name in _STATE_DIR_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(paths, "_override", None)
    yield config_dir


@pytest.fixture(autouse=True)
def _reset_blob_storage_provider():
    """The default blob provider caches its base path (`<configDir>/blobs`) on first use, as
    upstream's module singleton does. Tests switch PROMPTFOO_CONFIG_DIR, so drop it after each."""
    yield
    from promptphoo.blobs import reset_blob_storage_provider

    reset_blob_storage_provider()
