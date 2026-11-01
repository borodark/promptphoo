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
    # The database connection is a module global opened once against the config directory of
    # the moment. A connection left open by an earlier test on the same worker would point
    # this test's writes at that test's database, so close it on both sides.
    from promptphoo import db

    db.close_db()
    yield config_dir
    db.close_db()


@pytest.fixture(autouse=True)
def _reset_blob_storage_provider():
    """The default blob provider caches its base path (`<configDir>/blobs`) on first use, as
    upstream's module singleton does. Tests switch PROMPTFOO_CONFIG_DIR, so drop it after each."""
    yield
    from promptphoo.blobs import reset_blob_storage_provider

    reset_blob_storage_provider()


@pytest.fixture(autouse=True)
def _reset_safe_mode():
    """`eval --safe-mode` sets safe mode for the process, as upstream's cliState does. Reset it so
    one test's safe mode cannot leak into the next."""
    from promptphoo import cli_state

    cli_state.set_safe_mode(False)
    yield
    cli_state.set_safe_mode(False)
