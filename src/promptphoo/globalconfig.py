"""The global config file `<configDir>/promptfoo.yaml`.

Ports src/globalConfig/globalConfig.ts, the email helpers of src/globalConfig/accounts.ts and the
api-key lookup of src/globalConfig/cloud.ts. The file is shared with Node promptfoo, so it is
read with the js-yaml rules and written in js-yaml's `dump` layout.
"""

from __future__ import annotations

import os
import uuid
from typing import Any

import yaml

from promptphoo.paths import get_config_directory_path
from promptphoo.yamlload import load_yaml


class _JsYamlDumper(yaml.SafeDumper):
    """js-yaml `dump`: block style, insertion order, sequences indented under their key."""

    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        super().increase_indent(flow, False)


def dump_js_yaml(value: Any) -> str:
    if value == {}:
        return "{}\n"
    return yaml.dump(
        value,
        Dumper=_JsYamlDumper,
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
        width=10**9,
    )


def _config_file_path(create: bool = False) -> str:
    return os.path.join(get_config_directory_path(create), "promptfoo.yaml")


def write_global_config(config: dict[str, Any]) -> None:
    with open(_config_file_path(True), "w", encoding="utf-8") as f:
        f.write(dump_js_yaml(config))


def read_global_config() -> dict[str, Any]:
    config_dir = get_config_directory_path()
    config_file_path = os.path.join(config_dir, "promptfoo.yaml")
    global_config: dict[str, Any] = {"id": str(uuid.uuid4())}
    if os.path.exists(config_file_path):
        with open(config_file_path, encoding="utf-8") as f:
            loaded = load_yaml(f.read())
        global_config = loaded if isinstance(loaded, dict) and loaded else {}
        if not global_config.get("id"):
            global_config = {**global_config, "id": str(uuid.uuid4())}
            write_global_config(global_config)
    else:
        os.makedirs(config_dir, exist_ok=True)
        with open(config_file_path, "w", encoding="utf-8") as f:
            f.write(dump_js_yaml(global_config))
    return global_config


def write_global_config_partial(partial_config: dict[str, Any]) -> None:
    """Merge top-level keys into the stored config; a None value removes the key."""
    updated = dict(read_global_config())
    for key, value in partial_config.items():
        if value is not None:
            updated[key] = value
        else:
            updated.pop(key, None)
    write_global_config(updated)


def get_user_email() -> str | None:
    account = read_global_config().get("account")
    email = account.get("email") if isinstance(account, dict) else None
    return email or None


def set_user_email(email: str) -> None:
    account = read_global_config().get("account")
    account = dict(account) if isinstance(account, dict) else {}
    account["email"] = email
    write_global_config_partial({"account": account})


def get_cloud_api_key() -> str | None:
    """cloudConfig.getApiKey(): the saved key, then PROMPTFOO_API_KEY."""
    cloud = read_global_config().get("cloud")
    saved = cloud.get("apiKey") if isinstance(cloud, dict) else None
    return saved or os.environ.get("PROMPTFOO_API_KEY") or None
