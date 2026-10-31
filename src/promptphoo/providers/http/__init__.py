"""HTTP provider package. Ports src/providers/http.ts, httpTransforms.ts, httpMultipart.ts,
transformUtils.ts and urlEncoding.ts.

The registry resolves `http:` / `https:` ids to `promptphoo.providers.http:create_provider`.
"""

from promptphoo.providers.http.multipart import (
    decode_url_component,
    normalize_file_path,
    render_http_multipart_body,
    resolve_path,
)
from promptphoo.providers.http.provider import (
    HttpProvider,
    content_type_is_json,
    create_provider,
    determine_request_body,
    estimate_token_count,
    process_json_body,
    process_text_body,
)
from promptphoo.providers.http.raw import (
    escape_json_variables,
    extract_body_from_raw_request,
    parse_raw_request,
    render_raw_request_with_nunjucks,
    url_encode_raw_request_path,
)
from promptphoo.providers.http.signature import (
    diagnose_private_key_material,
    generate_signature,
    preprocess_signature_auth_config,
)
from promptphoo.providers.http.transforms import (
    create_session_parser,
    create_transform_request,
    create_transform_response,
    create_validate_status,
    load_transform_module,
    parse_file_transform_reference,
)

__all__ = [
    "HttpProvider",
    "content_type_is_json",
    "create_provider",
    "create_session_parser",
    "create_transform_request",
    "create_transform_response",
    "create_validate_status",
    "decode_url_component",
    "determine_request_body",
    "diagnose_private_key_material",
    "escape_json_variables",
    "estimate_token_count",
    "extract_body_from_raw_request",
    "generate_signature",
    "load_transform_module",
    "normalize_file_path",
    "parse_file_transform_reference",
    "parse_raw_request",
    "preprocess_signature_auth_config",
    "process_json_body",
    "process_text_body",
    "render_http_multipart_body",
    "render_raw_request_with_nunjucks",
    "resolve_path",
    "url_encode_raw_request_path",
]
