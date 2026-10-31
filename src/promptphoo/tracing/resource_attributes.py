"""Ports src/tracing/resourceAttributes.ts.

Resource-attribute keys that link SDK-emitted OTEL logs back to the evaluation trace. The
claude-agent-sdk provider sets them through `OTEL_RESOURCE_ATTRIBUTES` and the OTLP /v1/logs
receiver reads them; both sides must agree, so these constants are the single source.
"""

PROMPTFOO_RESOURCE_ATTR_TRACE_ID = "promptfoo.trace_id"
PROMPTFOO_RESOURCE_ATTR_PARENT_SPAN_ID = "promptfoo.parent_span_id"
