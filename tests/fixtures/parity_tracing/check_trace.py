def get_assert(output, context):
    trace = context.get("trace") or {}
    # Only the target's spans: promptfoo's own spans (test case, provider call, graders) vary.
    names = sorted(
        span["name"]
        for span in trace.get("spans", [])
        if span["name"].startswith(("agent.", "llm.", "tool."))
    )
    return {"pass": len(names) == 4, "score": 1, "reason": ",".join(names)}
