"""Assertion type names. Ports BaseAssertionTypesSchema / SpecialAssertionTypesSchema."""

BASE_ASSERTION_TYPES: tuple[str, ...] = (
    "agent-rubric", "answer-relevance", "bleu", "classifier", "contains", "contains-all",
    "contains-any", "contains-html", "contains-json", "contains-sql", "contains-xml",
    "context-faithfulness", "context-recall", "context-relevance", "conversation-relevance",
    "cost", "equals", "factuality", "finish-reason", "g-eval", "gleu", "guardrails", "icontains",
    "icontains-all", "icontains-any", "is-html", "is-json", "is-refusal", "is-sql",
    "is-valid-function-call", "is-valid-openai-function-call", "is-valid-openai-tools-call",
    "is-xml", "javascript", "latency", "levenshtein", "llm-rubric", "pi", "meteor",
    "model-graded-closedqa", "model-graded-factuality", "moderation", "perplexity",
    "perplexity-score", "python", "regex", "rouge-l", "rouge-n", "rouge-s", "ruby", "similar",
    "similar:cosine", "similar:dot", "similar:euclidean", "starts-with", "tool-call-f1",
    "skill-used", "trajectory:goal-success", "trajectory:tool-args-match",
    "trajectory:step-count", "trajectory:tool-sequence", "trajectory:tool-used",
    "trace-error-spans", "trace-span-count", "trace-span-duration", "search-rubric", "webhook",
    "word-count",
)  # fmt: skip

SPECIAL_ASSERTION_TYPES: tuple[str, ...] = ("select-best", "human", "max-score")
