def call_api(prompt, options, context):
    return {"output": prompt.upper(), "tokenUsage": {"total": 5, "prompt": 2, "completion": 3}}
