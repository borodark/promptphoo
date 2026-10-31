import json


def call_api(prompt, options, context):
    content = json.loads(prompt)[0]["content"]
    for block in content.split('<Text index="')[1:]:
        index, _, body = block.partition('">')
        if "Paris" in body.split("</Text>")[0]:
            return {"output": index, "tokenUsage": {"total": 5, "prompt": 4, "completion": 1}}
    return {"output": "none"}
