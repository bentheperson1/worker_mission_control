import json


def parse_message(raw):
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None

    if "type" not in data:
        return None

    return json.dumps(data)