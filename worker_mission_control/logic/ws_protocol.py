import json


def parse_message(raw):
	if isinstance(raw, bytes):
		raw = raw.decode("utf-8", errors="replace")
	
	try:
		data = json.loads(raw)
	except json.JSONDecodeError:
		return None
	
	if not isinstance(data, dict):
		return None
	
	return json.dumps(data)