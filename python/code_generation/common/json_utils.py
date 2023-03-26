import json
def read_json(path):
    with open(path, "r") as json_file:
        return json.loads(json_file.read())
