import json
from src.common.definitions import CONFIG


def read_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)

def get_key_from_json_file(file_path, key):
    return read_json(file_path)[key]
