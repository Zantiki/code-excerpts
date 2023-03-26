import os
from pathlib import Path

definition_path = os.path.dirname(os.path.realpath(__file__))
DATA_ROOT = Path("{}/../data_csvs".format(definition_path))
CONTENT_ROOT = Path("{}/..".format(definition_path))
DOCUMENTATION_ROOT = Path("{}/../documentation".format(definition_path))