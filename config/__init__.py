import os
import yaml
import sys

config_dir = os.path.dirname(__file__)
if sys.platform == 'win32':
    config_path = os.path.join(config_dir, 'dev_win_small.yaml')
else:
    config_path = os.path.join(config_dir, 'dev.yaml')
with open(config_path, 'r', encoding='utf8') as file:
    conf_yaml = yaml.safe_load(file)