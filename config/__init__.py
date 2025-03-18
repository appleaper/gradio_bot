import yaml
import sys

if sys.platform == 'win32':
    config_path = r'C:\use\code\RapidOcr_small\config\dev_win.yaml'
else:
    config_path = '/home/pandas/snap/code/RapidOcr/config/dev.yaml'
with open(config_path, 'r', encoding='utf8') as file:
    conf_yaml = yaml.safe_load(file)