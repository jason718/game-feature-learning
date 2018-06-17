
import yaml


def get_config(config):
    """Load yaml file"""
    with open(config, 'r') as stream:
        return yaml.load(stream)
