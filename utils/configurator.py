import os
import yaml
import pprint
from argparse import ArgumentParser
from datetime import datetime


def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)


class Configurator:
    def __init__(self, args: ArgumentParser, config_file: str) -> None:
        assert os.path.exists(config_file)
        assert isinstance(args, ArgumentParser)
        self.config = load_config(config_file)
        for key, value in vars(args.parse_args()).items():
            if value is not None:
                self.config[key] = value
            if key not in self.config.keys():
                self.config[key] = value
        self.config['exp_name'] = (self.config['env_name'].replace('/', '-') +
                                   f"_{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}")

    def __getitem__(self, item):
        return self.config[item]

    def __setitem__(self, key, value):
        self.config[key] = value

    def save(self, config_save_path: str) -> None:
        fp = open(config_save_path, 'w+')
        fp.write(yaml.dump(self.config))
        fp.close()

    def print(self):
        pprint.PrettyPrinter(indent=2).pprint(self.config)

    def __str__(self):
        return yaml.dump(self.config)

