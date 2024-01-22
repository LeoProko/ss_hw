import importlib
import json
import logging
import os
from datetime import datetime
from functools import reduce, partial
from operator import getitem
from pathlib import Path
import hydra
from omegaconf import DictConfig, OmegaConf

from src import text_encoder as text_encoder_module
from src.base.base_text_encoder import BaseTextEncoder
from src.logger import setup_logging
from src.text_encoder import CTCCharTextEncoder
from src.utils import read_json, write_json, ROOT_PATH


@hydra.main(config_path="src/configs", config_name="config.yaml")
class ConfigParser:
    def __init__(self, config, resume=None, modification=None, run_id=None):
        """
        class to parse configuration json file. Handles hyperparameters for training,
        initializations of modules, checkpoint saving and logging module.
        :param config: Dict containing configurations, hyperparameters for training.
                       contents of `config.json` file for example.
        :param resume: String, path to the checkpoint being loaded.
        :param modification: Dict {keychain: value}, specifying position values to be replaced
                             from config dict.
        :param run_id: Unique Identifier for training processes.
                       Used to save checkpoints and training log. Timestamp is being used as default
        """
        # load config file and apply modification
        self._config = _update_config(config, modification)
        self.resume = resume
        self._text_encoder = None

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self._config.trainer.save_dir)

        exper_name = self.config["name"]
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r"%m%d_%H%M%S")
        self._save_dir = str(save_dir / "models" / exper_name / run_id)
        self._log_dir = str(save_dir / "log" / exper_name / run_id)

        # make directory for saving checkpoints and log.
        exist_ok = run_id == ""
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)

        # # save updated config file to the checkpoint dir
        # write_json(self.config, self.save_dir / "config.json")

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}

    def save_config(self):
        OmegaConf.save(self._config, self.save_dir / "config.yaml")


    @staticmethod
    def init_obj(obj_dict, default_module, *args, device=None, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj(config['param'], module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        if "module" in obj_dict:
            default_module = importlib.import_module(obj_dict["module"])

        module_name = obj_dict["type"]
        module_args = dict(obj_dict["args"])
        if device is not None:
            module_args["device"] = device
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return getattr(default_module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
        module_name = self[name]["type"]
        module_args = dict(self[name]["args"])
        assert all(
            [k not in module_args for k in kwargs]
        ), "Overwriting kwargs given in config file is not allowed"
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = "verbosity option {} is invalid. Valid options are {}.".format(
            verbosity, self.log_levels.keys()
        )
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def get_text_encoder(self) -> BaseTextEncoder:
        if self._text_encoder is None:
            if "text_encoder" not in self._config:
                self._text_encoder = CTCCharTextEncoder()
            else:
                self._text_encoder = self.init_obj(
                    self["text_encoder"], default_module=text_encoder_module
                )
        return self._text_encoder

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return Path(self._save_dir)

    @property
    def log_dir(self):
        return Path(self._log_dir)

    @classmethod
    def get_default_configs(cls):
        config_path = ROOT_PATH / "src" / "config.json"
        with config_path.open() as f:
            return cls(json.load(f))

    @classmethod
    def get_test_configs(cls):
        config_path = ROOT_PATH / "src" / "tests" / "config.json"
        with config_path.open() as f:
            return cls(json.load(f))


# helper functions to update config dict with custom cli options
def _update_config(config, modification):
    if modification is None:
        return config

    for k, v in modification.items():
        OmegaConf.update(config, key, value)

    return config


def _get_opt_name(flags):
    for flg in flags:
        if flg.startswith("--"):
            return flg.replace("--", "")
    return flags[0].replace("--", "")


def _set_by_path(tree, keys, value):
    """Set a value in a nested object in tree by sequence of keys."""
    keys = keys.split(";")
    _get_by_path(tree, keys[:-1])[keys[-1]] = value


def _get_by_path(tree, keys):
    """Access a nested object in tree by sequence of keys."""
    return reduce(getitem, keys, tree)
