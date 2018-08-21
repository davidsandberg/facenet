# -*- coding:utf-8 -*-
import os

from six.moves import configparser


class OperationalError(Exception):
    """operation error."""


class Dictionary(dict):
    """ custom dict."""

    def __getattr__(self, key):
        return self.get(key, None)

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class Config:
    def __init__(self, cfg="config.conf"):
        """
        @param file_name: file name without extension.
        @param cfg: configuration file path.
        """
        config = configparser.ConfigParser()
        config.read(cfg)

        for section in config.sections():
            setattr(self, section, Dictionary())
            for name, raw_value in config.items(section):
                try:
                    # Ugly fix to avoid '0' and '1' to be parsed as a
                    # boolean value.
                    # We raise an exception to goto fail^w parse it
                    # as integer.
                    if config.get(section, name) in ["0", "1"]:
                        raise ValueError

                    value = config.getboolean(section, name)
                except ValueError:
                    try:
                        value = config.getint(section, name)
                    except ValueError:
                        value = config.get(section, name)

                setattr(getattr(self, section), name, value)

    def get(self, section):
        """Get option.
        @param section: section to fetch.
        @return: option value.
        """
        try:
            return getattr(self, section)
        except AttributeError as e:
            raise OperationalError("Option %s is not found in "
                                   "configuration, error: %s" %
                                   (section, e))
