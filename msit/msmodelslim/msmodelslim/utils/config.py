# Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.

import os
from configparser import ConfigParser

from pydantic import BaseModel

from msmodelslim.utils.exception import ConfigError, EnvVarError
from msmodelslim.utils.security.path import get_valid_read_path

CONFIG_PATH = '../config/config.ini'

SECTION_URL = 'URL'
KEY_REPO_URL = 'repository_url'
KEY_QUESTION_AND_ANSWER_URL = 'question_and_answer_url'

ENV_VAR_LOG_LEVEL = 'MSMODELSLIM_LOG_LEVEL'

VALID_LOG_LEVELS = ['info', 'debug']


class URLs(BaseModel):
    repository: str
    question_and_answer: str


class EnvVars(BaseModel):
    log_level: str


class ModelSlimConfig(BaseModel):
    urls: URLs
    env_vars: EnvVars


def init_config():
    file_config = ConfigParser()

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.abspath(os.path.join(cur_dir, CONFIG_PATH))
    config_path = get_valid_read_path(config_path)
    file_config.read(config_path)

    if not file_config.has_section(SECTION_URL):
        raise ConfigError(f"Invalid config section: {SECTION_URL}",
                          action=f'Please check the config file {config_path}')

    if not file_config.has_option(SECTION_URL, KEY_REPO_URL):
        raise ConfigError(f"Invalid config key: {KEY_REPO_URL}",
                          action=f'Please check the config file {config_path}')

    if not file_config.has_option(SECTION_URL, KEY_QUESTION_AND_ANSWER_URL):
        raise ConfigError(f"Invalid config key: {KEY_QUESTION_AND_ANSWER_URL}",
                          action=f'Please check the config file {config_path}')

    logger_level = os.getenv(ENV_VAR_LOG_LEVEL, 'INFO').lower()
    if logger_level not in VALID_LOG_LEVELS:
        raise EnvVarError(f"Invalid log level: {logger_level}, must be in {VALID_LOG_LEVELS}",
                          action=f'Please check the environment variable {ENV_VAR_LOG_LEVEL}')

    modelslim_config = ModelSlimConfig(
        urls=URLs(
            repository=file_config.get(SECTION_URL, KEY_REPO_URL),
            question_and_answer=file_config.get(SECTION_URL, KEY_QUESTION_AND_ANSWER_URL)
        ),
        env_vars=EnvVars(log_level=logger_level),
    )
    return modelslim_config


msmodelslim_config = init_config()
