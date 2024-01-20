"""
Workbench for testing
"""


import os
import ast
import pickle

from typing import Callable

import git
import yaml
import click
import chromadb

from chromadb import Collection

from groundcrew import system_prompts as sp, utils
from groundcrew.code import extract_python_from_file, init_db
from groundcrew.agent import Agent
from groundcrew.dataclasses import Config


@click.command()
@click.option('--config', '-c', default='config.yaml')
def main(config: str):
    """"""

    with open(config, 'r') as f:
        config = yaml.safe_load(f)
    config = Config(**config)


    print(config.repository)


    repo = git.Repo(config.repository)

    print(repo.head.commit.hexsha.__class__)


if __name__ == '__main__':
    main()
