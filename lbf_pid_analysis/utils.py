#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Utility functions for LBF data analysis.
#
# Copyright (c) Honda Research Institute Europe GmbH
# This file is part of lbf_pid_analysis.
# lbf_pid_analysis is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# lbf_pid_analysis is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# You should have received a copy of the GNU General Public License
# along with lbf_pid_analysis. If not, see <http://www.gnu.org/licenses/>.
#
#
import sys
import logging
from pathlib import Path

import yaml


def read_settings(filename):
    """Read settings YAML files.

    Parameters
    ----------
    filename : str
        Filename of settings file

    Returns
    -------
    dict
        Settings dictionary
    """
    with open(filename, "r") as read_file:
        settings = yaml.safe_load(read_file)
    return settings


def initialize_logger(log_name):
    """Initialize logging.

    Configure logging format. Log to console and a log file.

    Parameters
    ----------
    log_name : str
        Name of logfile
    """
    log_format = "%(asctime)s - %(levelname)-4s  [%(filename)s:%(funcName)10s():l %(lineno)d] %(message)s"
    log_fmt = "%Y-%m-%d - %H:%M:%S"
    logging.basicConfig(
        stream=sys.stdout,
        format=log_format,
        datefmt=log_fmt,
        level=logging.INFO,
    )
    log_directory = Path(Path.cwd()).joinpath("../log")
    log_directory.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        Path(log_directory).joinpath(f"{log_name}.log"), mode="w"
    )
    # Log file handler name before adding the handler such that the path does
    # not appear in the log.
    logging.info("Logging to file %s", file_handler.baseFilename)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger("").addHandler(file_handler)  # add to root logger
