#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Custom package settings
#
#  Copyright (C)
#  Honda Research Institute Europe GmbH
#  Carl-Legien-Str. 30
#  63073 Offenbach/Main
#  Germany
#
#  UNPUBLISHED PROPRIETARY MATERIAL.
#  ALL RIGHTS RESERVED.
#
#

pylintConf = "pyproject.toml"

sqLevel = "advanced"

sqOptInRules = ["GEN12", "GEN14", "PY01", "PY03", "DOC04"]

sqOptOutRules = [
    "DOC03",  # there is a shell script running the analysis, this is documented in the README, I could not get this to run when moving the shell script to an 'examples' folder
]

copyright = [
    "Copyright (c) Honda Research Institute Europe GmbH",
    "This file is part of lbf_pid_analysis.",
    "lbf_pid_analysis is free software: you can redistribute it and/or modify",
    "it under the terms of the GNU General Public License as published by",
    "the Free Software Foundation, either version 3 of the License, or",
    "(at your option) any later version.",
    "lbf_pid_analysis is distributed in the hope that it will be useful,",
    "but WITHOUT ANY WARRANTY; without even the implied warranty of",
    "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the",
    "GNU General Public License for more details.",
    "You should have received a copy of the GNU General Public License",
    "along with lbf_pid_analysis. If not, see <http://www.gnu.org/licenses/>.",
]


# EOF
