#===============================================================================
#
#  CellDL and bondgraph tools
#
#  Copyright (c) 2020 - 2025 David Brooks
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#===============================================================================

from typing import Any

import structlog
from structlog.dev import BRIGHT, GREEN, RESET_ALL

#===============================================================================

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="ISO"),
        structlog.processors.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.dev.ConsoleRenderer(colors=True)
    ]
)

log = structlog.get_logger()

#===============================================================================

def bright() -> str:
#===================
    return BRIGHT

def pretty_log(s: Any) -> str:
#=============================
    return f'{RESET_ALL}{GREEN}{str(s)}{RESET_ALL}{BRIGHT}'

#===============================================================================
#===============================================================================
