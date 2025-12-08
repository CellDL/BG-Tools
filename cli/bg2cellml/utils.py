from typing import Any


import structlog
from structlog.dev import BRIGHT, GREEN, RESET_ALL


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


def bright() -> str:
#===================
    return BRIGHT
    return ''

def pretty_log(s: Any) -> str:
#=============================
    return f'{RESET_ALL}{GREEN}{str(s)}{RESET_ALL}{BRIGHT}'
    return s
