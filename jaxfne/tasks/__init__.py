"""Private task registry for the learning lane (not exported from ``jaxfne``).

Importing this package registers the built-in tasks. The core never imports a
task by name: a lane registers its file, and the contract test in
``tests/test_task_registry.py`` picks it up.
"""

from ._registry import TaskDescriptor, get_task, list_tasks, register_task
from .chain import CHAIN5, make_chain5
from .corridor import CORRIDOR7, make_corridor7

register_task(CHAIN5, make_chain5)
register_task(CORRIDOR7, make_corridor7)

__all__ = [
    "CHAIN5",
    "CORRIDOR7",
    "TaskDescriptor",
    "get_task",
    "list_tasks",
    "make_chain5",
    "make_corridor7",
    "register_task",
]
