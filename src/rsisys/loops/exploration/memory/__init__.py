"""Private history and memory for the exploration loop."""

from rsisys.loops._memory import MemoryBuffer


class ExplorationMemory(MemoryBuffer):
    owner = 'actor'
    history_kinds = ('trajectory',)
