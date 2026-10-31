"""Private history and memory for the decision-making loop."""

from rsisys.loops._memory import MemoryBuffer


class DecisionMemory(MemoryBuffer):
    owner = 'decision_maker'
    history_kinds = ('decision', 'operation', 'feedback')
