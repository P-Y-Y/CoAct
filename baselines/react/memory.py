from dataclasses import dataclass, field
from typing import List

@dataclass
class Step:
    thought: str
    action: str
    action_input: str
    observation: str

@dataclass
class ConversationMemory:
    steps: List[Step] = field(default_factory=list)
    final_answer: str | None = None

    def add_step(self, thought: str, action: str, action_input: str, observation: str):
        self.steps.append(Step(thought, action, action_input, observation))

    def set_final(self, answer: str):
        self.final_answer = answer
