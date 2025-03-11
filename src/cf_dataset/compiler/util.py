from dataclasses import dataclass


@dataclass
class CompilerOutput:
    """
    Dataclasses used to store compilation code and output
    """
    output: str
    code: int