from typing import NewType, List, NamedTuple
import numpy as np

Observation = NewType("Observation", np.ndarray)
Action = NewType("Action", np.int32)
Reward = NewType("Reward", np.float32)

Transition = NamedTuple(
    "Transition",
    [ 
        ("observation", Observation),
        ("actions", Action),
        ("rewards", Reward),
        ("next_observation", Observation),
    ]
)

Trace = NamedTuple(
    "Trace",
    [
        ("observations", List[Observation]),
        ("actions", List[Action]),
        ("rewards", List[Reward]),
    ]
)

