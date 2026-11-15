"""Hand-sized parity games suitable for diagrams and exhaustive inspection."""

from .game import ParityGame


def tiny_games() -> tuple[ParityGame, ...]:
    """Return three max-parity examples with 1, 8, and 16 positional profiles."""
    return (
        ParityGame(
            owner={0: 0, 1: 1}, priority={0: 0, 1: 1},
            successors={0: (1,), 1: (0,)}, name="alternating_cycle_even",
        ),
        ParityGame(
            owner={0: 0, 1: 1, 2: 1}, priority={0: 0, 1: 2, 2: 1},
            successors={0: (1, 2), 1: (0, 1), 2: (0, 2)}, name="choice_and_self_loops",
        ),
        ParityGame(
            owner={0: 0, 1: 0, 2: 1, 3: 1}, priority={0: 1, 1: 2, 2: 0, 3: 3},
            successors={0: (1, 2), 1: (0, 3), 2: (1, 2), 3: (0, 3)}, name="four_vertex_interaction",
        ),
    )
