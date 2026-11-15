"""Finite, total parity games and positional strategy enumeration.

Priority convention: max-parity. The player owning the greatest priority seen
infinitely often wins: even priorities for player 0, odd priorities for player 1.
Vertex labels may be any hashable Python values, including strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Hashable, Iterator, Mapping, TypeAlias

Vertex: TypeAlias = Hashable


def _ordered(values):
    """Sort consistently for ordinary labels, including mixed label types."""
    try:
        return tuple(sorted(values))
    except TypeError:
        return tuple(sorted(values, key=lambda value: str(value)))


def _format_vertices(values) -> str:
    return ", ".join(map(str, _ordered(values)))


@dataclass(frozen=True)
class ParityGame:
    """A finite max-parity game whose vertices can have string or integer labels."""

    owner: Mapping[Vertex, int]
    priority: Mapping[Vertex, int]
    successors: Mapping[Vertex, tuple[Vertex, ...]]
    name: str = "unnamed"

    def __post_init__(self) -> None:
        vertices = set(self.owner)
        if not vertices or set(self.priority) != vertices or set(self.successors) != vertices:
            raise ValueError("owner, priority, and successors must define the same nonempty vertex set")
        for v in vertices:
            if self.owner[v] not in (0, 1):
                raise ValueError(f"owner of {v!r} must be 0 or 1")
            if self.priority[v] < 0:
                raise ValueError(f"priority of {v!r} must be nonnegative")
            if not self.successors[v]:
                raise ValueError(f"vertex {v!r} has no successor; games must be total")
            unknown = set(self.successors[v]) - vertices
            if unknown:
                raise ValueError(f"vertex {v!r} points to unknown vertices {_format_vertices(unknown)}")

    @property
    def vertices(self) -> tuple[Vertex, ...]:
        return _ordered(self.owner)

    def vertices_owned_by(self, player: int) -> tuple[Vertex, ...]:
        return tuple(v for v in self.vertices if self.owner[v] == player)


@dataclass(frozen=True)
class StrategyProfile:
    """A positional choice for every vertex owned by each player."""

    player0: Mapping[Vertex, Vertex]
    player1: Mapping[Vertex, Vertex]

    def for_player(self, player: int) -> Mapping[Vertex, Vertex]:
        if player not in (0, 1):
            raise ValueError("player must be 0 or 1")
        return self.player0 if player == 0 else self.player1


def _choices(game: ParityGame, player: int) -> Iterator[dict[Vertex, Vertex]]:
    vertices = game.vertices_owned_by(player)
    options = [game.successors[v] for v in vertices]
    for destinations in product(*options):
        yield dict(zip(vertices, destinations))


def strategy_profile_count(game: ParityGame) -> int:
    """Return the number of complete positional profiles (may be enormous)."""
    count = 1
    for v in game.vertices:
        count *= len(game.successors[v])
    return count


def enumerate_strategy_profiles(
    game: ParityGame, *, max_profiles: int | None = 10_000
) -> Iterator[StrategyProfile]:
    """Yield every positional profile; refuse accidental huge enumerations by default."""
    count = strategy_profile_count(game)
    if max_profiles is not None and count > max_profiles:
        raise ValueError(f"{count:,} profiles exceed max_profiles={max_profiles:,}; pass a higher limit explicitly")
    strategies0 = tuple(_choices(game, 0))
    strategies1 = tuple(_choices(game, 1))
    for s0, s1 in product(strategies0, strategies1):
        yield StrategyProfile(s0, s1)


def play(game: ParityGame, profile: StrategyProfile, start: Vertex) -> dict[str, object]:
    """Follow a profile from ``start`` and return prefix, cycle, and winner."""
    if start not in game.owner:
        raise KeyError(f"unknown start vertex {start!r}")
    for player in (0, 1):
        expected = set(game.vertices_owned_by(player))
        supplied = set(profile.for_player(player))
        if expected != supplied:
            raise ValueError(f"player {player} strategy must define exactly vertices {_format_vertices(expected)}")
        for v, target in profile.for_player(player).items():
            if target not in game.successors[v]:
                raise ValueError(f"strategy sends {v!r} to non-successor {target!r}")

    first_seen: dict[Vertex, int] = {}
    path: list[Vertex] = []
    vertex = start
    while vertex not in first_seen:
        first_seen[vertex] = len(path)
        path.append(vertex)
        player = game.owner[vertex]
        vertex = profile.for_player(player)[vertex]
    cycle_start = first_seen[vertex]
    prefix = tuple(path[:cycle_start])
    cycle = tuple(path[cycle_start:])
    greatest_priority = max(game.priority[v] for v in cycle)
    return {
        "start": start,
        "prefix": prefix,
        "cycle": cycle,
        "cycle_priorities": tuple(game.priority[v] for v in cycle),
        "greatest_recurring_priority": greatest_priority,
        "winner": greatest_priority % 2,
    }


def read_pg(
    path: str | Path,
    *,
    name: str | None = None,
    vertex_labels: Mapping[int, Vertex] | None = None,
) -> ParityGame:
    """Read PGSolver ``.pg`` format, optionally mapping numeric IDs to labels.

    PGSolver files use integer IDs. ``vertex_labels={0: 'a', ...}`` provides
    human-readable labels in the resulting game while leaving the file standard.
    """
    source = Path(path)
    priorities: dict[int, int] = {}
    owners: dict[int, int] = {}
    edges: dict[int, tuple[int, ...]] = {}
    for line_no, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("parity "):
            continue
        if not line.endswith(";"):
            raise ValueError(f"{source}:{line_no}: expected semicolon at end of vertex row")
        fields = line[:-1].split()
        if len(fields) < 4:
            raise ValueError(f"{source}:{line_no}: malformed vertex row")
        v, prio, owner = map(int, fields[:3])
        if v in owners:
            raise ValueError(f"{source}:{line_no}: duplicate vertex {v}")
        priorities[v], owners[v] = prio, owner
        edges[v] = tuple(int(token) for token in fields[3].split(",") if token)

    if vertex_labels is not None:
        missing = set(owners) - set(vertex_labels)
        if missing:
            raise ValueError(f"vertex_labels is missing IDs: {_format_vertices(missing)}")
        labels = {v: vertex_labels[v] for v in owners}
        if len(set(labels.values())) != len(labels):
            raise ValueError("vertex_labels must map each vertex ID to a unique label")
    else:
        labels = {v: v for v in owners}

    mapped_owners = {labels[v]: owner for v, owner in owners.items()}
    mapped_priorities = {labels[v]: prio for v, prio in priorities.items()}
    mapped_edges = {labels[v]: tuple(labels[target] for target in targets) for v, targets in edges.items()}
    return ParityGame(mapped_owners, mapped_priorities, mapped_edges, name or source.stem)


def write_pg(game: ParityGame, path: str | Path) -> None:
    """Write standard PGSolver format, assigning numeric IDs to game labels."""
    target = Path(path)
    ids = {v: index for index, v in enumerate(game.vertices)}
    rows = [f"# Vertex labels: {', '.join(f'{ids[v]}={v}' for v in game.vertices)}", f"parity {len(game.vertices)};"]
    for v in game.vertices:
        successors = ",".join(str(ids[target]) for target in game.successors[v])
        rows.append(f"{ids[v]} {game.priority[v]} {game.owner[v]} {successors};")
    target.write_text("\n".join(rows) + "\n", encoding="utf-8")
