from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple

from .agents import RankingAgent
from .state import Hypothesis, MatchResult, ResearchGoal, TournamentSummary


@dataclass
class EloRanker:
    k: float = 24.0
    ratings: dict = field(default_factory=dict)

    def rating(self, hyp_id: str) -> float:
        return self.ratings.get(hyp_id, 1500.0)

    def update(self, winner: str, loser: str):
        import math

        ra, rb = self.rating(winner), self.rating(loser)
        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        eb = 1 - ea
        self.ratings[winner] = ra + self.k * (1 - ea)
        self.ratings[loser] = rb + self.k * (0 - eb)


def run_tournament(
    hypotheses: List[Hypothesis], goal: ResearchGoal, rnd: int, seed: int = 0
) -> TournamentSummary:
    rng = random.Random(seed + rnd)
    ranking = RankingAgent()
    elo = EloRanker()
    pairs: List[Tuple[Hypothesis, Hypothesis]] = []
    shuffled = hypotheses[:]
    rng.shuffle(shuffled)
    # if odd, the last one gets a bye
    for i in range(0, len(shuffled) - 1, 2):
        pairs.append((shuffled[i], shuffled[i + 1]))
    results: List[MatchResult] = []
    patterns = []
    for a, b in pairs:
        out = ranking.compare(a, b, goal)
        winner = a if out["winner"] == "A" else b
        loser = b if winner is a else a
        elo.update(winner.id, loser.id)
        results.append(
            MatchResult(
                a_id=a.id,
                b_id=b.id,
                winner_id=winner.id,
                loser_id=loser.id,
                reasoning=out["reasoning"],
            )
        )
        patterns.append(out["reasoning"])  # naive; could summarize later
    # project ELO back to hypotheses for downstream selection
    for h in hypotheses:
        h.score = elo.rating(h.id)
    return TournamentSummary(round_index=rnd, results=results, patterns=patterns)
