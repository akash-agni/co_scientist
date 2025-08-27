from __future__ import annotations

import uuid
from typing import Dict, List, Optional, TypedDict

from pydantic import BaseModel, Field


class ResearchGoal(BaseModel):
    text: str
    constraints: Dict[str, str] = Field(default_factory=dict)
    preferences: Dict[str, str] = Field(default_factory=dict)


class Citation(BaseModel):
    title: str
    url: str
    snippet: str


class Hypothesis(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str
    rationale: str
    citations: List[Citation] = Field(default_factory=list)
    score: float = 0.0  # overall composite ranking score
    parent_id: Optional[str] = None
    generation: int = 0  # evolution round


class Review(BaseModel):
    hypothesis_id: str
    strengths: List[str]
    weaknesses: List[str]
    risks: List[str]
    proposed_tests: List[str]
    updated_rationale: Optional[str] = None
    added_citations: List[Citation] = Field(default_factory=list)


class MatchResult(BaseModel):
    a_id: str
    b_id: str
    winner_id: str
    loser_id: str
    reasoning: str


class TournamentSummary(BaseModel):
    round_index: int
    results: List[MatchResult]
    patterns: List[str]  # e.g., recurring strengths/weaknesses across winners


class CoScientistState(TypedDict):
    goal: ResearchGoal
    round_index: int
    population: List[Hypothesis]
    reviews: Dict[str, Review]
    tournament: Optional[TournamentSummary]
    overview: Optional[str]
    params: Dict[str, int | float | str]
