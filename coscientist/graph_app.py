from __future__ import annotations

import logging
from typing import List

from langgraph.graph import END, StateGraph

from .agents import (EvolutionAgent, GenerationAgent, MetaReviewAgent,
                     ProximityAgent, ReflectionAgent)
from .state import CoScientistState, Hypothesis, ResearchGoal
from .tournament import run_tournament

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def node_generate(state: CoScientistState) -> CoScientistState:
    logger.info("Starting generation phase")
    params = state["params"]
    population_size = int(params.get("population", 6))
    logger.info(f"Generating initial population of size {population_size}")

    gen = GenerationAgent(n=population_size)
    hyps = gen.run(state["goal"], generation=state["round_index"])
    state["population"] = hyps

    logger.info(f"Generated {len(hyps)} hypotheses in round {state['round_index']}")
    return state


def node_reflect(state: CoScientistState) -> CoScientistState:
    logger.info("Starting reflection phase")
    refl = ReflectionAgent(use_web=True)
    reviews = {}

    logger.info(f"Reflecting on {len(state['population'])} hypotheses")
    for h in state["population"]:
        logger.debug(f"Reviewing hypothesis {h.id}")
        reviews[h.id] = refl.run(state["goal"], h)
    state["reviews"] = reviews

    logger.info(f"Completed {len(reviews)} reviews")
    return state


def node_rank(state: CoScientistState) -> CoScientistState:
    logger.info("Starting ranking phase")
    seed = int(state["params"].get("seed", 0))
    logger.info(f"Running tournament with seed {seed}")

    ts = run_tournament(
        state["population"], state["goal"], rnd=state["round_index"], seed=seed
    )
    state["tournament"] = ts

    logger.info("Tournament completed")
    return state


def node_evolve(state: CoScientistState) -> CoScientistState:
    logger.info("Starting evolution phase")
    keep_top = int(state["params"].get("keep_top", 4))
    logger.info(f"Keeping top {keep_top} hypotheses")

    pop = sorted(state["population"], key=lambda h: h.score, reverse=True)
    winners = pop[:keep_top]
    logger.info(f"Selected {len(winners)} winners for evolution")

    evo = EvolutionAgent()
    new_gen: List[Hypothesis] = []
    for w in winners:
        new_variants = evo.run(w, state["tournament"].patterns)
        logger.debug(f"Generated {len(new_variants)} variants from hypothesis {w.id}")
        new_gen.extend(new_variants)

    state["population"] = winners + new_gen
    state["round_index"] += 1
    logger.info(f"Evolution complete. New population size: {len(state['population'])}")
    return state


def node_proximity(state: CoScientistState) -> CoScientistState:
    logger.info("Starting proximity analysis")
    prox = ProximityAgent()

    for h in state["population"]:
        old_score = h.score
        h.score = 0.5 * h.score + 5 * prox.score(state["goal"], h)
        logger.debug(f"Hypothesis {h.id}: score adjusted from {old_score} to {h.score}")

    logger.info("Proximity analysis complete")
    return state


def node_meta_review(state: CoScientistState) -> CoScientistState:
    logger.info("Starting meta review")
    shortlist_size = int(state["params"].get("shortlist", 5))
    shortlist = sorted(state["population"], key=lambda h: h.score, reverse=True)[
        :shortlist_size
    ]

    logger.info(f"Reviewing top {len(shortlist)} hypotheses")
    meta = MetaReviewAgent()
    state["overview"] = meta.run(state["goal"], shortlist)

    logger.info("Meta review complete")
    return state


def build_app(rounds: int = 2) -> StateGraph:
    logger.info(f"Building application graph with {rounds} rounds")
    graph = StateGraph(CoScientistState)

    # Add nodes
    for node in ["generate", "reflect", "rank", "evolve", "proximity", "meta_review"]:
        logger.debug(f"Adding node: {node}")
        graph.add_node(node, globals()[f"node_{node}"])

    graph.set_entry_point("generate")
    logger.debug("Setting up graph edges")
    graph.add_edge("generate", "reflect")
    graph.add_edge("reflect", "rank")
    graph.add_edge("rank", "proximity")

    def should_stop(state: CoScientistState):
        current_round = state["round_index"]
        max_rounds = int(state["params"].get("rounds", rounds))
        logger.debug(f"Checking stop condition: round {current_round}/{max_rounds}")
        return current_round >= max_rounds

    graph.add_conditional_edges(
        "proximity",
        should_stop,
        {
            True: "meta_review",
            False: "evolve",
        },
    )
    graph.add_edge("evolve", "reflect")
    graph.add_edge("meta_review", END)

    logger.info("Graph construction complete")
    return graph
