from __future__ import annotations

import logging
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Rest of the imports remain the same...

# Create logger for this module
logger = logging.getLogger(__name__)

LLM_MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
CRITIC_MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
DEBATE_MODEL = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)


class GenerationAgent:
    def __init__(self, n: int = 4):
        self.n = n
        self.logger = logging.getLogger(f"{__name__}.GenerationAgent")
        self.logger.info(f"Initialized GenerationAgent with n={n}")

    def run(self, goal: ResearchGoal, generation: int) -> List[Hypothesis]:
        self.logger.info(
            f"Starting hypothesis generation for goal: {goal.text[:100]}..."
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", GENERATION_PROMPT),
                (
                    "human",
                    "Research goal: {goal}\nConstraints: {constraints}\nPreferences: {preferences}",
                ),
            ]
        )
        chain = prompt | LLM_MODEL
        hyps: List[Hypothesis] = []
        for i in range(self.n):
            self.logger.debug(f"Generating hypothesis {i+1}/{self.n}")
            msg = chain.invoke(
                {
                    "goal": goal.text,
                    "constraints": goal.constraints,
                    "preferences": goal.preferences,
                }
            )
            text = msg.content
            hyp_text = text.split("RATIONALE:")[0].replace("HYPOTHESIS:", "").strip()
            rationale = text.split("RATIONALE:")[-1].strip()
            hyp = Hypothesis(text=hyp_text, rationale=rationale, generation=generation)
            self.logger.debug(f"Generated hypothesis: {hyp_text[:100]}...")
            hyps.append(hyp)
        self.logger.info(f"Generated {len(hyps)} hypotheses")
        return hyps


class ReflectionAgent:
    def __init__(self, use_web: bool = True):
        self.use_web = use_web
        self.search = OpenAIWebSearch(k=5) if use_web else None
        self.logger = logging.getLogger(f"{__name__}.ReflectionAgent")
        self.logger.info(f"Initialized ReflectionAgent with use_web={use_web}")

    def run(self, goal: ResearchGoal, hyp: Hypothesis) -> Review:
        self.logger.info(f"Starting reflection for hypothesis: {hyp.text[:100]}...")
        snippets = []
        if self.use_web:
            q = f"{goal.text} hypothesis context: {hyp.text[:128]}"
            self.logger.debug(f"Performing web search with query: {q[:100]}...")
            results = self.search.search(q)[:3]
            self.logger.info(f"Found {len(results)} web search results")
            for r in results:
                snippets.append(
                    (r.get("title", ""), r.get("url", ""), r.get("content", "")[:400])
                )

        context = (
            "\n\n".join([f"- {t}\n{u}\n{c}" for (t, u, c) in snippets])
            if snippets
            else "(no web snippets)"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", REFLECTION_PROMPT),
                (
                    "human",
                    "Research goal: {goal}\nHypothesis: {hyp}\nRationale: {rat}\nWeb snippets:\n{snips}",
                ),
            ]
        )

        self.logger.debug("Generating review...")
        msg = (prompt | CRITIC_MODEL).invoke(
            {
                "goal": goal.text,
                "hyp": hyp.text,
                "rat": hyp.rationale,
                "snips": context,
            }
        )

        text = msg.content

        def _lines(tag: str) -> List[str]:
            import re

            m = re.search(rf"{tag}:(.*?)(?:\n\n|$)", text, re.S | re.I)
            if not m:
                self.logger.warning(f"No {tag} section found in review")
                return []
            items = [
                ln.strip("- • ") for ln in m.group(1).strip().split("\n") if ln.strip()
            ]
            return items

        rev = Review(
            hypothesis_id=hyp.id,
            strengths=_lines("STRENGTHS"),
            weaknesses=_lines("WEAKNESSES"),
            risks=_lines("RISKS"),
            proposed_tests=_lines("PROPOSED TESTS"),
            updated_rationale=None,
            added_citations=[
                Citation(title=t, url=u, snippet=s) for (t, u, s) in snippets
            ],
        )
        self.logger.info(
            f"Generated review with {len(rev.strengths)} strengths, {len(rev.weaknesses)} weaknesses"
        )
        return rev


class RankingAgent:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.RankingAgent")
        self.logger.info("Initialized RankingAgent")

    def compare(self, a: Hypothesis, b: Hypothesis, goal: ResearchGoal) -> Dict:
        self.logger.info(f"Comparing hypotheses: {a.text[:50]}... vs {b.text[:50]}...")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PAIRWISE_DEBATE_PROMPT),
                ("human", "Goal: {goal}\nA: {a}\nB: {b}"),
            ]
        )
        out = (
            (prompt | DEBATE_MODEL)
            .invoke(
                {
                    "goal": goal.text,
                    "a": f"{a.text}\nRATIONALE: {a.rationale}",
                    "b": f"{b.text}\nRATIONALE: {b.rationale}",
                }
            )
            .content
        )
        winner = "A" if "WINNER: A" in out else ("B" if "WINNER: B" in out else "A")
        reasoning = out.split("REASONING:")[-1].strip()
        self.logger.info(f"Comparison complete. Winner: {winner}")
        return {"winner": winner, "reasoning": reasoning}


class EvolutionAgent:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.EvolutionAgent")
        self.logger.info("Initialized EvolutionAgent")

    def run(self, base: Hypothesis, summary_patterns: List[str]) -> List[Hypothesis]:
        self.logger.info(f"Evolving hypothesis: {base.text[:100]}...")
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", EVOLUTION_PROMPT),
                (
                    "human",
                    "Base hypothesis: {h}\nRationale: {r}\nTournament patterns: {pats}",
                ),
            ]
        )
        msg = (prompt | LLM_MODEL).invoke(
            {"h": base.text, "r": base.rationale, "pats": "; ".join(summary_patterns)}
        )
        variants_text = [s.strip("- • ") for s in msg.content.split("\n") if s.strip()]
        hyps = []
        for i, vt in enumerate(variants_text[:2]):
            self.logger.debug(f"Creating variant {i+1}: {vt[:100]}...")
            hyps.append(
                Hypothesis(
                    text=vt,
                    rationale=base.rationale,
                    parent_id=base.id,
                    generation=base.generation + 1,
                )
            )
        self.logger.info(f"Generated {len(hyps)} variants")
        return hyps


class ProximityAgent:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.ProximityAgent")
        self.logger.info("Initialized ProximityAgent")

    def score(self, goal: ResearchGoal, hyp: Hypothesis) -> float:
        self.logger.info(f"Scoring hypothesis proximity: {hyp.text[:100]}...")
        prompt = ChatPromptTemplate.from_messages(
            [("system", PROXIMITY_PROMPT), ("human", "Goal: {g}\nHypothesis: {h}")]
        )
        out = (prompt | CRITIC_MODEL).invoke({"g": goal.text, "h": hyp.text}).content
        import re

        m = re.search(r"(\d{1,3})", out)
        score = max(0, min(100, int(m.group(1)))) if m else 50
        self.logger.info(f"Proximity score: {score}")
        return score


class MetaReviewAgent:
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.MetaReviewAgent")
        self.logger.info("Initialized MetaReviewAgent")

    def run(self, goal: ResearchGoal, shortlist: List[Hypothesis]) -> str:
        self.logger.info(f"Generating meta-review for {len(shortlist)} hypotheses")
        prompt = ChatPromptTemplate.from_messages(
            [("system", META_REVIEW_PROMPT), ("human", "Goal: {g}\nShortlist:\n{sl}")]
        )
        sl = "\n".join([f"- {h.text} (gen {h.generation})" for h in shortlist])
        result = (prompt | CRITIC_MODEL).invoke({"g": goal.text, "sl": sl}).content
        self.logger.info("Meta-review generation complete")
        return result
