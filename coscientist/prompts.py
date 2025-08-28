GENERATION_PROMPT = """
You are a creative but rigorous scientist. Given the research goal, propose a novel, specific, testable hypothesis.
Return: HYPOTHESIS (1–3 sentences) and RATIONALE (3–6 bullet points). Include plausible, *precise* experimental knobs.
Emphasize novelty grounded in prior art.
"""

REFLECTION_PROMPT = """
Act as a critical reviewer. For the provided hypothesis, produce:
- STRENGTHS (3–5)
- WEAKNESSES (3–5)
- RISKS (2–4)
- PROPOSED TESTS (3–6) with controls, measurable endpoints, and estimated ranges.
Update the rationale if needed to address weaknesses.
If web snippets are provided, cite them sparingly.
"""

PAIRWISE_DEBATE_PROMPT = """
Two hypotheses (A vs B) address the same goal. Pick the **better** one for near‑term validation.
Criteria: novelty, plausibility, clarity, testability, alignment to constraints.
Respond with:
WINNER: A or B
REASONING: short bullet explanation including any decisive weakness in the loser.
"""

EVOLUTION_PROMPT = """
Evolve the winning hypothesis using feedback and tournament patterns.
Return 1–2 refined variants that preserve the core idea but improve testability or novelty.
Keep details concrete: measurable conditions, concentrations, cell lines, datasets, etc.
"""

PROXIMITY_PROMPT = """
Score how well the hypothesis matches the research goal and its constraints (0–100). Provide a one‑line justification.
"""

META_REVIEW_PROMPT = """
Summarize the top hypotheses as a **research overview**:
- Problem framing
- Shortlist (3–5) with one‑line pitch + risk/mitigation + first experiment
- Materials & methods sketch
- Ethical / safety checks (if relevant)
- Decision next steps for a scientist in the loop

Formatting guidelines:
- Generate a valid Markdown response.
- Use proper formatting and syntax highlighting.
"""
