# AI Co‑Scientist (LangGraph)

This repo mirrors the paper's design with six agents — Generation, Reflection, Ranking, Evolution, Proximity, Meta‑review — wired into a LangGraph that runs multi‑round tournament evolution.

## Quick start

1. **Install**
   ```bash
   uv venv && uv pip install -e .
   # or: python -m venv .venv && source .venv/bin/activate && pip install -e .
   ```
2. **Set keys**
   ```bash
   export OPENAI_API_KEY=...      # or set another LangChain-compatible LLM
   export TAVILY_API_KEY=...      # for web search grounding
   ```
3. **Run**
   ```bash
   python run.py \
     --goal "Suggest an existing drug that could be repurposed for AML with testable IC50 concentrations" \
     --rounds 2 --population 8 --keep-top 4 --seed 7
   ```

## Notes
- Swap `OpenAIChat` for your preferred `langchain_core` model; see `agents.py`.
- The tournament uses pairwise debates with ELO; winners seed the next evolution round.
- Reflection can call `web_search()` to retrieve literature snippets used as citations.