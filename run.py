from __future__ import annotations

import argparse
import logging

from coscientist.graph_app import build_app
from coscientist.state import CoScientistState, ResearchGoal
import uuid
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("coscientist.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--goal", required=True, help="Research goal in natural language")
parser.add_argument("--rounds", type=int, default=1)
parser.add_argument("--population", type=int, default=2)
parser.add_argument("--keep-top", type=int, default=2)
parser.add_argument("--shortlist", type=int, default=2)
parser.add_argument("--seed", type=int, default=0)
args = parser.parse_args()

logger.info(f"Starting CoScientist with arguments: {vars(args)}")

initial: CoScientistState = {
    "goal": ResearchGoal(text=args.goal),
    "round_index": 0,
    "population": [],
    "reviews": {},
    "tournament": None,
    "overview": None,
    "params": {
        "rounds": args.rounds,
        "population": args.population,
        "keep_top": args.keep_top,
        "shortlist": args.shortlist,
        "seed": args.seed,
    },
}

logger.info(f"Initialized state with research goal: {args.goal}")
logger.info(
    f"Parameters: rounds={args.rounds}, population={args.population}, "
    f"keep_top={args.keep_top}, shortlist={args.shortlist}, seed={args.seed}"
)

try:
    logger.info("Building and compiling application...")
    app = build_app(rounds=args.rounds).compile()

    logger.info("Invoking application...")
    final = app.invoke(initial)

    logger.info("Application completed successfully")

    # Sort population once
    sorted_population = sorted(final["population"], key=lambda x: x.score, reverse=True)

    # Generate unique filename with UUID and datetime
    now = datetime.now()
    date_time_str = now.strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())
    filename = f"research_output_{date_time_str}_{unique_id}.md"
    
    # Prepare output content
    output_content = []
    output_content.append(f"# Research Results - {now.strftime('%Y-%m-%d %H:%M:%S')}\n")
    output_content.append(f"**Research Goal:** {args.goal}\n")
    
    output_content.append("## Research Overview\n")
    output_content.append(final["overview"])
    
    output_content.append("\n## Shortlisted Hypotheses\n")
    for h in sorted_population[: args.shortlist]:
        output_content.append(f"- **{h.text}** (score={h.score:.1f}, gen={h.generation}, id={h.id[:8]})")
    
    # Write to markdown file
    try:
        with open(filename, 'w') as f:
            f.write('\n'.join(output_content))
        logger.info(f"Research output saved to {filename}")
    except Exception as e:
        logger.error(f"Failed to save research output to file: {str(e)}")

    # Print to console as before
    print("\n=== RESEARCH OVERVIEW ===\n")
    print(final["overview"])
    logger.info("Research overview generated")

    print("\n=== SHORTLIST ===\n")
    logger.info(f"Generating shortlist of top {args.shortlist} hypotheses")

    for h in sorted_population[: args.shortlist]:
        print(f"- {h.text} (score={h.score:.1f}, gen={h.generation}, id={h.id[:8]})")
        logger.debug(
            f"Hypothesis: {h.id[:8]}, Score: {h.score:.1f}, Generation: {h.generation}"
        )

    logger.info("Program completed successfully")

except Exception as e:
    logger.error(f"An error occurred: {str(e)}", exc_info=True)
    raise
