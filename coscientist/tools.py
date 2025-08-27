from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict, List

from openai import OpenAI

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class OpenAIWebSearch:
    """Wrapper around OpenAI's Responses API Web Search tool.

    It asks the model to perform a web search and return a compact JSON array
    of results with {title, url, snippet}. We keep the parser lenient.
    """

    model: str = "gpt-4o-mini"
    k: int = 5

    def __post_init__(self):
        logger.info(f"Initializing OpenAIWebSearch with model={self.model}, k={self.k}")
        self.client = OpenAI()
        logger.debug("OpenAI client initialized")

    def _parse_json_array(self, text: str) -> List[Dict]:
        logger.debug(f"Attempting to parse JSON array from text: {text[:100]}...")
        try:
            start = text.find("[")
            end = text.rfind("]")
            if start != -1 and end != -1 and end > start:
                logger.debug(f"Found JSON array bounds: start={start}, end={end}")
                parsed_data = json.loads(text[start : end + 1])
                logger.info(
                    f"Successfully parsed JSON array with {len(parsed_data)} items"
                )
                return parsed_data
        except Exception as e:
            logger.error(f"Failed to parse JSON array: {str(e)}")
            pass
        logger.warning("Returning empty list due to parsing failure")
        return []

    def search(self, query: str) -> List[Dict]:
        logger.info(f"Performing web search for query: {query}")
        prompt = (
            f"Use web search to find high‑quality sources for: {query}"
            f"Then return ONLY a JSON array of up to {self.k} objects with keys: title, url, snippet (<=240 chars)."
        )
        logger.debug(f"Generated prompt: {prompt}")

        try:
            logger.debug("Making API call to OpenAI")
            resp = self.client.responses.create(
                model=self.model,
                input=prompt,
                tools=[{"type": "web_search"}],
            )
            logger.debug("Received response from OpenAI")

            text = resp.output_text or ""
            logger.debug(f"Extracted text from response: {text[:100]}...")

            items = self._parse_json_array(text)
            logger.debug(f"Parsed {len(items)} items from response")

            result = [
                {
                    "title": d.get("title", ""),
                    "url": d.get("url", ""),
                    "content": d.get("snippet", ""),
                }
                for d in items
            ][: self.k]
            logger.info(f"Returning {len(result)} search results")
            return result

        except Exception as e:
            logger.error(f"Search failed with error: {str(e)}")
            return []
