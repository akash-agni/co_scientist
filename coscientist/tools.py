from __future__ import annotations
import json, logging, os
from dataclasses import dataclass
from typing import Dict, List
from openai import OpenAI

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OpenAIWebSearch:
    # Use a standard model with the Responses API
    model: str = "gpt-4o-mini"
    k: int = 5

    def __post_init__(self):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE"),  # optional; remove if not using a proxy
        )

    def _parse_json_array(self, text: str) -> List[Dict]:
        try:
            start, end = text.find("["), text.rfind("]")
            if start != -1 and end > start:
                return json.loads(text[start:end+1])
        except Exception as e:
            logger.warning(f"JSON parse failed: {e}")
        return []

    def search(self, query: str) -> List[Dict]:
        prompt = (
            f"Use web search to find high-quality sources for: {query}\n\n"
            f"Return ONLY a JSON array (max {self.k}) with objects: "
            f'{{"title": str, "url": str, "snippet": str}}.'
        )
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=prompt,
                tools=[{"type": "web_search"}],  # built-in tool
            )
            text = resp.output_text or ""
            items = self._parse_json_array(text)
            return [
                {"title": d.get("title",""), "url": d.get("url",""), "content": d.get("snippet","")}
                for d in items
            ][: self.k]
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
