import os
import json
import logging
import requests
from typing import Dict, List, Optional, Any
import yaml
from .kv_store import KVStore


class Agent:
    """Persistent agent with key-value store memory."""

    def __init__(self, config_path: str = "config/config.yaml"):
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Initialize KV store
        db_path = self.config["database"]["path"].replace("${HOME}", os.path.expanduser("~"))
        self.kv_store = KVStore(db_path)

        # Initialize message history
        self.history: List[Dict[str, str]] = []
        self.max_history = self.config["agent"]["max_history"]

        # Initialize API client
        self.api_key = os.environ.get("KIMI_API_KEY")
        if not self.api_key:
            raise ValueError("KIMI_API_KEY environment variable not set")

        self.api_base_url = self.config["api"]["base_url"]
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # Mode: 'history' or 'kv_only'
        self.mode = "history"

    def set_mode(self, mode: str):
        """Set agent mode: 'history' or 'kv_only'."""
        if mode not in ["history", "kv_only"]:
            raise ValueError("Mode must be 'history' or 'kv_only'")
        self.mode = mode
        if mode == "kv_only":
            self.history = []  # Clear history in kv_only mode

    def _get_context(self, query: str) -> str:
        """Get context for the current query."""
        context_parts = []

        # In history mode, include recent messages
        if self.mode == "history" and self.history:
            context_parts.append("Recent conversation history:")
            for msg in self.history[-self.max_history :]:
                context_parts.append(f"{msg['role']}: {msg['content']}")

        # Always search KV store for relevant information
        # Extract potential entities from query
        entities = self._extract_entities(query)
        for entity_type, entity_name in entities:
            key = self.kv_store._canonicalize_key(entity_type, entity_name)
            value = self.kv_store.read(key)
            if value:
                context_parts.append(f"Memory ({key}): {value}")

        return "\n".join(context_parts) if context_parts else "No relevant context found."

    def _extract_entities(self, text: str) -> List[tuple]:
        """Extract potential entities from text."""
        entities = []

        # Simple entity extraction - in practice, you might use NER
        text_upper = text.upper()

        # Location entities
        locations = [
            "SAN FRANCISCO",
            "NEW YORK",
            "LOS ANGELES",
            "CHICAGO",
            "HOUSTON",
            "PHOENIX",
            "PHILADELPHIA",
            "SAN ANTONIO",
            "SAN DIEGO",
            "DALLAS",
            "SAN JOSE",
            "AUSTIN",
            "JACKSONVILLE",
            "FORT WORTH",
            "COLUMBUS",
            "SAN FRANCISCO BAY AREA",
            "BOSTON",
            "SEATTLE",
            "DENVER",
            "MIAMI",
        ]

        for loc in locations:
            if loc in text_upper:
                entities.append(("LOCATION", loc))

        # Organization entities
        orgs = [
            "GOOGLE",
            "MICROSOFT",
            "APPLE",
            "AMAZON",
            "FACEBOOK",
            "NETFLIX",
            "TESLA",
            "SPACE X",
            "ALIBABA",
            "TENCENT",
            "SAMSUNG",
        ]

        for org in orgs:
            if org in text_upper:
                entities.append(("ORGANIZATION", org))

        # Person entities
        persons = [
            "ALBERT EINSTEIN",
            "ISAAC NEWTON",
            "NIKOLA TESLA",
            "LEONARDO DA VINCI",
            "WILLIAM SHAKESPEARE",
            "NELSON MANDELA",
            "MARTIN LUTHER KING",
            "MALCOLM X",
        ]

        for person in persons:
            if person in text_upper:
                entities.append(("PERSON", person))

        # Time entities
        times = [
            "YESTERDAY",
            "TODAY",
            "TOMORROW",
            "LAST WEEK",
            "THIS WEEK",
            "NEXT WEEK",
            "LAST MONTH",
            "THIS MONTH",
            "NEXT MONTH",
            "LAST YEAR",
            "THIS YEAR",
            "NEXT YEAR",
        ]

        for time in times:
            if time in text_upper:
                entities.append(("TIME", time))

        return entities

    def _call_api(self, prompt: str) -> str:
        """Call the KIMI API with the given prompt."""
        try:
            response = requests.post(
                f"{self.api_base_url}/chat/completions",
                headers=self.headers,
                json={
                    "model": "kimi-large",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1024,
                },
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return "I apologize, but I'm having trouble connecting to the AI service. Please try again later."
        except (KeyError, IndexError) as e:
            logging.error(f"Unexpected API response format: {e}")
            return "I received an unexpected response from the AI service. Please try again."

    def respond(self, query: str) -> str:
        """Generate a response to the user's query."""
        # Get context from history and KV store
        context = self._get_context(query)

        # Prepare prompt for API
        prompt = f"Context:\n{context}\n\nUser query: {query}\n\nResponse:"

        # Call API to get response
        response = self._call_api(prompt)

        # Extract entities from response to store in KV store
        response_entities = self._extract_entities(response)
        for entity_type, entity_name in response_entities:
            key = self.kv_store._canonicalize_key(entity_type, entity_name)
            # Store the relevant part of the response
            self.kv_store.write(key, response)

        # Add to history
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": response})

        return response

    def get_kv_store_contents(self) -> List[Dict[str, Any]]:
        """Get all contents of the KV store."""
        keys = self.kv_store.get_all_keys()
        contents = []
        for key in keys:
            value = self.kv_store.read(key)
            contents.append({"key": key, "value": value})
        return contents

    def search_kv_store(self, pattern: str) -> List[Dict[str, Any]]:
        """Search the KV store for a pattern."""
        return self.kv_store.search(pattern)
