"""GraphRAG-light — Entity-graph-augmented retrieval adapter.

A clean, self-contained graph-based retrieval adapter using:
  - SQLite FTS5 for BM25 keyword search
  - networkx DiGraph for entity/relationship graph
  - OpenAI-compatible embeddings for semantic entity search

Three-signal Reciprocal Rank Fusion (RRF) at search time:
  1. BM25 over raw episode text
  2. Entity embedding similarity
  3. Graph neighborhood expansion (1-hop from matched entities)

Entity extraction happens in prepare() via LLM — ingest() is fast/offline.

Why this clusters with other data stores (~0.35-0.45):
  Graph structure captures *what entities exist* and *how they relate*, but
  LENS questions require understanding *how things change over time*. The
  graph is a static snapshot — it knows "A interacts with B" but doesn't
  encode that the *nature* of the interaction changed across episodes.

Requires:
    pip install networkx
    OpenAI-compatible LLM and embedding endpoints

Environment variables:
    LENS_LLM_API_BASE      LLM endpoint for entity extraction
    LENS_LLM_API_KEY       LLM API key
    LENS_LLM_MODEL         LLM model name
    LENS_EMBED_BASE_URL    Embedding endpoint
    LENS_EMBED_API_KEY     Embedding API key
    LENS_EMBED_MODEL       Embedding model name
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import urllib.request

from lens.adapters.base import (
    CapabilityManifest,
    Document,
    ExtraTool,
    FilterField,
    MemoryAdapter,
    SearchResult,
)
from lens.adapters.registry import register_adapter
from lens.adapters.sqlite import _fts5_escape

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ENTITY_TYPES = {"ACTOR", "OBJECT", "PLACE", "CONCEPT", "EVENT"}

_RELATIONSHIP_TYPES = {
    "RELATED_TO", "ACTED_ON", "PRODUCED", "BELONGS_TO",
    "OCCURRED_AT", "PRECEDES", "CONTRADICTS",
}

_EXTRACT_PROMPT = """\
Extract entities and relationships from this episode text.

Return ONLY valid JSON with this schema:
{
  "entities": [
    {"name": "exact name", "type": "TYPE", "description": "one-line description"}
  ],
  "relationships": [
    {"source": "entity name", "target": "entity name", "type": "TYPE", "description": "one-line description"}
  ]
}

Entity types:
- ACTOR: people, orgs, systems, agents (anything that acts)
- OBJECT: documents, metrics, artifacts, evidence items
- PLACE: locations, venues, jurisdictions
- CONCEPT: ideas, patterns, hypotheses, conditions
- EVENT: specific occurrences with temporal anchoring

Relationship types:
- RELATED_TO: general association (catch-all)
- ACTED_ON: subject performed action on object
- PRODUCED: created, generated, caused
- BELONGS_TO: membership, ownership, containment
- OCCURRED_AT: temporal/spatial anchoring
- PRECEDES: temporal ordering
- CONTRADICTS: opposing evidence/claims

Rules:
- Extract concrete, named entities only (not generic terms like "the system")
- Resolve pronouns to their referent: if text says "he reported" and referent is "Marcus Rivera", extract as "Marcus Rivera", not "he"
- Use the full proper name. "Dr. Sarah Chen" not "Dr. Chen" or "she". If multiple aliases exist, use the most complete/formal version.
- Keep descriptions under 100 characters
- Only include relationships between extracted entities
- If the text has no clear entities, return {"entities": [], "relationships": []}

Episode text:
"""

_DEFAULT_OPENAI_EMBED_MODEL = "text-embedding-3-small"
_EMBED_BATCH_SIZE = 20
_MAX_EMBED_CHARS = 6000

# ---------------------------------------------------------------------------
# Embedding helper (reuses pattern from sqlite_variants)
# ---------------------------------------------------------------------------


def _embed_texts_openai(
    texts: list[str],
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    _max_retries: int = 3,
) -> list[list[float]]:
    """Call OpenAI-compatible embedding API."""
    import time

    api_key = (
        api_key
        or os.environ.get("LENS_EMBED_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("LENS_LLM_API_KEY")
    )
    if not api_key:
        raise ValueError("Embedding API key required: set LENS_EMBED_API_KEY or OPENAI_API_KEY")

    if base_url:
        embed_url = base_url.rstrip("/")
        if not embed_url.endswith("/embeddings"):
            embed_url += "/embeddings"
    else:
        embed_url = "https://api.openai.com/v1/embeddings"

    # Truncate long texts
    texts = [t[:_MAX_EMBED_CHARS] for t in texts]

    body = json.dumps({"model": model, "input": texts}).encode()
    for attempt in range(_max_retries):
        req = urllib.request.Request(
            embed_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "lens-benchmark/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
                sorted_data = sorted(result["data"], key=lambda d: d["index"])
                return [d["embedding"] for d in sorted_data]
        except urllib.error.HTTPError as e:
            resp_body = ""
            try:
                resp_body = e.read().decode(errors="replace")
            except Exception:
                pass
            log.warning(
                "Embed HTTP %s (attempt %d/%d): %s",
                e.code, attempt + 1, _max_retries, resp_body[:500],
            )
            if attempt == _max_retries - 1:
                raise
            time.sleep(1 * (attempt + 1))
        except (urllib.error.URLError, OSError) as e:
            log.warning("Embed network error (attempt %d/%d): %s", attempt + 1, _max_retries, e)
            if attempt == _max_retries - 1:
                raise
            time.sleep(1 * (attempt + 1))
    return []  # unreachable, but satisfies type checker


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# LLM helper for entity extraction
# ---------------------------------------------------------------------------


def _llm_extract_entities(
    text: str,
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    _max_retries: int = 2,
) -> dict:
    """Extract entities and relationships from text via LLM.

    Returns {"entities": [...], "relationships": [...]}.
    On failure, returns empty lists.
    """
    import time

    model = model or os.environ.get("LENS_LLM_MODEL", "gpt-4o-mini")
    api_key = (
        api_key
        or os.environ.get("LENS_LLM_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    base_url = base_url or os.environ.get("LENS_LLM_API_BASE", "https://api.openai.com/v1")

    chat_url = base_url.rstrip("/")
    if not chat_url.endswith("/chat/completions"):
        chat_url += "/chat/completions"

    # Truncate very long episodes for extraction
    truncated = text[:8000] if len(text) > 8000 else text

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "user", "content": _EXTRACT_PROMPT + truncated},
        ],
        "temperature": 0.0,
        "max_tokens": 2000,
    }).encode()

    for attempt in range(_max_retries):
        req = urllib.request.Request(
            chat_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "lens-benchmark/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read())
                content = result["choices"][0]["message"]["content"]
                return _parse_extraction(content)
        except urllib.error.HTTPError as e:
            resp_body = ""
            try:
                resp_body = e.read().decode(errors="replace")
            except Exception:
                pass
            log.warning(
                "LLM extract HTTP %s (attempt %d/%d): %s",
                e.code, attempt + 1, _max_retries, resp_body[:500],
            )
            if attempt == _max_retries - 1:
                log.error("Entity extraction failed after %d attempts", _max_retries)
                return {"entities": [], "relationships": []}
            time.sleep(2 * (attempt + 1))
        except (urllib.error.URLError, OSError) as e:
            log.warning("LLM extract network error (attempt %d/%d): %s", attempt + 1, _max_retries, e)
            if attempt == _max_retries - 1:
                log.error("Entity extraction failed after %d attempts", _max_retries)
                return {"entities": [], "relationships": []}
            time.sleep(2 * (attempt + 1))
    return {"entities": [], "relationships": []}


def _parse_extraction(content: str) -> dict:
    """Parse LLM entity extraction output. Handles markdown code blocks and think tags."""
    # Strip <think>...</think> blocks (Qwen3.5 reasoning)
    content = re.sub(r'<think>[\s\S]*?</think>', '', content)
    content = content.strip()

    # Strip markdown code blocks
    if content.startswith("```"):
        lines = content.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = "\n".join(lines)

    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        # Use incremental decoder to find the first valid JSON object
        start = content.find("{")
        if start >= 0:
            decoder = json.JSONDecoder()
            try:
                data, _ = decoder.raw_decode(content, start)
            except json.JSONDecodeError:
                log.warning("Failed to parse entity extraction JSON")
                return {"entities": [], "relationships": []}
        else:
            log.warning("No JSON found in entity extraction response")
            return {"entities": [], "relationships": []}

    entities = data.get("entities", [])
    relationships = data.get("relationships", [])

    # Validate entity structure
    valid_entities = []
    for e in entities:
        if isinstance(e, dict) and "name" in e and "type" in e:
            valid_entities.append({
                "name": str(e["name"]),
                "type": str(e.get("type", "CONCEPT")),
                "description": str(e.get("description", "")),
            })

    valid_rels = []
    for r in relationships:
        if isinstance(r, dict) and "source" in r and "target" in r:
            valid_rels.append({
                "source": str(r["source"]),
                "target": str(r["target"]),
                "type": str(r.get("type", "RELATED_TO")),
                "description": str(r.get("description", "")),
            })

    return {"entities": valid_entities, "relationships": valid_rels}


# ---------------------------------------------------------------------------
# GraphRAG-light Adapter
# ---------------------------------------------------------------------------


@register_adapter("graphrag-light")
class GraphRAGLightAdapter(MemoryAdapter):
    """Graph-augmented retrieval adapter.

    Combines SQLite FTS5 keyword search, entity embedding similarity,
    and 1-hop graph neighborhood expansion via RRF fusion.
    """

    def __init__(
        self,
        db_path: str = ":memory:",
        embed_model: str | None = None,
        embed_api_key: str | None = None,
        embed_base_url: str | None = None,
        llm_model: str | None = None,
        llm_api_key: str | None = None,
        llm_base_url: str | None = None,
        rrf_k: int = 60,
    ) -> None:
        import networkx as nx

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._embed_model = embed_model or os.environ.get(
            "LENS_EMBED_MODEL", _DEFAULT_OPENAI_EMBED_MODEL
        )
        self._embed_api_key = embed_api_key
        self._embed_base_url = embed_base_url or os.environ.get("LENS_EMBED_BASE_URL")
        self._llm_model = llm_model
        self._llm_api_key = llm_api_key
        self._llm_base_url = llm_base_url
        self._rrf_k = rrf_k

        # Graph state
        self._graph: nx.DiGraph = nx.DiGraph()
        # Entity name (normalized) -> embedding vector
        self._entity_embeddings: dict[str, list[float]] = {}
        # Episodes pending entity extraction
        self._pending_episodes: list[str] = []

        self._ensure_tables()

    def _ensure_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                episode_id TEXT PRIMARY KEY,
                scope_id   TEXT NOT NULL,
                timestamp  TEXT NOT NULL,
                text       TEXT NOT NULL,
                meta       TEXT DEFAULT '{}'
            );
            CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts
                USING fts5(episode_id, text, content=episodes, content_rowid=rowid);

            CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
                INSERT INTO episodes_fts(rowid, episode_id, text)
                    VALUES (new.rowid, new.episode_id, new.text);
            END;
            CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
                INSERT INTO episodes_fts(episodes_fts, rowid, episode_id, text)
                    VALUES ('delete', old.rowid, old.episode_id, old.text);
            END;
        """)
        self._conn.commit()

    @staticmethod
    def _normalize_entity(name: str) -> str:
        """Normalize entity name for deduplication."""
        import re as _re
        n = name.strip().lower()
        n = _re.sub(r'\s+', ' ', n)
        for article in ("the ", "a ", "an "):
            if n.startswith(article):
                n = n[len(article):]
                break
        return n

    # --- Entity deduplication ---

    def _find_dedup_candidates(self, norm_name: str, ent_data: dict) -> list[str]:
        """Find existing graph entities that might be the same as *norm_name*.

        Stage 1: name-based (substring containment or >50% token overlap).
        Stage 2: embedding-based (cosine > 0.85) if no name matches and embeddings exist.
        Returns up to 5 candidate normalized names.
        """
        candidates: list[str] = []

        # Stage 1 — name-based
        cand_tokens = set(norm_name.split())
        for existing in self._graph.nodes:
            if existing == norm_name:
                continue
            # Substring containment
            if norm_name in existing or existing in norm_name:
                candidates.append(existing)
                continue
            # Token overlap
            ex_tokens = set(existing.split())
            if cand_tokens and ex_tokens:
                overlap = len(cand_tokens & ex_tokens)
                min_len = min(len(cand_tokens), len(ex_tokens))
                if min_len > 0 and overlap / min_len > 0.5:
                    candidates.append(existing)

        # Stage 2 — embedding-based (only if no name matches and embeddings exist)
        if not candidates and self._entity_embeddings:
            ent_text = f"{ent_data.get('name', norm_name)}: {ent_data.get('description', '')}"
            try:
                query_vec = _embed_texts_openai(
                    [ent_text],
                    model=self._embed_model,
                    api_key=self._embed_api_key,
                    base_url=self._embed_base_url,
                )[0]
            except Exception:
                log.warning("Dedup embedding failed for %s", norm_name)
                return []

            for ex_name, ex_vec in self._entity_embeddings.items():
                if ex_name == norm_name:
                    continue
                sim = _cosine_similarity(query_vec, ex_vec)
                if sim > 0.85:
                    candidates.append(ex_name)

        return candidates[:5]

    def _llm_dedup_check(
        self,
        candidate_name: str,
        candidate_data: dict,
        existing_names: list[str],
    ) -> dict | None:
        """Ask LLM whether *candidate_name* matches any of *existing_names*.

        Returns ``{"existing_name": ..., "canonical_name": ...}`` on match,
        ``None`` otherwise. Fails open on any error.
        """
        import time

        model = self._llm_model or os.environ.get("LENS_LLM_MODEL", "gpt-4o-mini")
        api_key = (
            self._llm_api_key
            or os.environ.get("LENS_LLM_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        base_url = self._llm_base_url or os.environ.get("LENS_LLM_API_BASE", "https://api.openai.com/v1")

        chat_url = base_url.rstrip("/")
        if not chat_url.endswith("/chat/completions"):
            chat_url += "/chat/completions"

        # Build existing entity descriptions
        existing_descs: list[str] = []
        for en in existing_names:
            node = self._graph.nodes.get(en, {})
            desc = node.get("description", "")
            display = node.get("name", en)
            existing_descs.append(f"- {display} (normalized: {en}): {desc}")

        prompt = (
            "Are any of these existing entities the same as the candidate? "
            "Consider name variations, abbreviations, titles, contextual equivalence.\n\n"
            f"Candidate: {candidate_name}\n"
            f"Description: {candidate_data.get('description', '')}\n\n"
            "Existing entities:\n" + "\n".join(existing_descs) + "\n\n"
            'Return JSON: {"match": true, "existing_name": "<normalized name>", "canonical_name": "<best display name>"} '
            'or {"match": false}'
        )

        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 200,
        }).encode()

        for attempt in range(2):
            req = urllib.request.Request(
                chat_url,
                data=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "lens-benchmark/1.0",
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    result = json.loads(resp.read())
                    content = result["choices"][0]["message"]["content"]
                    # Strip markdown fences
                    content = content.strip()
                    if content.startswith("```"):
                        lines = content.split("\n")
                        if lines[0].startswith("```"):
                            lines = lines[1:]
                        if lines and lines[-1].strip() == "```":
                            lines = lines[:-1]
                        content = "\n".join(lines)
                    data = json.loads(content)
                    if data.get("match"):
                        return {
                            "existing_name": data["existing_name"],
                            "canonical_name": data.get("canonical_name", candidate_name),
                        }
                    return None
            except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
                log.warning("Dedup LLM HTTP error (attempt %d/2): %s", attempt + 1, e)
                if attempt == 0:
                    time.sleep(1)
                continue
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                log.warning("Dedup LLM parse error: %s", e)
                return None

        return None  # fail open

    def _merge_entity(
        self,
        existing_norm: str,
        new_norm: str,
        new_data: dict,
        canonical_name: str | None = None,
    ) -> None:
        """Merge *new_data* into the existing graph node at *existing_norm*."""
        node = self._graph.nodes[existing_norm]
        node["source_episodes"] = node.get("source_episodes", set()) | new_data.get("source_episodes", set())
        if len(new_data.get("description", "")) > len(node.get("description", "")):
            node["description"] = new_data["description"]
        if canonical_name and canonical_name != node.get("name"):
            node["name"] = canonical_name

    # --- Data loading ---

    def reset(self, scope_id: str) -> None:
        import networkx as nx

        cur = self._conn.cursor()
        cur.execute("DELETE FROM episodes WHERE scope_id = ?", (scope_id,))
        self._conn.commit()
        self._graph = nx.DiGraph()
        self._entity_embeddings.clear()
        self._pending_episodes.clear()

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        """Store episode text in SQLite FTS. No LLM calls."""
        cur = self._conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO episodes (episode_id, scope_id, timestamp, text, meta) "
            "VALUES (?, ?, ?, ?, ?)",
            (episode_id, scope_id, timestamp, text, json.dumps(meta or {})),
        )
        self._conn.commit()
        self._pending_episodes.append(episode_id)

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        """Extract entities from pending episodes, build graph, compute embeddings."""
        if not self._pending_episodes:
            return

        log.info(
            "GraphRAG prepare: extracting entities from %d pending episodes",
            len(self._pending_episodes),
        )

        # 1. Extract entities from each pending episode
        new_entities: dict[str, dict] = {}  # normalized_name -> {type, description, source_episodes}
        new_relationships: list[dict] = []

        for ep_id in self._pending_episodes:
            cur = self._conn.cursor()
            cur.execute("SELECT text FROM episodes WHERE episode_id = ?", (ep_id,))
            row = cur.fetchone()
            if not row:
                continue

            extraction = _llm_extract_entities(
                row["text"],
                model=self._llm_model,
                api_key=self._llm_api_key,
                base_url=self._llm_base_url,
            )

            for entity in extraction["entities"]:
                norm_name = self._normalize_entity(entity["name"])
                if norm_name in new_entities:
                    # Merge: append source episode
                    new_entities[norm_name]["source_episodes"].add(ep_id)
                    # Keep longer description
                    if len(entity["description"]) > len(new_entities[norm_name]["description"]):
                        new_entities[norm_name]["description"] = entity["description"]
                else:
                    new_entities[norm_name] = {
                        "name": entity["name"],
                        "type": entity["type"],
                        "description": entity["description"],
                        "source_episodes": {ep_id},
                    }

            for rel in extraction["relationships"]:
                rel["source_episodes"] = {ep_id}
                new_relationships.append(rel)

        # 1b. Deduplicate new entities against existing graph
        merge_map: dict[str, str] = {}  # new_norm -> existing_norm
        for norm_name, ent_data in new_entities.items():
            if self._graph.has_node(norm_name):
                continue  # exact match — will merge in step 2
            candidates = self._find_dedup_candidates(norm_name, ent_data)
            if candidates:
                result = self._llm_dedup_check(ent_data["name"], ent_data, candidates)
                if result is not None:
                    merge_map[norm_name] = result["existing_name"]
                    self._merge_entity(
                        result["existing_name"], norm_name, ent_data,
                        result.get("canonical_name"),
                    )

        # 2. Update graph with new entities and relationships
        for norm_name, ent_data in new_entities.items():
            if norm_name in merge_map:
                continue  # already merged into existing node
            if self._graph.has_node(norm_name):
                # Merge source episodes
                existing = self._graph.nodes[norm_name]
                existing["source_episodes"] = existing["source_episodes"] | ent_data["source_episodes"]
                if len(ent_data["description"]) > len(existing.get("description", "")):
                    existing["description"] = ent_data["description"]
            else:
                self._graph.add_node(
                    norm_name,
                    name=ent_data["name"],
                    type=ent_data["type"],
                    description=ent_data["description"],
                    source_episodes=ent_data["source_episodes"],
                )

        for rel in new_relationships:
            src = self._normalize_entity(rel["source"])
            tgt = self._normalize_entity(rel["target"])
            # Apply merge map to relationship endpoints
            src = merge_map.get(src, src)
            tgt = merge_map.get(tgt, tgt)
            # Only add edge if both nodes exist
            if self._graph.has_node(src) and self._graph.has_node(tgt):
                if self._graph.has_edge(src, tgt):
                    edge = self._graph.edges[src, tgt]
                    edge["weight"] = edge.get("weight", 1) + 1
                    edge["source_episodes"] = edge["source_episodes"] | rel["source_episodes"]
                else:
                    self._graph.add_edge(
                        src, tgt,
                        type=rel["type"],
                        description=rel["description"],
                        source_episodes=rel["source_episodes"],
                        weight=1,
                    )

        # 3. Compute embeddings for new entities (skip merged ones)
        entities_needing_embeddings = [
            norm_name for norm_name in new_entities
            if norm_name not in self._entity_embeddings and norm_name not in merge_map
        ]

        if entities_needing_embeddings:
            # Build embedding texts: "name: description"
            embed_texts = []
            for norm_name in entities_needing_embeddings:
                node = self._graph.nodes[norm_name]
                text = f"{node['name']}: {node.get('description', '')}"
                embed_texts.append(text)

            # Batch embed
            all_vectors: list[list[float]] = []
            for i in range(0, len(embed_texts), _EMBED_BATCH_SIZE):
                batch = embed_texts[i : i + _EMBED_BATCH_SIZE]
                vectors = _embed_texts_openai(
                    batch,
                    model=self._embed_model,
                    api_key=self._embed_api_key,
                    base_url=self._embed_base_url,
                )
                all_vectors.extend(vectors)

            for norm_name, vec in zip(entities_needing_embeddings, all_vectors):
                self._entity_embeddings[norm_name] = vec

        log.info(
            "GraphRAG prepare: graph has %d nodes, %d edges, %d embeddings",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
            len(self._entity_embeddings),
        )

        self._pending_episodes.clear()

    # --- Search methods ---

    def _fts_search(
        self, query: str, filters: dict | None, limit: int,
    ) -> list[SearchResult]:
        """BM25 keyword search over episode text."""
        safe_query = _fts5_escape(query)
        if not safe_query:
            return []

        sql = (
            "SELECT e.episode_id, e.text, e.meta, bm25(episodes_fts) AS rank "
            "FROM episodes_fts f "
            "JOIN episodes e ON e.episode_id = f.episode_id "
            "WHERE episodes_fts MATCH ? "
        )
        params: list = [safe_query]
        if filters:
            if "scope_id" in filters:
                sql += "AND e.scope_id = ? "
                params.append(filters["scope_id"])
            if "start_date" in filters:
                sql += "AND e.timestamp >= ? "
                params.append(filters["start_date"])
            if "end_date" in filters:
                sql += "AND e.timestamp <= ? "
                params.append(filters["end_date"])
        sql += "ORDER BY rank LIMIT ?"
        params.append(limit)

        cur = self._conn.cursor()
        try:
            cur.execute(sql, params)
        except sqlite3.OperationalError:
            return []

        results = []
        for row in cur.fetchall():
            meta = json.loads(row["meta"]) if row["meta"] else {}
            results.append(
                SearchResult(
                    ref_id=row["episode_id"],
                    text=row["text"][:500],
                    score=abs(row["rank"]),
                    metadata=meta,
                )
            )
        return results

    def _entity_embedding_search(
        self, query: str, top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Find entities most similar to query by embedding.

        Returns list of (normalized_entity_name, similarity_score).
        """
        if not self._entity_embeddings:
            return []

        query_vec = _embed_texts_openai(
            [query],
            model=self._embed_model,
            api_key=self._embed_api_key,
            base_url=self._embed_base_url,
        )[0]

        scored = []
        for norm_name, emb in self._entity_embeddings.items():
            sim = _cosine_similarity(query_vec, emb)
            scored.append((norm_name, sim))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _graph_neighborhood_episodes(
        self, entity_matches: list[tuple[str, float]], hop_decay: float = 0.5,
    ) -> dict[str, float]:
        """Expand matched entities to 1-hop graph neighborhood.

        Returns {episode_id: score} aggregating entity similarity × edge weight × decay.
        """
        episode_scores: dict[str, float] = {}

        for norm_name, entity_sim in entity_matches:
            if not self._graph.has_node(norm_name):
                continue

            # Direct entity episodes
            source_eps = self._graph.nodes[norm_name].get("source_episodes", set())
            for ep_id in source_eps:
                episode_scores[ep_id] = episode_scores.get(ep_id, 0) + entity_sim

            # 1-hop neighbors (both successors and predecessors)
            neighbors = set(self._graph.successors(norm_name)) | set(self._graph.predecessors(norm_name))
            for neighbor in neighbors:
                # Edge weight (use max of both directions)
                weight = 1
                if self._graph.has_edge(norm_name, neighbor):
                    weight = max(weight, self._graph.edges[norm_name, neighbor].get("weight", 1))
                if self._graph.has_edge(neighbor, norm_name):
                    weight = max(weight, self._graph.edges[neighbor, norm_name].get("weight", 1))

                neighbor_eps = self._graph.nodes[neighbor].get("source_episodes", set())
                hop_score = entity_sim * hop_decay * min(weight, 5) / 5.0  # cap weight influence
                for ep_id in neighbor_eps:
                    episode_scores[ep_id] = episode_scores.get(ep_id, 0) + hop_score

        return episode_scores

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Three-signal RRF fusion search."""
        if not query or not query.strip():
            return []

        limit = limit or 10
        fetch_limit = limit * 3

        # Signal 1: BM25 keyword search
        fts_results = self._fts_search(query, filters, fetch_limit)

        # Signal 2 + 3: Entity embedding + graph neighborhood
        entity_matches = self._entity_embedding_search(query, top_k=10)
        graph_episode_scores = self._graph_neighborhood_episodes(entity_matches)

        # Convert graph scores to ranked SearchResult list
        graph_ranked_ids = sorted(
            graph_episode_scores.keys(),
            key=lambda eid: graph_episode_scores[eid],
            reverse=True,
        )[:fetch_limit]

        # Build graph results (need episode text for SearchResult)
        graph_results: list[SearchResult] = []
        for rank_idx, ep_id in enumerate(graph_ranked_ids):
            cur = self._conn.cursor()
            cur.execute("SELECT text, meta FROM episodes WHERE episode_id = ?", (ep_id,))
            row = cur.fetchone()
            if row:
                meta = json.loads(row["meta"]) if row["meta"] else {}
                graph_results.append(
                    SearchResult(
                        ref_id=ep_id,
                        text=row["text"][:500],
                        score=graph_episode_scores[ep_id],
                        metadata=meta,
                    )
                )

        # RRF merge: fts_results × graph_results
        return _rrf_merge(fts_results, graph_results, k=self._rrf_k, limit=limit)

    def retrieve(self, ref_id: str) -> Document | None:
        cur = self._conn.cursor()
        cur.execute(
            "SELECT episode_id, text, meta FROM episodes WHERE episode_id = ?",
            (ref_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        meta = json.loads(row["meta"]) if row["meta"] else {}
        return Document(ref_id=row["episode_id"], text=row["text"], metadata=meta)

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["keyword", "semantic", "graph"],
            filter_fields=[
                FilterField(name="scope_id", field_type="string", description="Filter by scope ID"),
                FilterField(name="start_date", field_type="string", description="Filter episodes after this ISO date"),
                FilterField(name="end_date", field_type="string", description="Filter episodes before this ISO date"),
            ],
            max_results_per_search=5,
            supports_date_range=True,
            extra_tools=[
                ExtraTool(
                    name="batch_retrieve",
                    description="Retrieve multiple episodes at once by their IDs. More efficient than calling retrieve() in a loop.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "ref_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of episode IDs to retrieve",
                            }
                        },
                        "required": ["ref_ids"],
                    },
                )
            ],
        )

    def call_extended_tool(self, tool_name: str, arguments: dict) -> object:
        if tool_name == "batch_retrieve":
            ref_ids = arguments.get("ref_ids", [])
            results = []
            for rid in ref_ids:
                doc = self.retrieve(rid)
                if doc:
                    results.append(doc.to_dict())
            return results
        msg = f"Unknown extended tool: {tool_name!r}"
        raise NotImplementedError(msg)


# ---------------------------------------------------------------------------
# RRF merge utility
# ---------------------------------------------------------------------------


def _rrf_merge(
    results_a: list[SearchResult],
    results_b: list[SearchResult],
    k: int = 60,
    limit: int = 10,
) -> list[SearchResult]:
    """Merge two ranked lists using Reciprocal Rank Fusion.

    RRF score(d) = 1/(k + rank_a) + 1/(k + rank_b)
    Documents not present in a list get rank = len(list) + 1.
    """
    all_results: dict[str, SearchResult] = {}
    for r in results_a:
        all_results[r.ref_id] = r
    for r in results_b:
        if r.ref_id not in all_results:
            all_results[r.ref_id] = r

    ranks_a = {r.ref_id: i + 1 for i, r in enumerate(results_a)}
    ranks_b = {r.ref_id: i + 1 for i, r in enumerate(results_b)}

    default_rank_a = len(results_a) + 1
    default_rank_b = len(results_b) + 1

    scored: list[tuple[float, SearchResult]] = []
    for ref_id, result in all_results.items():
        ra = ranks_a.get(ref_id, default_rank_a)
        rb = ranks_b.get(ref_id, default_rank_b)
        rrf_score = 1.0 / (k + ra) + 1.0 / (k + rb)
        scored.append((
            rrf_score,
            SearchResult(ref_id=result.ref_id, text=result.text, score=rrf_score, metadata=result.metadata),
        ))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[1] for s in scored[:limit]]
