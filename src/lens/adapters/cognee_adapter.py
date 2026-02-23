"""Cognee (GraphRAG) memory adapter for LENS.

Uses Cognee's graph-aware knowledge pipeline: ingests episodes as text,
builds a knowledge graph via cognify(), and retrieves chunks via vector
search. All embedded databases (SQLite, LanceDB, Kuzu) — no extra container.

Requires:
    uv add cognee

Environment variables:
    COGNEE_LLM_API_KEY     LLM API key (required)
    COGNEE_LLM_MODEL       LLM model (default: meta-llama/Llama-3.3-70B-Instruct-Turbo)
    COGNEE_LLM_ENDPOINT    LLM API base URL (default: https://api.together.xyz/v1)
    COGNEE_EMBED_API_KEY   Embedding API key (required)
    COGNEE_EMBED_MODEL     Embedding model (default: Alibaba-NLP/gte-modernbert-base)
    COGNEE_EMBED_ENDPOINT  Embedding API base URL (default: https://api.together.xyz/v1)
    COGNEE_EMBED_DIMS      Embedding dimensions (default: 768)
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import uuid

from lens.adapters.base import (
    CapabilityManifest,
    Document,
    ExtraTool,
    MemoryAdapter,
    SearchResult,
)
from lens.adapters.registry import register_adapter
from lens.core.errors import AdapterError

log = logging.getLogger(__name__)

_EP_ID_RE = re.compile(r"^\[([^\]]+)\]")


def _parse_ep_id(text: str) -> str | None:
    """Extract episode_id from '[ep_id] ...' prefixed text."""
    m = _EP_ID_RE.match(text)
    return m.group(1) if m else None


# ---------------------------------------------------------------------------
# Thread-hosted event loop — avoids asyncio.run() cross-loop issues
# ---------------------------------------------------------------------------


class _AsyncRunner:
    """Hosts a persistent event loop in a daemon thread for sync→async bridging."""

    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._thread.start()

    def run(self, coro, timeout: float = 600.0):
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=timeout)


_RUNNER: _AsyncRunner | None = None
_RUNNER_LOCK = threading.Lock()

# Tracks whether we have already monkey-patched cognee's LiteLLMEmbeddingEngine.
# Together AI rejects the 'dimensions' parameter in embedding requests; this patch
# prevents cognee from ever sending it.
_COGNEE_PATCHED = False


def _get_runner() -> _AsyncRunner:
    global _RUNNER
    if _RUNNER is None:
        with _RUNNER_LOCK:
            if _RUNNER is None:
                _RUNNER = _AsyncRunner()
    return _RUNNER


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


@register_adapter("cognee")
class CogneeAdapter(MemoryAdapter):
    """Cognee GraphRAG adapter for LENS.

    Ingests episodes as raw text, builds a knowledge graph with LLM entity
    extraction (cognify), and retrieves episodes via CHUNKS vector search.

    Scope isolation: reset() calls cognee.prune() to clear all data, then
    assigns a fresh dataset name. Episodes are buffered in ingest() and
    processed in prepare() (before agent's budget clock).

    Note: cognee stores databases inside the package directory by default.
    Set COGNEE_DATA_DIR to override.
    """

    requires_metering: bool = False

    def __init__(self) -> None:
        self._llm_api_key = os.environ.get("COGNEE_LLM_API_KEY", "")
        self._llm_model = os.environ.get(
            "COGNEE_LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo"
        )
        self._llm_endpoint = os.environ.get(
            "COGNEE_LLM_ENDPOINT", "https://api.together.xyz/v1"
        )
        self._embed_api_key = os.environ.get("COGNEE_EMBED_API_KEY", "")
        self._embed_model = os.environ.get(
            "COGNEE_EMBED_MODEL", "Alibaba-NLP/gte-modernbert-base"
        )
        self._embed_endpoint = os.environ.get(
            "COGNEE_EMBED_ENDPOINT", "https://api.together.xyz/v1"
        )
        self._embed_dims = int(os.environ.get("COGNEE_EMBED_DIMS", "768"))

        # Propagate embedding config via env vars so pydantic settings pick them up
        self._apply_env_config()

        # Per-run state
        self._dataset_id: str | None = None
        self._text_cache: dict[str, str] = {}
        self._pending_episodes: list[dict] = []

    def _apply_env_config(self) -> None:
        """Propagate COGNEE_* env vars to the names cognee's pydantic settings read."""
        # Disable ACL to avoid race condition in cognee's user lookup
        os.environ.setdefault("ENABLE_BACKEND_ACCESS_CONTROL", "false")
        if self._embed_api_key:
            os.environ.setdefault("EMBEDDING_API_KEY", self._embed_api_key)
        if self._embed_model:
            # Pass the embedding model name as-is. Together AI models like
            # "Alibaba-NLP/gte-modernbert-base" must NOT be prefixed with
            # "together_ai/" — that compound name causes 422 errors.
            # The EMBEDDING_PROVIDER=openai setting (below) tells litellm to
            # route via OpenAI-compatible endpoint at EMBEDDING_ENDPOINT.
            os.environ.setdefault("EMBEDDING_MODEL", self._embed_model)
        # Set EMBEDDING_DIMENSIONS so LanceDB schema uses the correct vector size.
        # Together AI rejects the 'dimensions' API param — _get_cognee() patches embed_text
        # to suppress it at call time while keeping self.dimensions intact for schema use.
        os.environ.setdefault("EMBEDDING_DIMENSIONS", str(self._embed_dims))
        if self._embed_endpoint:
            os.environ.setdefault("EMBEDDING_ENDPOINT", self._embed_endpoint)
        os.environ.setdefault("EMBEDDING_PROVIDER", "openai")

        # Monkey-patch tiktoken to handle non-OpenAI embedding model names.
        # Cognee uses tiktoken for text chunking; it raises KeyError for models
        # it doesn't recognise (e.g. "Alibaba-NLP/gte-modernbert-base").
        # We fall back to cl100k_base, which is fine for chunk-size estimation.
        try:
            import tiktoken  # noqa: PLC0415

            _orig = tiktoken.encoding_for_model

            def _patched_encoding_for_model(model_name: str):
                try:
                    return _orig(model_name)
                except KeyError:
                    return tiktoken.get_encoding("cl100k_base")

            tiktoken.encoding_for_model = _patched_encoding_for_model
        except Exception:
            pass  # tiktoken not installed or patching failed — cognee will handle it

    def _get_cognee(self):
        """Lazy import cognee and configure LLM settings."""
        global _COGNEE_PATCHED  # noqa: PLW0603

        try:
            import cognee  # noqa: PLC0415
        except ImportError as e:
            raise AdapterError(
                "cognee not installed. Run: uv add cognee"
            ) from e

        # Monkey-patch litellm.aembedding once per process to strip the 'dimensions'
        # kwarg before any API call. Together AI rejects this param entirely, but
        # LiteLLMEmbeddingEngine always adds it when self.dimensions is not None.
        # We patch at the litellm level (not cognee) so self.dimensions stays 768,
        # keeping LanceDB schema creation (dimension_count()) correct.
        if not _COGNEE_PATCHED:
            try:
                import litellm as _litellm  # noqa: PLC0415

                # Patch 1: Strip 'dimensions' from embedding calls — Together AI rejects it
                _orig_aembedding = _litellm.aembedding

                async def _aembedding_no_dims(*args, **kwargs):
                    kwargs.pop("dimensions", None)
                    return await _orig_aembedding(*args, **kwargs)

                _litellm.aembedding = _aembedding_no_dims

                # Patch 2: Inject max_tokens for LLM calls — GenericAPIAdapter stores
                # max_completion_tokens but never passes it to litellm.acompletion().
                # Together AI defaults to ~8192 which is too low for graph extraction
                # on long operational log episodes.
                _orig_acompletion = _litellm.acompletion

                async def _acompletion_with_max_tokens(*args, **kwargs):
                    if "max_tokens" not in kwargs and "max_completion_tokens" not in kwargs:
                        kwargs["max_tokens"] = 16384
                    return await _orig_acompletion(*args, **kwargs)

                _litellm.acompletion = _acompletion_with_max_tokens

                # Clear embedding engine cache so fresh engine picks up config
                from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (  # noqa: PLC0415
                    create_embedding_engine as _cee,
                )

                _cee.cache_clear()
                log.debug("Patched litellm: suppressed dimensions, injected max_tokens=16384")
            except Exception as _pe:
                log.warning("Failed to patch litellm: %s", _pe)
            _COGNEE_PATCHED = True

        # Configure LLM programmatically (persists for process lifetime).
        # cognee uses litellm internally; model name must have provider prefix
        # e.g. "openai/Qwen/..." for OpenAI-compatible endpoints (Together AI).
        # The default model format in cognee is "openai/gpt-5-mini" — same pattern.
        llm_model = self._llm_model
        if not llm_model.startswith(("openai/", "together_ai/", "anthropic/", "bedrock/")):
            llm_model = f"openai/{llm_model}"
        try:
            cognee.config.set_llm_config({
                "llm_provider": "openai",
                "llm_model": llm_model,
                "llm_endpoint": self._llm_endpoint,
                "llm_api_key": self._llm_api_key,
            })
        except Exception as e:
            log.warning("cognee.config.set_llm_config failed: %s", e)

        return cognee

    def reset(self, scope_id: str, cache_key: str | None = None) -> None:
        """Clear all cognee data and set a fresh dataset name.

        Calls cognee.prune.prune_data() + prune_system() to wipe all graphs,
        vectors, and relational data — ensuring clean isolation between runs.

        Note: cognee.prune is a CLASS with static async methods, NOT a callable.
        cognee.prune() would just create a class instance (no-op). We must call
        the actual prune_data() and prune_system() static methods.

        If cache_key is provided, it is used as the suffix for deterministic
        dataset naming (enables cache reconnection).
        """
        suffix = cache_key or uuid.uuid4().hex[:8]
        safe_scope = "".join(c if c.isalnum() or c == "_" else "_" for c in scope_id)
        self._dataset_id = f"lens_{safe_scope}_{suffix}"
        self._text_cache = {}
        self._pending_episodes = []

        cognee = self._get_cognee()
        runner = _get_runner()

        # Belt-and-suspenders: remove cognee's database directory to guarantee
        # clean state. cognee.prune can leave dangling LanceDB references that
        # cause IO errors on subsequent runs.
        import shutil  # noqa: PLC0415

        cognee_db_dir = os.path.join(
            os.path.dirname(cognee.__file__),
            ".cognee_system",
            "databases",
        )
        if os.path.isdir(cognee_db_dir):
            try:
                shutil.rmtree(cognee_db_dir)
                log.debug("Removed cognee database directory: %s", cognee_db_dir)
            except OSError as e:
                log.warning("Failed to remove cognee DB dir: %s", e)

        cognee_cache_dir = os.path.join(
            os.path.dirname(cognee.__file__),
            ".cognee_cache",
        )
        if os.path.isdir(cognee_cache_dir):
            try:
                shutil.rmtree(cognee_cache_dir)
            except OSError as e:
                log.warning("Failed to remove cognee cache dir: %s", e)

        try:
            runner.run(cognee.prune.prune_data(), timeout=120.0)
            runner.run(
                cognee.prune.prune_system(
                    graph=True, vector=True, metadata=True, cache=True
                ),
                timeout=120.0,
            )
        except Exception as e:
            log.warning(
                "cognee.prune failed during reset for scope '%s': %s (continuing with fresh dataset)",
                scope_id,
                e,
            )

    def ingest(
        self,
        episode_id: str,
        scope_id: str,
        timestamp: str,
        text: str,
        meta: dict | None = None,
    ) -> None:
        """Buffer an episode for batch processing in prepare().

        Instant (<1ms) — no LLM or I/O calls. Episodes accumulate here and
        are flushed to cognee in prepare() before the agent's budget clock.
        """
        if self._dataset_id is None:
            raise AdapterError("reset() must be called before ingest()")

        content = f"[{episode_id}] {timestamp}: {text}"
        self._pending_episodes.append(
            {"episode_id": episode_id, "content": content}
        )
        self._text_cache[episode_id] = text

    def prepare(self, scope_id: str, checkpoint: int) -> None:
        """Flush buffered episodes to cognee and build the knowledge graph.

        Calls cognee.add() for each episode (stores raw chunks) then
        cognee.cognify() to extract entities and build the graph.
        This is where LLM processing happens — before agent's budget clock.
        """
        if not self._pending_episodes or self._dataset_id is None:
            return

        cognee = self._get_cognee()

        # Add each episode as raw text to cognee
        for item in self._pending_episodes:
            try:
                _get_runner().run(
                    cognee.add(
                        data=item["content"],
                        dataset_name=self._dataset_id,
                    ),
                    timeout=120.0,
                )
            except Exception as e:
                raise AdapterError(
                    f"cognee.add() failed for episode '{item['episode_id']}' "
                    f"at checkpoint {checkpoint}: {e}"
                ) from e

        # Build knowledge graph from added episodes.
        # cognify() does LLM entity extraction → graph + embeddings.
        # Non-fatal: if graph extraction hits token limits (common with dense
        # operational logs), chunks from earlier successful runs are still
        # searchable. Failing hard here would waste the entire run.
        try:
            _get_runner().run(
                cognee.cognify(datasets=[self._dataset_id]),
                timeout=600.0,
            )
        except Exception as e:
            log.warning(
                "cognee.cognify() failed at checkpoint %d (non-fatal, "
                "earlier chunks may still be searchable): %s",
                checkpoint,
                e,
            )

        self._pending_episodes = []

    def search(
        self,
        query: str,
        filters: dict | None = None,
        limit: int | None = None,
    ) -> list[SearchResult]:
        """Search cognee using CHUNKS vector search, returning episode-grounded results.

        Uses cognee's vector search over chunked episode text. Each chunk is
        prefixed with [episode_id], which we parse to build the ref_id for
        evidence grounding.

        Falls back to SUMMARIES search if CHUNKS collection doesn't exist
        (can happen if the cognify pipeline didn't create DocumentChunk_text).
        """
        if not query or not query.strip() or self._dataset_id is None:
            return []

        cap = limit or 10
        cognee = self._get_cognee()
        SearchType = cognee.SearchType

        raw = None
        # Don't filter by datasets= here — since reset() calls prune(), there's
        # only one dataset in the system. Passing datasets=[name] can cause
        # DatasetNotFoundError if cognee normalises the name differently.
        for search_type in (SearchType.CHUNKS, SearchType.SUMMARIES):
            try:
                raw = _get_runner().run(
                    cognee.search(
                        query_text=query,
                        query_type=search_type,
                        top_k=cap * 2,  # over-fetch for dedup
                    ),
                    timeout=60.0,
                )
                st_name = getattr(search_type, "name", str(search_type))
                log.debug(
                    "cognee.search(%s) returned %d items",
                    st_name,
                    len(raw) if raw else 0,
                )
                if raw:
                    break
            except Exception as e:
                st_name = getattr(search_type, "name", str(search_type))
                log.warning("cognee.search(%s) failed: %s", st_name, e)
                continue

        if not raw:
            log.warning(
                "cognee search returned empty for query=%r (tried CHUNKS + SUMMARIES)",
                query[:80],
            )
            return []

        search_results: list[SearchResult] = []
        seen_episode_ids: set[str] = set()

        for item in raw or []:
            if len(search_results) >= cap:
                break

            # Unwrap cognee SearchResult → payload dicts or plain strings
            chunk_list = self._extract_chunks(item)
            if not chunk_list:
                log.warning(
                    "_extract_chunks: empty from type=%s repr=%.200s",
                    type(item).__name__,
                    repr(item),
                )
            for chunk_text in chunk_list:
                if len(search_results) >= cap:
                    break
                if not chunk_text:
                    continue

                ep_id = _parse_ep_id(chunk_text)
                if ep_id is None:
                    # Best-effort: try matching against known episode text prefixes
                    ep_id = self._match_episode(chunk_text)

                if ep_id is None or ep_id in seen_episode_ids:
                    continue
                seen_episode_ids.add(ep_id)

                search_results.append(
                    SearchResult(
                        ref_id=ep_id,
                        text=chunk_text[:500],
                        score=0.5,
                    )
                )

        log.debug(
            "search() returning %d results",
            len(search_results),
        )
        return search_results

    def _extract_chunks(self, item) -> list[str]:
        """Unpack a cognee search result item into a list of chunk text strings.

        cognee.search() returns different shapes depending on whether ACL mode
        is enabled (cognee 0.5.2 enables it by default for Kuzu+LanceDB):

        ACL mode ON (default in 0.5.2):
            [{"dataset_id": UUID, "dataset_name": "...", "search_result": [chunk_dicts]}]
            Each item is a dict wrapper; actual chunks are under "search_result" key.

        ACL mode OFF:
            [chunk_dict_1, chunk_dict_2, ...]  (flat list of payload dicts)
            Each item is a dict with "text" and "id" keys.

        Also handles:
            - SearchResult-like objects with .search_result attribute (verbose mode)
            - Plain strings (only_context=True)
            - Lists of dicts/strings
        """
        # First try .search_result attribute (verbose mode / object-based results)
        search_result = getattr(item, "search_result", item)

        # ACL mode: dict with "search_result" key wrapping actual chunk data.
        # getattr() above doesn't work for plain dicts — check dict key access.
        if isinstance(search_result, dict) and "search_result" in search_result:
            search_result = search_result["search_result"]

        # Direct dict with "text" — a chunk payload dict (non-ACL mode)
        if isinstance(search_result, dict):
            text = search_result.get("text") or search_result.get("content") or ""
            return [str(text)] if text else []

        if isinstance(search_result, str):
            return [search_result] if search_result else []

        if isinstance(search_result, list):
            texts = []
            for entry in search_result:
                if isinstance(entry, str):
                    texts.append(entry)
                elif isinstance(entry, dict):
                    text = entry.get("text") or entry.get("content") or ""
                    if text:
                        texts.append(str(text))
            return texts

        log.warning(
            "_extract_chunks: unhandled type=%s repr=%.200s",
            type(search_result).__name__,
            repr(search_result),
        )
        return []

    def _match_episode(self, chunk_text: str) -> str | None:
        """Attempt to match a chunk back to an episode via substring matching."""
        chunk_lower = chunk_text.lower()[:200]
        for ep_id, ep_text in self._text_cache.items():
            # Check if the chunk overlaps significantly with the episode text
            ep_prefix = ep_text[:100].lower()
            if ep_prefix and ep_prefix[:50] in chunk_lower:
                return ep_id
        return None

    def retrieve(self, ref_id: str) -> Document | None:
        """Retrieve full episode text by episode_id from local cache."""
        text = self._text_cache.get(ref_id)
        if text is None:
            return None
        return Document(ref_id=ref_id, text=text)

    def get_capabilities(self) -> CapabilityManifest:
        return CapabilityManifest(
            search_modes=["semantic", "graph"],
            max_results_per_search=10,
            extra_tools=[
                ExtraTool(
                    name="batch_retrieve",
                    description=(
                        "Retrieve multiple full episodes by their reference IDs in a single call. "
                        "PREFER this over calling memory_retrieve multiple times — it uses only "
                        "one tool call instead of one per document. "
                        "After memory_search, pass all ref_ids you want to read to this tool."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "ref_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of reference IDs to retrieve.",
                            },
                        },
                        "required": ["ref_ids"],
                    },
                ),
            ],
        )

    def call_extended_tool(self, tool_name: str, arguments: dict) -> object:
        if tool_name == "batch_retrieve":
            ref_ids = arguments.get("ref_ids", [])
            docs = []
            for ref_id in ref_ids:
                doc = self.retrieve(ref_id)
                if doc is not None:
                    docs.append(doc.to_dict())
            return {"documents": docs, "count": len(docs)}
        return super().call_extended_tool(tool_name, arguments)

    def get_cache_state(self) -> dict | None:
        """Return state needed to reconnect to cognee's embedded databases."""
        if not self._dataset_id:
            return None
        return {
            "dataset_id": self._dataset_id,
            "text_cache": self._text_cache,
        }

    def restore_cache_state(self, state: dict) -> bool:
        """Restore from cached state — skip prune and cognify, just reconnect."""
        try:
            self._dataset_id = state["dataset_id"]
            self._text_cache = state.get("text_cache", {})
            self._pending_episodes = []
            # Ensure cognee module is loaded (patches applied)
            self._get_cognee()
            log.info(
                "Restored Cognee cache: dataset=%s, %d episodes",
                self._dataset_id,
                len(self._text_cache),
            )
            return True
        except Exception as e:
            log.warning("Failed to restore Cognee cache: %s", e)
            return False
