"""Shared Modal infrastructure for LENS benchmark.

Defines the Modal app, volumes, and base images used by both
the LLM inference server and the embedding server.
"""
import modal

app = modal.App("lens-benchmark")

# Persistent volume for caching HuggingFace model weights across deployments.
model_cache = modal.Volume.from_name("lens-model-cache", create_if_missing=True)
MODEL_CACHE_PATH = "/root/.cache/huggingface"
