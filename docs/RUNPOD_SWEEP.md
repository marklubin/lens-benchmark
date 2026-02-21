# RunPod Multi-Pod Sweep Guide

Run the full LENS benchmark (144 runs: 8 adapters x 6 scopes x 3 budgets) across 6 RunPod worker pods in ~2 hours.

## Prerequisites

- `RUNPOD_API_KEY` env var set
- SSH key at `~/.ssh/id_ed25519` (added to RunPod account settings)
- Dev dependencies installed: `uv sync --extra dev`
- Docker image pushed: `ghcr.io/marklubin/ollama-embed-ssh:latest`

## Architecture

```
Local Machine                        RunPod Infrastructure
+--------------------------+         +-----------------------------------+
| pod_manager.py           |         | lens-embed (L4, $0.48/hr)        |
|   create-volume          |--API--->|   ollama: nomic-embed-text        |
|   create-embed           |         |   SSH on port 22                  |
|   submit-sweep           |         |                                   |
|   ssh-config             |         | lens-worker-{group} x6 (A100)    |
|   status                 |         |   vLLM: Qwen3-32B-AWQ            |
|   destroy-pods           |         |   adapter containers              |
+--------------------------+         |                                   |
                                     | Network volume (100GB, persists)  |
                                     |   AWQ weights cached (~18GB)      |
                                     +-----------------------------------+
```

## Quick Start

### 1. Create volume (one-time)

```bash
python3 scripts/pod_manager.py create-volume
```

Finds existing `lens-data` volume or creates one. Volumes persist across sweeps.

### 2. Launch embedding pod

```bash
python3 scripts/pod_manager.py create-embed
```

Creates `lens-embed` on an NVIDIA L4 ($0.48/hr) using the pre-built image with nomic-embed-text already loaded. SSH-enabled.

### 3. Get SSH access

```bash
# Wait ~1 min for pod to boot, then:
python3 scripts/pod_manager.py ssh-config
ssh lens-embed  # verify it works
```

`ssh-config` reads the RunPod API and writes the pod's public SSH host/port into `~/.ssh/config`.

### 4. Smoke test (8 runs)

```bash
python3 scripts/pod_manager.py submit-sweep --smoke-test
```

Runs each adapter on scope-01/standard only. Check timing — every adapter must finish under 5 minutes.

### 5. Monitor progress

```bash
# Pod status
python3 scripts/pod_manager.py list-pods

# Git-based progress (commits per sweep branch)
python3 scripts/pod_manager.py status

# SSH into a worker for live logs
python3 scripts/pod_manager.py ssh-config
ssh lens-worker-graphiti
```

### 6. Full sweep (144 runs)

```bash
python3 scripts/pod_manager.py submit-sweep
```

### 7. Collect results

```bash
python3 scripts/pod_manager.py merge-results
python3 scripts/collect_results.py
```

### 8. Cleanup

```bash
# Destroy ALL pods (workers + embed)
python3 scripts/pod_manager.py destroy-pods

# Or keep embed running for the next sweep
python3 scripts/pod_manager.py destroy-pods --keep-embed
```

Volume is never destroyed by `destroy-pods` — it persists for future sweeps with cached model weights.

## Pod Groups

| Pod | GPU | Adapters | Containers | Cost/hr |
|-----|-----|----------|------------|---------|
| `lens-embed` | L4 | -- | ollama | $0.48 |
| `lens-worker-fast` | A100 80GB | chunked-hybrid, compaction | -- | $1.19 |
| `lens-worker-letta` | A100 80GB | letta, letta-sleepy | letta | $1.19 |
| `lens-worker-mem0` | A100 80GB | mem0-raw | qdrant | $1.19 |
| `lens-worker-cognee` | A100 80GB | cognee | -- | $1.19 |
| `lens-worker-graphiti` | A100 80GB | graphiti | falkordb | $1.19 |
| `lens-worker-hindsight` | A100 80GB | hindsight | hindsight | $1.19 |

**Total**: ~$7.60/hr for all 7 pods. Full sweep (~2 hrs) costs ~$15.

## Timeouts

- **Run**: 5 minutes hard limit. If exceeded, the adapter is broken — check logs and fix.
- **Score**: 2 minutes per run.
- **vLLM boot**: 5 minutes (includes AWQ model download on first use).

## SSH Config

`~/.ssh/config` entries are managed by `pod_manager.py ssh-config`. The entries use:
- `StrictHostKeyChecking no` — RunPod host keys change every boot
- `UserKnownHostsFile /dev/null` — don't pollute known_hosts
- `User root` — RunPod containers run as root

Manual override: edit the `HostName` and `Port` for `lens-embed` or `lens-worker-*` in `~/.ssh/config`.

## Troubleshooting

### Adapter exceeds 5-minute timeout

```bash
# SSH into the worker
python3 scripts/pod_manager.py ssh-config
ssh lens-worker-graphiti

# Check vLLM health
curl localhost:8000/health

# Check adapter container
podman ps
podman logs falkordb
```

### Embed pod not reachable from workers

Workers connect to the embed pod via its public IP. Verify:
```bash
# On the worker pod:
curl http://<embed-ip>:11434/v1/models
```

### vLLM OOM or CUDA errors

AWQ model (~18GB) + KV cache for 16 seqs should fit in A100 80GB. If OOM:
- Reduce `--max-num-seqs` (default 16)
- Check if adapter containers are consuming GPU memory

## Rebuilding the Embed Image

```bash
cd ~/runpod-infra/images/ollama-embed-ssh
podman build -t ghcr.io/marklubin/ollama-embed-ssh:latest .
podman push ghcr.io/marklubin/ollama-embed-ssh:latest
```

The image pre-loads nomic-embed-text at build time for instant boot.
