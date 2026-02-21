#!/usr/bin/env python3
"""Pod manager: local CLI for managing LENS sweep pods on RunPod.

Only touches pods/volumes named lens-*. Uses runpod Python library + REST API.

Usage:
    python3 scripts/pod_manager.py create-volume
    python3 scripts/pod_manager.py create-embed
    python3 scripts/pod_manager.py submit-sweep [--smoke-test] [--dry-run] [--group <name>]
    python3 scripts/pod_manager.py list-pods
    python3 scripts/pod_manager.py status
    python3 scripts/pod_manager.py logs <group>
    python3 scripts/pod_manager.py destroy-pods [--group <name>]
    python3 scripts/pod_manager.py merge-results

Environment:
    RUNPOD_API_KEY   - RunPod API key (required)
    LENS_REPO_URL    - Git repo URL for worker pods (required for submit-sweep)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pod_manager")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POD_PREFIX = "lens-"
# Worker GPUs — need >=48GB VRAM for Qwen3-32B-AWQ (~18GB) + KV cache
WORKER_GPU_TYPES = [
    "NVIDIA A100 80GB PCIe",  # 80GB, $1.19/hr — ideal
    "NVIDIA A100 SXM",        # 80GB, $1.39/hr
    "NVIDIA RTX A6000",       # 48GB, $0.33/hr — cheap, sufficient for AWQ
    "NVIDIA L40S",            # 48GB, $0.79/hr
    "NVIDIA L40",             # 48GB, $0.69/hr
]
EMBED_GPU_TYPES = [
    "NVIDIA L4",              # 24GB, $0.48/hr
    "NVIDIA RTX A4000",       # 16GB, $0.28/hr
    "NVIDIA RTX 4000 Ada Generation",  # 20GB, $0.34/hr
    "NVIDIA RTX A5000",       # 24GB, $0.36/hr
]
CLOUD_TYPE = "COMMUNITY"
DOCKER_IMAGE = "ghcr.io/marklubin/vllm-ssh:v0.8.5.post1"
EMBED_DOCKER_IMAGE = os.environ.get(
    "LENS_EMBED_IMAGE", "ghcr.io/marklubin/ollama-embed-ssh:latest"
)
VOLUME_NAME = "lens-data"
VOLUME_SIZE_GB = 100
CONTAINER_DISK_GB = 50

# Shared infra volume registry — check these first
KNOWN_VOLUMES = {
    "CA-MTL-3": "efy84wp5gx",
    "US-KS-2": "5u19x4ibbm",
    "US-NC-1": "fhh62pgdw3",
    "US-TX-3": "ignh1z4yjx",
}

# Worker pod definitions
WORKER_GROUPS = {
    "fast": {
        "adapters": ["chunked-hybrid", "compaction"],
        "containers": "",
    },
    "letta": {
        "adapters": ["letta", "letta-sleepy"],
        "containers": "letta",
    },
    "mem0": {
        "adapters": ["mem0-raw"],
        "containers": "qdrant",
    },
    "cognee": {
        "adapters": ["cognee"],
        "containers": "",
    },
    "graphiti": {
        "adapters": ["graphiti"],
        "containers": "falkordb",
    },
    "hindsight": {
        "adapters": ["hindsight"],
        "containers": "hindsight",
    },
}

SCOPES = ["01", "02", "03", "04", "05", "06"]
BUDGETS = ["standard", "4k", "2k"]


def _get_api_key() -> str:
    key = os.environ.get("RUNPOD_API_KEY", "")
    if not key:
        # Try reading from runpodctl config
        config_path = Path.home() / ".runpod" / "config.toml"
        if config_path.exists():
            for line in config_path.read_text().splitlines():
                if line.strip().startswith("apikey"):
                    key = line.split("=", 1)[1].strip().strip('"')
                    break
    if not key:
        log.error("RUNPOD_API_KEY not set and no ~/.runpod/config.toml found")
        sys.exit(1)
    return key


def _get_ssh_pubkey() -> str:
    """Read the user's SSH public key for pod injection."""
    key_path = os.environ.get("SSH_PUBKEY_PATH", "")
    if not key_path:
        key_path = str(Path.home() / ".ssh" / "id_ed25519.pub")
    p = Path(key_path)
    if not p.exists():
        log.warning("No SSH public key at %s — SSH into pods will fail", key_path)
        return ""
    return p.read_text().strip()


def _init_runpod():
    """Initialize runpod library with API key."""
    try:
        import runpod  # noqa: PLC0415
    except ImportError:
        log.error("runpod not installed. Run: uv add --dev runpod")
        sys.exit(1)
    runpod.api_key = _get_api_key()
    return runpod


def _rest_api(method: str, path: str, data: dict | None = None) -> dict:
    """Make a direct REST API call to RunPod (for volume management)."""
    import requests  # noqa: PLC0415

    api_key = _get_api_key()
    url = f"https://api.runpod.io/v2{path}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    if method == "GET":
        resp = requests.get(url, headers=headers, timeout=30)
    elif method == "POST":
        resp = requests.post(url, headers=headers, json=data or {}, timeout=30)
    elif method == "DELETE":
        resp = requests.delete(url, headers=headers, timeout=30)
    else:
        raise ValueError(f"Unknown method: {method}")

    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Volume management
# ---------------------------------------------------------------------------


def _find_lens_volume(runpod_mod) -> dict | None:
    """Find existing lens-data volume via GraphQL."""
    try:
        # Use runpod's built-in method to get pods — volumes are returned via GraphQL
        import requests  # noqa: PLC0415

        api_key = _get_api_key()
        query = """
        query {
            myself {
                networkVolumes {
                    id
                    name
                    size
                    dataCenterId
                }
            }
        }
        """
        resp = requests.post(
            "https://api.runpod.io/graphql",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"query": query},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()
        volumes = result.get("data", {}).get("myself", {}).get("networkVolumes", [])

        for vol in volumes:
            if vol.get("name") == VOLUME_NAME:
                return vol

        # Also check known shared infra volumes
        for vol in volumes:
            vid = vol.get("id", "")
            if vid in KNOWN_VOLUMES.values():
                log.info("Found shared infra volume: %s (%s)", vol.get("name"), vid)
                return vol

    except Exception as e:
        log.warning("Failed to list volumes: %s", e)

    return None


def _get_gpu_availability() -> list[dict]:
    """Query RunPod for GPU availability by datacenter."""
    try:
        import requests  # noqa: PLC0415

        api_key = _get_api_key()
        query = """
        query GpuTypes {
            gpuTypes {
                id
                displayName
                communityCloud
                communityPrice
            }
        }
        """
        resp = requests.post(
            "https://api.runpod.io/graphql",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"query": query},
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json().get("data", {}).get("gpuTypes", [])
    except Exception as e:
        log.warning("Failed to query GPU availability: %s", e)
        return []


def _create_volume(datacenter_id: str) -> str:
    """Create a new network volume. Returns volume ID."""
    import requests  # noqa: PLC0415

    api_key = _get_api_key()
    query = """
    mutation CreateVolume($input: CreateNetworkVolumeInput!) {
        createNetworkVolume(input: $input) {
            id
            name
            size
            dataCenterId
        }
    }
    """
    variables = {
        "input": {
            "name": VOLUME_NAME,
            "size": VOLUME_SIZE_GB,
            "dataCenterId": datacenter_id,
        }
    }
    resp = requests.post(
        "https://api.runpod.io/graphql",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"query": query, "variables": variables},
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()

    errors = result.get("errors")
    if errors:
        log.error("Volume creation failed: %s", errors)
        sys.exit(1)

    vol = result["data"]["createNetworkVolume"]
    log.info("Created volume: %s (%s) in %s", vol["name"], vol["id"], vol["dataCenterId"])
    return vol["id"]


def cmd_create_volume(args):
    """Find or create lens-data volume."""
    runpod_mod = _init_runpod()
    vol = _find_lens_volume(runpod_mod)

    if vol:
        log.info("Volume already exists: %s (%s) in %s", vol["name"], vol["id"], vol.get("dataCenterId", "?"))
        return

    # Pick a datacenter — prefer ones with known volumes
    datacenter_id = args.datacenter or "US-TX-3"
    log.info("Creating volume '%s' (%dGB) in %s", VOLUME_NAME, VOLUME_SIZE_GB, datacenter_id)

    if args.dry_run:
        log.info("[DRY RUN] Would create volume")
        return

    _create_volume(datacenter_id)


# ---------------------------------------------------------------------------
# Pod creation
# ---------------------------------------------------------------------------


def _create_pod(
    name: str,
    gpu_type: str,
    docker_image: str,
    env_vars: dict[str, str],
    volume_id: str | None = None,
    docker_start_cmd: str | None = None,
    gpu_count: int = 1,
    container_disk_gb: int = CONTAINER_DISK_GB,
    volume_mount_path: str = "/runpod-volume",
    ports: str = "22/tcp",
    dry_run: bool = False,
) -> str | None:
    """Create a RunPod GPU pod. Returns pod ID."""
    import requests  # noqa: PLC0415

    api_key = _get_api_key()

    # Build env list for GraphQL
    env_list = {k: v for k, v in env_vars.items()}

    query = """
    mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
        podFindAndDeployOnDemand(input: $input) {
            id
            name
            desiredStatus
            imageName
            machine {
                gpuDisplayName
            }
        }
    }
    """

    pod_input: dict = {
        "name": name,
        "imageName": docker_image,
        "gpuTypeId": gpu_type,
        "gpuCount": gpu_count,
        "cloudType": CLOUD_TYPE,
        "containerDiskInGb": container_disk_gb,
        "ports": ports,
        "env": [{"key": k, "value": v} for k, v in env_list.items()],
    }

    if volume_id:
        pod_input["networkVolumeId"] = volume_id
        pod_input["volumeMountPath"] = volume_mount_path

    if docker_start_cmd:
        pod_input["dockerArgs"] = docker_start_cmd

    if dry_run:
        log.info("[DRY RUN] Would create pod: %s", name)
        log.info("  GPU: %s, Image: %s", gpu_type, docker_image)
        log.info("  Env: %s", json.dumps(env_list, indent=2))
        if docker_start_cmd:
            log.info("  Cmd: %s", docker_start_cmd)
        return None

    log.info("Creating pod: %s (GPU: %s)", name, gpu_type)

    resp = requests.post(
        "https://api.runpod.io/graphql",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"query": query, "variables": {"input": pod_input}},
        timeout=60,
    )
    resp.raise_for_status()
    result = resp.json()

    errors = result.get("errors")
    if errors:
        log.error("Pod creation failed for %s: %s", name, errors)
        return None

    pod = result["data"]["podFindAndDeployOnDemand"]
    log.info("Created pod: %s (ID: %s)", pod["name"], pod["id"])
    return pod["id"]


def _detect_embed_url() -> str:
    """Auto-detect embed URL from running lens-embed pod."""
    pods = _list_lens_pods()
    for pod in pods:
        if pod.get("name") == "lens-embed":
            runtime = pod.get("runtime") or {}
            ports = runtime.get("ports") or []
            for port in ports:
                if port.get("privatePort") == 11434 and port.get("isIpPublic"):
                    url = f"http://{port['ip']}:{port['publicPort']}/v1"
                    log.info("Auto-detected embed URL: %s", url)
                    return url
            log.warning("lens-embed pod found but no public Ollama port available (still booting?)")
            return ""
    return ""


def _build_jobs(adapters: list[str], scopes: list[str], budgets: list[str]) -> list[dict]:
    """Build job list for a worker group."""
    jobs = []
    for adapter in adapters:
        for scope in scopes:
            for budget in budgets:
                jobs.append({"adapter": adapter, "scope": scope, "budget": budget})
    return jobs


def cmd_create_embed(args):
    """Create the shared embedding pod.

    The embed pod does NOT need a network volume — the model is pre-baked into the
    Docker image.  This lets RunPod place it in any region with available GPUs.
    """
    _init_runpod()

    # Override GPU list if user specified one
    gpu_candidates = [args.embed_gpu] if args.embed_gpu else list(EMBED_GPU_TYPES)

    ssh_pubkey = _get_ssh_pubkey()
    env_vars = {
        "OLLAMA_HOST": "0.0.0.0:11434",
        "PUBLIC_KEY": ssh_pubkey,
    }

    pod_id = None
    for gpu_type in gpu_candidates:
        pod_id = _create_pod(
            name="lens-embed",
            gpu_type=gpu_type,
            docker_image=EMBED_DOCKER_IMAGE,
            env_vars=env_vars,
            volume_id=None,  # no volume needed — model pre-loaded in image
            container_disk_gb=20,
            ports="22/tcp,11434/tcp",
            dry_run=args.dry_run,
        )
        if pod_id or args.dry_run:
            break
        log.info("No %s available, trying next GPU type...", gpu_type)

    if pod_id:
        log.info("Embed pod created: %s", pod_id)
        log.info("Image: %s (model pre-loaded, SSH enabled)", EMBED_DOCKER_IMAGE)
        log.info("GPU: %s", gpu_type)
        log.info("Run 'pod_manager.py ssh-config' once ready, then 'ssh lens-embed'")
    elif not args.dry_run:
        log.error("Could not create embed pod — no GPUs available for any of: %s", gpu_candidates)


def cmd_submit_sweep(args):
    """Create worker pods for the sweep."""
    _init_runpod()

    vol = _find_lens_volume(_init_runpod())
    volume_id = vol["id"] if vol else None

    if not volume_id:
        log.warning("No volume found — AWQ model (~18GB) will download on each pod boot")

    repo_url = args.repo_url or os.environ.get("LENS_REPO_URL", "")
    if not repo_url and not args.dry_run:
        # Auto-detect from local git remote
        try:
            result = subprocess.run(
                ["git", "remote", "get-url", "origin"],
                capture_output=True, text=True, check=True,
            )
            repo_url = result.stdout.strip()
            log.info("Auto-detected repo URL: %s", repo_url)
        except Exception:
            log.error("LENS_REPO_URL not set, --repo-url not provided, and no git remote found")
            sys.exit(1)

    # Workers need push access — embed a GitHub token in the HTTPS URL
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    if not gh_token and not args.dry_run:
        try:
            result = subprocess.run(
                ["gh", "auth", "token"], capture_output=True, text=True, check=True,
            )
            gh_token = result.stdout.strip()
        except Exception:
            log.warning("No GITHUB_TOKEN or gh CLI — workers won't be able to push results")

    if gh_token and repo_url.startswith("https://github.com/"):
        # Inject token: https://github.com/... -> https://x-access-token:TOKEN@github.com/...
        repo_url = repo_url.replace(
            "https://github.com/",
            f"https://x-access-token:{gh_token}@github.com/",
        )
        log.info("Injected GitHub token into repo URL for push access")

    embed_url = args.embed_url or os.environ.get("EMBED_URL", "")
    if not embed_url and not args.dry_run:
        # Auto-detect from running lens-embed pod
        embed_url = _detect_embed_url()
        if not embed_url:
            log.error("No --embed-url provided and no running lens-embed pod found")
            sys.exit(1)

    # Determine scopes and budgets
    scopes = SCOPES
    budgets = BUDGETS
    if args.smoke_test:
        scopes = ["01"]
        budgets = ["standard"]

    # Filter groups
    groups = dict(WORKER_GROUPS)
    if args.group:
        if args.group not in groups:
            log.error("Unknown group: %s (available: %s)", args.group, list(groups.keys()))
            sys.exit(1)
        groups = {args.group: groups[args.group]}

    # Create git branches
    if not args.dry_run:
        for group_name in groups:
            branch = f"lens-sweep/{group_name}"
            try:
                subprocess.run(
                    ["git", "branch", branch, "HEAD"],
                    capture_output=True, text=True,
                )
                subprocess.run(
                    ["git", "push", "origin", branch],
                    capture_output=True, text=True,
                )
                log.info("Created branch: %s", branch)
            except Exception as e:
                log.warning("Branch creation: %s", e)

    # Create worker pods
    total_runs = 0
    created_pods = 0
    for group_name, group_def in groups.items():
        jobs = _build_jobs(group_def["adapters"], scopes, budgets)
        total_runs += len(jobs)
        branch = f"lens-sweep/{group_name}"

        env_vars = {
            "LENS_BRANCH": branch,
            "LENS_GROUP": group_name,
            "LENS_JOBS": json.dumps(jobs),
            "LENS_REPO_URL": repo_url,
            "LENS_CONTAINERS": group_def.get("containers", ""),
            "VLLM_API_KEY": "lens-benchmark",
            "VLLM_MODEL": "Qwen/Qwen3-32B-AWQ",
            "PUBLIC_KEY": _get_ssh_pubkey(),
        }
        if embed_url:
            env_vars["EMBED_URL"] = embed_url

        # Bootstrap: install deps, clone repo, then delegate to pod_setup.sh.
        # The vLLM image has Python but not git/curl, so install those first.
        start_cmd = (
            "bash -c '"
            "apt-get update -qq && apt-get install -y -qq git openssh-client curl && "
            "curl -LsSf https://astral.sh/uv/install.sh | sh && "
            "export PATH=$HOME/.local/bin:$PATH && "
            "git clone --branch ${LENS_BRANCH} --depth 1 ${LENS_REPO_URL} /workspace/lens-benchmark && "
            "cd /workspace/lens-benchmark && "
            "bash scripts/pod_setup.sh"
            "'"
        )

        pod_name = f"lens-worker-{group_name}"
        gpu_candidates = [args.worker_gpu] if args.worker_gpu else list(WORKER_GPU_TYPES)

        pod_id = None
        for gpu_type in gpu_candidates:
            # Try with volume first, then without
            for vid in ([volume_id, None] if volume_id else [None]):
                pod_id = _create_pod(
                    name=pod_name,
                    gpu_type=gpu_type,
                    docker_image=DOCKER_IMAGE,
                    env_vars=env_vars,
                    volume_id=vid,
                    docker_start_cmd=start_cmd,
                    dry_run=args.dry_run,
                )
                if pod_id or args.dry_run:
                    if vid is None and volume_id:
                        log.info("  (created without volume — model will download)")
                    break
            if pod_id or args.dry_run:
                break
            log.info("No %s available, trying next GPU type...", gpu_type)

        if not pod_id and not args.dry_run:
            log.error("FAILED to create %s — no GPUs available", pod_name)
            created_pods -= 1

        created_pods += 1

    log.info("=== Sweep submitted: %d runs across %d pods (%d created) ===",
             total_runs, len(groups), created_pods)
    if args.smoke_test:
        log.info("(smoke test: scope 01, standard budget only)")


# ---------------------------------------------------------------------------
# Pod management
# ---------------------------------------------------------------------------


def _list_lens_pods() -> list[dict]:
    """List all lens-* pods."""
    import requests  # noqa: PLC0415

    api_key = _get_api_key()
    query = """
    query {
        myself {
            pods {
                id
                name
                desiredStatus
                runtime {
                    uptimeInSeconds
                    ports {
                        ip
                        isIpPublic
                        privatePort
                        publicPort
                        type
                    }
                }
                machine {
                    gpuDisplayName
                }
            }
        }
    }
    """
    resp = requests.post(
        "https://api.runpod.io/graphql",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"query": query},
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()
    pods = result.get("data", {}).get("myself", {}).get("pods", [])
    return [p for p in pods if p.get("name", "").startswith(POD_PREFIX)]


def _get_ssh_info(pod: dict) -> tuple[str, int] | None:
    """Extract SSH host and port from pod runtime ports."""
    runtime = pod.get("runtime") or {}
    ports = runtime.get("ports") or []
    for port in ports:
        if port.get("privatePort") == 22 and port.get("isIpPublic"):
            return port.get("ip", ""), port.get("publicPort", 0)
    return None


def cmd_list_pods(args):
    """List all lens-* pods."""
    _init_runpod()
    pods = _list_lens_pods()

    if not pods:
        log.info("No lens-* pods found")
        return

    log.info("%-25s %-12s %-8s %-25s %s", "NAME", "STATUS", "UPTIME", "SSH", "GPU")
    for pod in pods:
        name = pod.get("name", "?")
        status = pod.get("desiredStatus", "?")
        runtime = pod.get("runtime") or {}
        uptime = runtime.get("uptimeInSeconds", 0)
        uptime_str = f"{uptime // 60}m" if uptime else "—"
        gpu = (pod.get("machine") or {}).get("gpuDisplayName", "?")
        ssh_info = _get_ssh_info(pod)
        ssh_str = f"{ssh_info[0]}:{ssh_info[1]}" if ssh_info else "—"
        log.info("%-25s %-12s %-8s %-25s %s", name, status, uptime_str, ssh_str, gpu)


def cmd_ssh_config(args):
    """Update ~/.ssh/config with current RunPod pod SSH ports.

    Reads running lens-* pods, finds their public SSH ports, and writes
    Host entries to ~/.ssh/config. Existing lens-* entries are replaced;
    new ones are appended.
    """
    import re  # noqa: PLC0415

    _init_runpod()
    pods = _list_lens_pods()

    if not pods:
        log.info("No lens-* pods found")
        return

    ssh_config_path = Path.home() / ".ssh" / "config"
    if not ssh_config_path.exists():
        log.error("~/.ssh/config not found — creating it")
        ssh_config_path.parent.mkdir(mode=0o700, exist_ok=True)
        ssh_config_path.write_text("")

    config_text = ssh_config_path.read_text()

    # Remove any existing lens-* Host blocks (commented or uncommented)
    # Match from "Host lens-..." to the next "Host " or end of file
    config_text = re.sub(
        r"(?m)^#?\s*Host lens-[^\n]*\n(?:#?\s+[^\n]*\n)*",
        "",
        config_text,
    )
    # Also remove the RunPod template comment block if present
    config_text = re.sub(
        r"# RunPod pods —[^\n]*\n(?:#[^\n]*\n)*",
        "",
        config_text,
    )
    # Clean up trailing whitespace
    config_text = config_text.rstrip() + "\n"

    # Build new Host blocks for all running pods with SSH
    new_blocks = []
    for pod in pods:
        name = pod.get("name", "")
        if not name:
            continue

        ssh_info = _get_ssh_info(pod)
        if not ssh_info:
            log.info("%-25s no SSH port (pod not ready?)", name)
            continue

        ssh_host, ssh_port = ssh_info
        block = (
            f"\nHost {name}\n"
            f"    HostName {ssh_host}\n"
            f"    Port {ssh_port}\n"
            f"    User root\n"
            f"    IdentityFile ~/.ssh/id_ed25519\n"
            f"    StrictHostKeyChecking no\n"
            f"    UserKnownHostsFile /dev/null\n"
            f"    LogLevel ERROR\n"
        )
        new_blocks.append(block)
        log.info("%-25s → ssh %s (%s:%s)", name, name, ssh_host, ssh_port)

    if not new_blocks:
        log.info("No pods with SSH ports found")
        return

    config_text += "\n# RunPod pods (managed by pod_manager.py ssh-config)\n"
    config_text += "".join(new_blocks)

    ssh_config_path.write_text(config_text)
    log.info("Wrote %d entries to %s", len(new_blocks), ssh_config_path)


def cmd_status(args):
    """Check sweep progress by counting commits on sweep branches."""
    log.info("=== Sweep Status ===")

    for group_name in WORKER_GROUPS:
        branch = f"lens-sweep/{group_name}"
        try:
            result = subprocess.run(
                ["git", "log", f"main..origin/{branch}", "--oneline"],
                capture_output=True, text=True,
            )
            if result.returncode == 0:
                commits = [line for line in result.stdout.strip().split("\n") if line]
                log.info("%-20s %d commits", group_name, len(commits))
                for c in commits[-5:]:  # Show last 5
                    log.info("  %s", c)
            else:
                log.info("%-20s (no branch)", group_name)
        except Exception as e:
            log.warning("%-20s error: %s", group_name, e)


def cmd_logs(args):
    """Fetch pod logs for a group."""
    _init_runpod()
    pod_name = f"lens-worker-{args.group}"
    pods = _list_lens_pods()
    pod = next((p for p in pods if p.get("name") == pod_name), None)

    if not pod:
        log.error("Pod not found: %s", pod_name)
        return

    import requests  # noqa: PLC0415

    api_key = _get_api_key()
    query = """
    query PodLogs($podId: String!) {
        pod(input: { podId: $podId }) {
            id
            name
            runtime {
                container {
                    logs
                }
            }
        }
    }
    """
    resp = requests.post(
        "https://api.runpod.io/graphql",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"query": query, "variables": {"podId": pod["id"]}},
        timeout=30,
    )
    resp.raise_for_status()
    result = resp.json()
    pod_data = result.get("data", {}).get("pod", {})
    runtime = pod_data.get("runtime") or {}
    container = runtime.get("container") or {}
    logs = container.get("logs", "No logs available")
    print(logs)


def cmd_destroy_pods(args):
    """Terminate lens-* pods (volume persists).

    Default: destroys ALL lens-* pods including embed.
    Use --keep-embed to preserve the embedding pod between sweeps.
    Use --group to destroy only a specific worker group.
    """
    _init_runpod()
    pods = _list_lens_pods()

    if args.group:
        target = f"lens-worker-{args.group}"
        pods = [p for p in pods if p.get("name") == target]
    elif args.keep_embed:
        pods = [p for p in pods if p.get("name") != "lens-embed"]

    if not pods:
        log.info("No pods to destroy")
        return

    import requests  # noqa: PLC0415

    api_key = _get_api_key()
    names = [p.get("name", "?") for p in pods]
    log.info("Terminating %d pods: %s", len(pods), ", ".join(names))

    for pod in pods:
        pod_name = pod.get("name", "?")
        pod_id = pod["id"]

        if args.dry_run:
            log.info("[DRY RUN] Would terminate: %s (%s)", pod_name, pod_id)
            continue

        log.info("Terminating: %s (%s)", pod_name, pod_id)
        query = """
        mutation TerminatePod($input: PodTerminateInput!) {
            podTerminate(input: $input)
        }
        """
        try:
            resp = requests.post(
                "https://api.runpod.io/graphql",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"query": query, "variables": {"input": {"podId": pod_id}}},
                timeout=30,
            )
            resp.raise_for_status()
            log.info("  Terminated: %s", pod_name)
        except Exception as e:
            log.error("  Failed to terminate %s: %s", pod_name, e)

    log.info("Volume '%s' preserved for future sweeps", VOLUME_NAME)


def cmd_merge_results(args):
    """Merge all lens-sweep/* branches into main."""
    log.info("=== Merging sweep results ===")

    # Fetch all remote branches
    subprocess.run(["git", "fetch", "--all"], capture_output=True, text=True)

    merged = 0
    for group_name in WORKER_GROUPS:
        branch = f"lens-sweep/{group_name}"
        remote_branch = f"origin/{branch}"

        # Check if remote branch exists
        result = subprocess.run(
            ["git", "rev-parse", "--verify", remote_branch],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.info("SKIP %s (no remote branch)", group_name)
            continue

        # Check for new commits
        result = subprocess.run(
            ["git", "log", f"main..{remote_branch}", "--oneline"],
            capture_output=True, text=True,
        )
        commits = [line for line in result.stdout.strip().split("\n") if line]
        if not commits:
            log.info("SKIP %s (no new commits)", group_name)
            continue

        log.info("MERGE %s (%d commits)", group_name, len(commits))

        if args.dry_run:
            log.info("[DRY RUN] Would merge %s", branch)
            continue

        result = subprocess.run(
            ["git", "merge", remote_branch, "--no-edit", "-m", f"Merge sweep results: {group_name}"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            log.error("Merge failed for %s: %s", group_name, result.stderr)
        else:
            merged += 1
            log.info("  Merged: %s", group_name)

    log.info("Merged %d branches", merged)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="LENS sweep pod manager")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    sub = parser.add_subparsers(dest="command", required=True)

    # create-volume
    p_vol = sub.add_parser("create-volume", help="Find or create lens-data volume")
    p_vol.add_argument("--datacenter", default=None, help="Datacenter ID (default: US-TX-3)")
    p_vol.set_defaults(func=cmd_create_volume)

    # create-embed
    p_embed = sub.add_parser("create-embed", help="Create shared embedding pod")
    p_embed.add_argument("--embed-gpu", default=None, help="GPU type for embed pod (default: NVIDIA L4)")
    p_embed.set_defaults(func=cmd_create_embed)

    # submit-sweep
    p_sweep = sub.add_parser("submit-sweep", help="Submit sweep jobs to worker pods")
    p_sweep.add_argument("--smoke-test", action="store_true", help="Only scope 01, standard budget")
    p_sweep.add_argument("--group", default=None, help="Only this worker group")
    p_sweep.add_argument("--repo-url", default=None, help="Git repo URL")
    p_sweep.add_argument("--embed-url", default=None, help="Embedding server URL")
    p_sweep.add_argument("--worker-gpu", default=None, help="Force specific GPU type for workers")
    p_sweep.set_defaults(func=cmd_submit_sweep)

    # list-pods
    p_list = sub.add_parser("list-pods", help="List all lens-* pods")
    p_list.set_defaults(func=cmd_list_pods)

    # ssh-config
    p_ssh = sub.add_parser("ssh-config", help="Update ~/.ssh/config with pod SSH ports")
    p_ssh.set_defaults(func=cmd_ssh_config)

    # status
    p_status = sub.add_parser("status", help="Check sweep progress")
    p_status.set_defaults(func=cmd_status)

    # logs
    p_logs = sub.add_parser("logs", help="Fetch pod logs")
    p_logs.add_argument("group", help="Worker group name")
    p_logs.set_defaults(func=cmd_logs)

    # destroy-pods
    p_destroy = sub.add_parser("destroy-pods", help="Terminate all lens-* pods (including embed)")
    p_destroy.add_argument("--group", default=None, help="Only this worker group")
    p_destroy.add_argument("--keep-embed", action="store_true", help="Keep the embed pod running")
    p_destroy.set_defaults(func=cmd_destroy_pods)

    # merge-results
    p_merge = sub.add_parser("merge-results", help="Merge sweep branches into main")
    p_merge.set_defaults(func=cmd_merge_results)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
