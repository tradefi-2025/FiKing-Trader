#!/usr/bin/env python3
"""Launcher for FiKing-Trader services.

Starts the Flask API server, configured workers, and optional extra commands
(e.g., databases or other microservices) defined via environment variables.
"""

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_WORKERS_FILE = PROJECT_ROOT / "src" / "services" / "workers.json"


def _load_workers(workers_file: Path) -> Dict[str, str]:
    if not workers_file.exists():
        return {}
    with workers_file.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        return {}
    return {str(k): str(v) for k, v in data.items()}


def _module_for_worker(name: str) -> Optional[str]:
    module_path = PROJECT_ROOT / "src" / "services" / name / "worker.py"
    if module_path.exists():
        return f"src.services.{name}.worker"
    return None


def _start_process(cmd: List[str], env: Dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=env)


def _parse_extra_cmds(raw: str) -> List[List[str]]:
    if not raw:
        return []
    try:
        items = json.loads(raw)
    except json.JSONDecodeError:
        items = [raw]
    if not isinstance(items, list):
        items = [str(items)]
    commands: List[List[str]] = []
    for item in items:
        if not item:
            continue
        commands.append(shlex.split(str(item)))
    return commands


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FiKing-Trader services")
    parser.add_argument("--no-server", action="store_true", help="Skip Flask server")
    parser.add_argument("--no-workers", action="store_true", help="Skip workers")
    parser.add_argument(
        "--workers",
        default="",
        help="Comma-separated worker names to run (default: all in workers.json)",
    )
    parser.add_argument(
        "--workers-file",
        default=str(DEFAULT_WORKERS_FILE),
        help="Path to workers.json (default: src/services/workers.json)",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(
        os.pathsep
    )

    processes: List[subprocess.Popen] = []

    if not args.no_server:
        server_cmd = [sys.executable, "-m", "src.server.flask_server"]
        processes.append(_start_process(server_cmd, env))

    if not args.no_workers:
        workers_file = Path(args.workers_file)
        workers = _load_workers(workers_file)
        if args.workers:
            requested = [w.strip() for w in args.workers.split(",") if w.strip()]
        else:
            requested = list(workers.keys())

        for name in requested:
            module = _module_for_worker(name)
            if not module:
                print(f"[launcher] worker module not found: {name}")
                continue
            worker_cmd = [sys.executable, "-m", module]
            processes.append(_start_process(worker_cmd, env))

    extra_cmds = _parse_extra_cmds(os.getenv("LAUNCHER_EXTRA_CMDS", ""))
    for cmd in extra_cmds:
        if cmd:
            processes.append(_start_process(cmd, env))

    if not processes:
        print("[launcher] nothing to run")
        return 1

    def _shutdown(signum, _frame):
        print(f"[launcher] shutting down (signal {signum})")
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        time.sleep(1)
        for proc in processes:
            if proc.poll() is None:
                proc.kill()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while True:
            alive = [p for p in processes if p.poll() is None]
            if not alive:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown(signal.SIGINT, None)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
