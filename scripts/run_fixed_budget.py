#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, subprocess, time
from pathlib import Path

def main() -> None:
    p = argparse.ArgumentParser(description="Launch a reproducible fixed-budget HYDRA experiment.")
    p.add_argument("--name", required=True)
    p.add_argument("--output", default="runs/experiment.json")
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    if not args.command:
        p.error("a command is required after the options")
    start = time.time()
    proc = subprocess.run(args.command, text=True)
    payload = {
        "name": args.name,
        "command": args.command,
        "returncode": proc.returncode,
        "elapsed_seconds": time.time() - start,
        "status": "passed" if proc.returncode == 0 else "failed",
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(proc.returncode)

if __name__ == "__main__":
    main()
