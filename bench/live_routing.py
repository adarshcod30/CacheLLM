"""Prove that one proxy sends each model to its own host, live.

    GEMINI_API_KEY=... GROQ_API_KEY=... uv run python bench/live_routing.py \\
        --key GEMINI_API_KEY --key GROQ_API_KEY \\
        --expect gemini-2.5-flash=gemini --expect openai/gpt-oss-20b=groq \\
        --expect qwen2.5:0.5b=ollama

Every key passed with --key goes to one proxy, which must then send each model
to the host named after its `=`, keep a separate cache per host, and never
serve one host's answer for another. Keys come from the environment, or from a
.env file with --key NAME=path/to/.env. They are never printed, and the run
fails if one shows up in the proxy's log.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import signal
import subprocess
import sys
import time
from typing import Any

import httpx
from openai import OpenAI

from bench.live_check import FINLAND, ROOT, ask, free_port, read_key


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--key", action="append", default=[], help="VAR or VAR=path/to/.env")
    parser.add_argument("--expect", action="append", required=True, help="MODEL=HOST")
    parser.add_argument("--name", default="routing", help="Label for the results file.")
    args = parser.parse_args()

    keys: dict[str, str] = {}
    for spec in args.key:
        var, _, env_file = spec.partition("=")
        keys[var] = read_key(var, env_file or None)
    expected = [tuple(spec.rsplit("=", 1)) for spec in args.expect]

    out_dir = ROOT / "results" / "live"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{args.name}.log"
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    env = {n: os.environ[n] for n in ("PATH", "HOME", "TMPDIR") if n in os.environ}
    env.update(keys)
    env.update({"CACHELLM_BACKEND": "memory", "NO_COLOR": "1", "LANG": "en_US.UTF-8"})

    with open(log_path, "w") as log:
        proxy = subprocess.Popen(
            [sys.executable, "-m", "cachellm", "serve", "--port", str(port)],
            cwd=out_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    rows: list[dict[str, Any]] = []
    try:
        for _ in range(480):
            try:
                if httpx.get(f"{url}/readyz", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        else:
            print("the proxy never became ready, see", log_path)
            sys.exit(1)

        sdk = OpenAI(base_url=f"{url}/v1", api_key="not-needed", max_retries=0, timeout=180)
        providers = httpx.get(f"{url}/admin/providers", timeout=5).json()
        print(f"\none proxy on {url}, routing to:\n")
        for r in providers["routes"]:
            note = "  (default)" if r["default"] else ""
            listed = f"{r['models_listed']} models listed" if r["models_listed"] else "no list"
            print(f"  {r['name']:<34} {listed:<20} {r['reason']}{note}")

        for model, host in expected:
            first = ask(sdk, model, FINLAND, temperature=0)
            again = ask(sdk, model, FINLAND, temperature=0)
            why = httpx.get(f"{url}/admin/route/{model}", timeout=5).json()
            rows.append(
                {"model": model, "expected": host, "first": first, "again": again, "route": why}
            )
        missing = httpx.post(
            f"{url}/v1/chat/completions",
            json={
                "model": "mistral/mistral-large-latest",
                "messages": [{"role": "user", "content": FINLAND}],
            },
            timeout=30,
        )
        owners: dict[str, int] = {}
        for card in httpx.get(f"{url}/v1/models", timeout=10).json()["data"]:
            owners[card["owned_by"]] = owners.get(card["owned_by"], 0) + 1
        dashboard = subprocess.run(
            [sys.executable, "-m", "cachellm", "stats", "--url", url, "--limit", "12"],
            env={k: v for k, v in env.items() if k not in keys},
            capture_output=True,
            text=True,
            cwd=out_dir,
        ).stdout
    finally:
        proxy.send_signal(signal.SIGINT)
        try:
            proxy.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proxy.kill()

    log_text = log_path.read_text()
    print(f"\n{'model':<30}{'expected':<9}{'went to':<9}{'first':<7}{'again':<7}  why")
    for row in rows:
        print(
            f"{row['model']:<30}{row['expected']:<9}{row['first']['upstream']:<9}"
            f"{row['first']['cache']:<7}{row['again']['cache']:<7}  {row['route']['reason']}"
        )
    checks = [
        (
            f"{row['model']} is answered by {row['expected']}",
            row["first"]["status"] == 200 and row["first"]["upstream"] == row["expected"],
        )
        for row in rows
    ]
    checks += [
        (
            "each host missed once, so no host was served another's answer",
            all(row["first"]["cache"] == "MISS" for row in rows),
        ),
        (
            "each repeat hit the cache of the host that answered it",
            all(
                row["again"]["cache"] == "HIT" and row["again"]["upstream"] == row["expected"]
                for row in rows
            ),
        ),
        (
            "a prefix for a host with no key is refused with the fix",
            missing.status_code == 400 and "MISTRAL_API_KEY" in missing.text,
        ),
        (
            "the model list includes every expected host",
            all(owners.get(host, 0) > 0 for _, host in expected),
        ),
    ]
    checks += [
        (f"{var} never appears in the proxy log", value not in log_text)
        for var, value in keys.items()
    ]
    print(f"\nmodels on offer by host: {owners}\n")
    passed = all(ok for _, ok in checks)
    for label, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + label)
    print("\n" + re.sub(r"\x1b\[[0-9;]*m", "", dashboard))

    for row in rows:
        for part in ("first", "again"):
            row[part]["answer"] = str(row[part]["answer"])[:200]
    (out_dir / f"{args.name}.json").write_text(
        json.dumps(
            {
                "name": args.name,
                "run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "passed": passed,
                "routes": [
                    {k: r[k] for k in ("key", "name", "default", "models_listed", "local")}
                    for r in providers["routes"]
                ],
                "models_by_host": owners,
                "checks": [{"check": label, "passed": bool(ok)} for label, ok in checks],
                "requests": rows,
            },
            indent=2,
        )
    )
    log_path.unlink(missing_ok=True)
    sys.exit(0 if passed else 2)


if __name__ == "__main__":
    main()
