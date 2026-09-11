"""Prove one real provider end to end, through a real CacheLLM proxy.

Unit tests talk to a simulated upstream. This talks to the actual service, with
your key, using the official OpenAI SDK the way an application would. It
starts its own proxy on a free port, runs ten requests that each check one
behaviour, prints what the cache did, and writes the evidence to results/live/.

    GROQ_API_KEY=gsk_... uv run python bench/live_check.py --var GROQ_API_KEY \\
        --model openai/gpt-oss-20b --name groq

The key goes into the proxy's environment and nowhere else. The script never
prints it, and it fails the run if the key shows up in the proxy's log.
Detection picks the host from the variable name, exactly as it would for you.
A full run costs a fraction of a cent. Leave out --var for a local host such as
Ollama, which needs no key and is found because it is running.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
from openai import APIError, APIStatusError, OpenAI

ROOT = Path(__file__).resolve().parents[1]
FINLAND = "What is the capital of Finland?"
FINLAND_REWORDED = "Which city is the capital of Finland?"
SWEDEN = "What is the capital of Sweden?"
HTTP_404 = "In one sentence, what does the HTTP status code 404 mean?"
NO_SUCH_MODEL = "no-such-model-cachellm"


def read_key(var: str | None, env_file: str | None) -> str:
    if not var:
        return ""
    if not env_file:
        if value := os.environ.get(var, "").strip():
            return value
        sys.exit(f"{var} is not set. Export it, or pass --env-file.")
    for raw in Path(env_file).read_text().splitlines():
        line = raw.strip().removeprefix("export ").strip()
        name, sep, value = line.partition("=")
        if not sep or name.strip() != var:
            continue
        value = value.strip()
        if value[:1] in "\"'":
            end = value.find(value[0], 1)
            value = value[1:end] if end > 0 else value[1:]
        else:
            value = value.split(" #")[0].strip()
        if value:
            return value
    sys.exit(f"{var} is missing or empty in {env_file}")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def ask(client: OpenAI, model: str, text: str, stream: bool = False, **kw: Any) -> dict[str, Any]:
    """One request, with the cache's verdict read from the response headers."""
    started = time.perf_counter()
    row: dict[str, Any] = {"model": model, "question": text, "stream": stream}
    messages: Any = [{"role": "user", "content": text}]
    try:
        if stream:
            raw_stream = client.chat.completions.with_raw_response.create(
                model=model, messages=messages, stream=True, **kw
            )
            parts: list[str] = []
            finish = None
            tokens = 0
            for chunk in raw_stream.parse():
                for choice in chunk.choices:
                    parts.append(choice.delta.content or "")
                    finish = choice.finish_reason or finish
                if chunk.usage is not None:
                    tokens = chunk.usage.total_tokens
            answer = "".join(parts)
            h, status = raw_stream.headers, raw_stream.http_response.status_code
        else:
            raw_once = client.chat.completions.with_raw_response.create(
                model=model, messages=messages, stream=False, **kw
            )
            parsed = raw_once.parse()
            answer = parsed.choices[0].message.content or ""
            finish = parsed.choices[0].finish_reason
            tokens = parsed.usage.total_tokens if parsed.usage else 0
            h, status = raw_once.headers, raw_once.http_response.status_code
        row.update(
            status=status,
            cache=h.get("x-cache", "-"),
            upstream=h.get("x-cache-upstream", "-"),
            tier=h.get("x-cache-tier", "-"),
            similarity=h.get("x-cache-similarity", "-"),
            finish=finish,
            answer=answer,
            tokens=tokens,
        )
    except APIStatusError as exc:
        row.update(
            status=exc.status_code,
            cache="-",
            tier="-",
            similarity="-",
            finish="error",
            answer=str(exc.message),
            tokens=0,
            upstream="-",
        )
    except APIError as exc:  # an error event in the middle of a stream
        row.update(
            status=200,
            cache="-",
            tier="-",
            similarity="-",
            finish="error",
            answer=str(exc.message),
            tokens=0,
            upstream="-",
        )
    row["ms"] = round((time.perf_counter() - started) * 1000, 1)
    return row


def run(args: argparse.Namespace) -> int:
    key = read_key(args.var, args.env_file)
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    out_dir = ROOT / "results" / "live"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"{args.name}.log"

    # A clean environment, so the only provider the proxy can find is this one.
    env = {
        name: os.environ[name]
        for name in ("PATH", "HOME", "TMPDIR", "FASTEMBED_CACHE_PATH")
        if name in os.environ
    }
    env.update({"CACHELLM_BACKEND": "memory", "NO_COLOR": "1", "LANG": "en_US.UTF-8"})
    if args.var:
        env[args.var] = key
    with open(log_path, "w") as log:
        proxy = subprocess.Popen(
            [sys.executable, "-m", "cachellm", "serve", "--port", str(port)],
            cwd=out_dir,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
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
            return 1

        sdk = OpenAI(base_url=f"{url}/v1", api_key="not-needed", max_retries=0, timeout=180)
        m = args.model
        r = {
            "first": ask(sdk, m, FINLAND, temperature=0),
            "repeat": ask(sdk, m, FINLAND, temperature=0),
            "reworded": ask(sdk, m, FINLAND_REWORDED, temperature=0),
            "other": ask(sdk, m, SWEDEN, temperature=0),
            "stream": ask(sdk, m, HTTP_404, stream=True, temperature=0),
            "stream_again": ask(sdk, m, HTTP_404, stream=True, temperature=0),
            "hot": ask(sdk, m, FINLAND, temperature=1.0),
            "bad_model": ask(sdk, NO_SUCH_MODEL, FINLAND, temperature=0),
            "bad_model_stream": ask(sdk, NO_SUCH_MODEL, FINLAND, stream=True, temperature=0),
        }
        for n, extra in enumerate(args.also or []):
            r[f"also{n}"] = ask(sdk, extra, FINLAND, temperature=0)
            r[f"also{n}_again"] = ask(sdk, extra, FINLAND, temperature=0)

        records = httpx.get(f"{url}/admin/requests", params={"limit": 100}, timeout=5).json()
        items = records.get("requests", records) if isinstance(records, dict) else records
        routes = httpx.get(f"{url}/admin/providers", timeout=5).json().get("routes", [])
        local = any(route.get("default") and route.get("local") for route in routes)
        spent = sum(float(i.get("spent_usd") or 0) for i in items)
        saved = sum(float(i.get("saved_usd") or 0) for i in items)
        dashboard = subprocess.run(
            [sys.executable, "-m", "cachellm", "stats", "--url", url, "--limit", "14"],
            env={k: v for k, v in env.items() if k != args.var},
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
    detected = next(
        (
            json.loads(line).get("using")
            for line in log_text.splitlines()
            if "provider_autodetected" in line
        ),
        "nothing detected",
    )
    checks = [
        (
            "the first ask is a miss answered by the model",
            r["first"]["cache"] == "MISS"
            and r["first"]["status"] == 200
            and bool(r["first"]["answer"].strip()),
        ),
        (
            "the same question again is an exact hit",
            r["repeat"]["cache"] == "HIT" and r["repeat"]["tier"] == "exact",
        ),
        (
            "a reworded question is a meaning hit",
            r["reworded"]["cache"] == "HIT" and r["reworded"]["tier"] == "semantic",
        ),
        (
            "hits return the stored answer word for word",
            r["first"]["answer"] == r["repeat"]["answer"] == r["reworded"]["answer"],
        ),
        (
            "a different country is not served Finland's answer",
            r["other"]["cache"] == "MISS" and "helsinki" not in r["other"]["answer"].lower(),
        ),
        (
            "a stream is answered live on a miss",
            r["stream"]["cache"] == "MISS" and bool(r["stream"]["answer"].strip()),
        ),
        ("the same stream again is a hit", r["stream_again"]["cache"] == "HIT"),
        (
            "the replayed stream matches the original exactly",
            r["stream_again"]["answer"] == r["stream"]["answer"],
        ),
        ("a hot temperature bypasses the cache", r["hot"]["cache"] == "BYPASS"),
        (
            "an unknown model returns the host's own error status",
            r["bad_model"]["status"] in (400, 404),
        ),
        ("so does a stream to an unknown model", r["bad_model_stream"]["status"] in (400, 404)),
        (
            "token usage came back, streamed or not",
            r["first"]["tokens"] > 0 and r["stream"]["tokens"] > 0,
        ),
        (
            "a local model counts as free" if local else "spend on misses is counted",
            spent == 0 if local else spent > 0,
        ),
    ]
    if key:
        checks.append(("the API key never appears in the proxy log", key not in log_text))
    for n, extra in enumerate(args.also or []):
        checks.append(
            (
                f"{extra} answers and then hits",
                r[f"also{n}"]["status"] == 200 and r[f"also{n}_again"]["cache"] == "HIT",
            )
        )

    print(f"\n{args.name}: {m} via {detected}\n")
    print(f"{'request':<18}{'status':>6}  {'cache':<7}{'tier':<9}{'score':>6}  {'ms':>9}  answer")
    for label, row in r.items():
        text = " ".join(str(row["answer"]).split())[:64]
        print(
            f"{label:<18}{row['status']:>6}  {row['cache']:<7}{row['tier']:<9}"
            f"{str(row['similarity'])[:5]:>6}  {row['ms']:>9}  {text}"
        )
    print(f"\nspent ${spent:.6f} on misses, ${saved:.6f} modelled savings on hits\n")
    passed = all(ok for _, ok in checks)
    for label, ok in checks:
        print(("PASS  " if ok else "FAIL  ") + label)
    print("\n" + re.sub(r"\x1b\[[0-9;]*m", "", dashboard))

    (out_dir / f"{args.name}.json").write_text(
        json.dumps(
            {
                "name": args.name,
                "model": m,
                "detected": detected,
                "run_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "passed": passed,
                "checks": [{"check": label, "passed": bool(ok)} for label, ok in checks],
                "requests": {
                    label: {**row, "answer": str(row["answer"])[:300]} for label, row in r.items()
                },
                "spent_usd": round(spent, 8),
                "saved_usd": round(saved, 8),
            },
            indent=2,
        )
    )
    log_path.unlink(missing_ok=True)  # it did its job, and logs are not evidence to keep
    return 0 if passed else 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--var", help="Environment variable holding the key, e.g. GROQ_API_KEY. Omit for Ollama."
    )
    parser.add_argument("--model", required=True, help="Model to test, as your app would send it.")
    parser.add_argument("--name", required=True, help="Label for this run and its results file.")
    parser.add_argument("--also", action="append", help="Another model to check, repeatable.")
    parser.add_argument(
        "--env-file", help="Read the key from this .env file instead of the environment."
    )
    sys.exit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
