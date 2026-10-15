"""Separate subscription-authenticated Codex SDK experiment.

Install: python -m pip install -e '.[codex-smoke]'
Run:     python examples/codex_sdk_smoke.py
No .env loading, API-key fallback, Gemini calls, or application changes.
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from openai_codex import ApprovalMode, AsyncCodex, CodexConfig, Sandbox
from openai_codex.errors import CodexError
from pydantic import ValidationError

from structured_output_agent.exceptions import ResponseParseError, SchemaValidationError
from structured_output_agent.schemas import SupportTicket
from structured_output_agent.validation import ResponseValidator

ROOT = Path(__file__).resolve().parents[1]
PROMPT = (
    "Classify this synthetic customer support message: "
    "I was charged twice for my subscription. Please have a human help me. "
    "Return only the final JSON ticket classification."
)


async def run_smoke(model: str | None) -> int:
    scratch = ROOT / "work" / "codex-sdk-smoke"
    scratch.mkdir(parents=True, exist_ok=True)
    config = CodexConfig(
        cwd=str(scratch),
        # Override ambient key values; never load the application's .env.
        env={"OPENAI_API_KEY": "", "CODEX_API_KEY": ""},
        config_overrides=('model_provider="openai"', 'web_search="disabled"'),
        client_name="structured_output_sdk_smoke",
        client_title="Structured Output SDK Smoke Test",
    )
    print("Starting the local Codex SDK runtime…", flush=True)
    async with AsyncCodex(config=config) as codex:
        state = await codex.account()
        account = state.account.root if state.account is not None else None
        if account is None or account.type != "chatgpt":
            print("No ChatGPT-authenticated account was found. No inference request was sent.")
            print("Run codex login and choose Sign in with ChatGPT, then retry this script.")
            return 2
        print("Authentication confirmed: ChatGPT (no API-key fallback).", flush=True)
        thread = await codex.thread_start(
            cwd=str(scratch), model=model, model_provider="openai", ephemeral=True,
            sandbox=Sandbox.read_only, approval_mode=ApprovalMode.deny_all,
            base_instructions=(
                "You classify support messages into JSON. Do not use tools, inspect "
                "files, execute commands, browse, or modify anything. Answer using "
                "only the supplied support message and output schema."
            ),
        )
        print("Requesting one classification turn with a JSON output schema…", flush=True)
        result = await thread.run(PROMPT, output_schema=SupportTicket.model_json_schema())
        status = getattr(result.status, "value", result.status)
        print(f"Codex turn status: {status}", flush=True)
        if status != "completed" or not result.final_response:
            print("No completed model response was received. Raw runtime errors are not printed.")
            return 3
        ticket = ResponseValidator().validate(result.final_response, SupportTicket)
        print("PASS: live Codex output passed strict local Pydantic validation.")
        print(ticket.model_dump_json(indent=2))
        return 0


async def bounded_run(model: str | None, timeout_seconds: float) -> int:
    try:
        return await asyncio.wait_for(run_smoke(model), timeout=timeout_seconds)
    except TimeoutError:
        print("The SDK smoke test timed out. No automatic second turn was requested.")
        return 4
    except (ResponseParseError, SchemaValidationError, ValidationError):
        print("The model response did not pass local validation.")
        return 5
    except CodexError as error:
        if "workspace routing discovery failed" in str(error).lower():
            print("Codex SDK could not discover workspace routing during the account check.")
            print("No model request was sent; subscription inference is not verified.")
            print("Check Codex connectivity/account/workspace access, then retry. The cached CLI login status alone does not verify inference access.")
        else:
            print(f"Codex SDK failed ({type(error).__name__}). Raw errors and credentials are not printed.")
        return 6
    except OSError as error:
        print(f"Local runtime could not start ({type(error).__name__}). Check runtime installation and filesystem permissions.")
        return 7


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Optional model available to your Codex account; default uses Codex configuration.")
    parser.add_argument("--timeout", type=float, default=90, help="Overall timeout in seconds (default 90).")
    args = parser.parse_args()
    if not 0 < args.timeout <= 300:
        parser.error("--timeout must be greater than zero and at most 300")
    raise SystemExit(asyncio.run(bounded_run(args.model, args.timeout)))


if __name__ == "__main__":
    main()
