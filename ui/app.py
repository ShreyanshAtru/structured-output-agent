"""Streamlit presentation layer: python -m streamlit run ui/app.py."""
import json
from pathlib import Path
import streamlit as st

from structured_output_agent.playground import PlaygroundError, PlaygroundRunner, SCENARIOS, session_metrics
from structured_output_agent.schemas import SupportTicket

ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Structured Output Agent", page_icon="🧩", layout="wide")
st.title("Structured Output Agent")
st.caption("Test schema enforcement, inspect retries, and follow each run from model call to validated result.")
if "reports" not in st.session_state:
    st.session_state.reports = []

with st.sidebar:
    st.header("Run settings")
    mode_label = st.selectbox("Mode", ["Offline demo", "Live Codex", "Live Gemini"], key="mode")
    mode = "fake" if mode_label == "Offline demo" else "live"
    provider = "codex" if mode_label == "Live Codex" else "gemini"
    scenario = st.selectbox("Scenario", SCENARIOS, index=1, key="scenario", disabled=mode == "live")
    if provider == "codex":
        model = st.text_input("Codex model (optional)", value="", key="codex_model", help="Leave blank to use your configured Codex default.")
        st.caption("Uses your local ChatGPT sign-in. Run codex login in the project virtual environment if needed.")
    else:
        model = st.text_input("Gemini model", value="gemini-3.8-flash", disabled=mode == "fake", key="model")
    max_attempts = st.slider("Maximum validation attempts", 1, 10, 3, key="max_attempts")
    max_tools = st.slider("Maximum tool calls", 0, 20, 5, key="max_tools")
    enable_tools = st.checkbox("Enable ticket context tool", value=True, key="enable_tools")
    backend_label = st.selectbox("Observability", ["Local JSONL", "Logfire"], key="backend")
    backend = "local" if backend_label == "Local JSONL" else "logfire"
    capture_raw = st.checkbox("Keep raw responses in this session", value=mode == "fake", key=f"capture_{mode}")
    st.caption("Raw responses stay in session memory and are excluded from telemetry. Live capture is off by default.")
    if backend == "logfire":
        st.caption("Sends sanitized telemetry to the Logfire project configured in .env.")
    else:
        st.caption("Appends sanitized events to the local log path configured in .env.")
    if st.button("Clear session history", key="clear_history"):
        st.session_state.reports = []
        st.session_state.pop("selected_run", None)
        st.rerun()

st.subheader("Support ticket playground")
if mode == "fake":
    st.info("Offline responses are scripted to demonstrate specific failures. Editing the message does not change their content. No LLM API calls are made.")
elif provider == "codex":
    st.info("Live Codex sends your message and validated tool context through your ChatGPT-authenticated Codex SDK. Subscription limits apply. Runs start only when you click Run agent.")
else:
    st.info("Live runs send your message and tool context to Gemini using the key in .env. API usage may incur charges. A request is sent only when you click Run agent.")

prompt = st.text_area("Support message", value="I was charged twice. Look up TICKET-001 and help me resolve the duplicate charge.", height=110, max_chars=10000, key="prompt")
custom_json = "{}"
if mode == "fake" and scenario == "Custom JSON → recovery":
    custom_json = st.text_area("First model response (JSON to test)", value='{"priority":"urgent"}', height=140, max_chars=50000, key="custom_json")
    st.caption("If invalid and budget remains, the next scripted response is a valid ticket.")
with st.expander("Output contract · SupportTicket"):
    st.json(SupportTicket.model_json_schema(), expanded=False)
st.caption("The validation budget includes the first final response. Tool continuations have a separate run-wide limit.")

if st.button("Run agent", type="primary", key="run"):
    report = None
    try:
        with st.spinner("Running agent and collecting diagnostics…"):
            report = PlaygroundRunner(ROOT).run(
                prompt=prompt, mode=mode, scenario=scenario, max_attempts=max_attempts,
                max_tool_calls=max_tools, backend=backend, model=model,
                capture_raw=capture_raw, custom_json=custom_json, enable_tools=enable_tools, provider=provider,
            )
    except PlaygroundError as error:
        st.error(str(error))
    except Exception:
        # Never render raw exceptions: SDK/config exceptions may contain secrets.
        st.error("The run could not complete because of an application or startup error. Check configuration and sanitized local logs.")
    if report is not None:
        st.session_state.reports = (st.session_state.reports + [report])[-10:]
        st.session_state.selected_run = report.result.run_id

reports = st.session_state.reports
if not reports:
    st.caption("Run an offline scenario to see validated output, failures, retries, and telemetry here.")
    st.stop()

st.divider()
by_id = {r.result.run_id: r for r in reports}
selected = st.selectbox("Inspect a completed run", list(by_id),
    format_func=lambda rid: f"{by_id[rid].scenario} · {by_id[rid].result.status} · {rid[:8]}", key="selected_run")
report = by_id[selected]
result = report.result
st.caption(f"Saved run: {'Offline demo' if report.mode == 'fake' else report.scenario} · {report.model} · {report.backend} · {result.run_id}")
status, attempts, tools, latency = st.columns(4)
status.metric("Status", result.status.title())
attempts.metric("Validation attempts", len(result.attempts))
tools.metric("Tool calls", result.tool_calls)
latency.metric("Latency", f"{result.duration_ms:.1f} ms")
if report.telemetry_failures:
    st.warning(f"Agent outcome is available, but telemetry reported {report.telemetry_failures} backend failures. Remote delivery is not guaranteed.")
if report.dropped_events:
    st.warning(f"The in-session event limit dropped {report.dropped_events} events.")

output_tab, validation_tab, responses_tab, telemetry_tab = st.tabs(["Result", "Validation & retries", "Model responses", "Observability"])
with output_tab:
    if result.status == "success":
        st.success("Response passed the SupportTicket schema.")
        st.json(result.data.model_dump())
    else:
        st.error(f"{result.error.code}: {result.error.message}")
        st.caption("Inspect the failed attempt and validation details in the next tab.")
    st.download_button("Download run result", result.model_dump_json(indent=2),
                       file_name=f"run-{result.run_id}.json", mime="application/json", key="download_result")

with validation_tab:
    retry_count = sum(e["event"] == "retry.scheduled" for e in report.events)
    st.write(f"{retry_count} schema-repair retries scheduled for this run.")
    for attempt in result.attempts:
        with st.expander(f"Attempt {attempt.attempt} · {attempt.status} · {attempt.duration_ms:.1f} ms", expanded=attempt.status == "failure"):
            if attempt.error:
                st.write(f"{attempt.error.code}: {attempt.error.message}")
            else:
                st.write("Final response passed validation.")
            issues = [{"Stage": e.get("stage", "final"), "Tool": e.get("tool", "—"), "Field": e.get("field", "<root>"), "Issue": e.get("issue_code", e.get("error_code"))}
                      for e in report.events if e["event"] == "validation.issue" and e.get("attempt") == attempt.attempt]
            parse_events = [e for e in report.events if e["event"] == "validation.failed" and e.get("attempt") == attempt.attempt and e.get("line") is not None]
            for event in parse_events:
                st.caption(f"JSON syntax error at line {event['line']}, column {event['column']}.")
            if issues:
                st.dataframe(issues, hide_index=True, width="stretch")
            tool_failures = [e for e in report.events if e["event"] == "validation.failed"
                             and e.get("attempt") == attempt.attempt and e.get("stage") in ("input", "output")]
            for event in tool_failures:
                st.write(f"Tool {event.get('tool', 'unknown')} failed {event['stage']} validation.")
            feedback = [c["repair_feedback"] for c in report.calls
                        if c["attempt"] == attempt.attempt and c["repair_feedback"]]
            if feedback:
                st.caption("Repair guidance sent for this attempt")
                st.code(feedback[0], language="text")
    st.caption("Issue codes are sanitized Pydantic diagnostics. The output contract above defines required fields, types, and allowed values.")

with responses_tab:
    st.caption("Model call order includes tool continuations; several calls can belong to one validation attempt.")
    for call in report.calls:
        with st.expander(f"Call {call['call']} · attempt {call['attempt']} · {call['kind']}"):
            if "raw" in call:
                st.code(call["raw"], language="json")
                if call.get("truncated"):
                    st.caption("Display truncated to 5,000 characters. Validation used the full response.")
            elif call["kind"] == "provider_error":
                st.write("The provider call failed; raw exception details are excluded.")
            else:
                st.write("Raw capture was disabled for this run.")

with telemetry_tab:
    st.write("Correlated event timeline")
    event_filter = st.selectbox("Timeline filter", ["All events", "Failures & retries", "Completed spans"], key="timeline_filter")
    events = list(report.events)
    if event_filter == "Failures & retries":
        events = [e for e in events if e.get("status") == "failure" or e["event"] in ("validation.failed", "validation.issue", "attempt.failed", "retry.scheduled")]
    elif event_filter == "Completed spans":
        events = [e for e in events if e["event"].endswith(".finished")]
    columns = ("timestamp", "event", "attempt", "stage", "tool", "status", "error_code", "field", "issue_code", "duration_ms")
    st.dataframe([{k: e.get(k) for k in columns} for e in events], hide_index=True, width="stretch")
    st.caption("Each event includes run/span correlation. Raw prompts, model responses, tool payloads, and credentials are excluded.")
    with st.expander("Full sanitized trace"):
        st.json(list(report.events), expanded=False)
    st.download_button("Download sanitized events", "\n".join(json.dumps(e) for e in report.events),
                       file_name=f"events-{result.run_id}.jsonl", mime="application/x-ndjson", key="download_events")
    if report.backend == "logfire":
        st.caption("The configured Logfire project receives the same events and native spans. Open its Live view and filter by this run ID. Delivery has not been confirmed by the UI.")
        st.link_button("Open Logfire", "https://logfire.pydantic.dev/")
    else:
        st.caption("The local backend also appended these events to the configured JSONL log.")

st.divider()
st.subheader("Session outcomes")
st.caption("Last 10 completed runs in this browser session. Offline, Codex, and Gemini results are calculated separately.")
metric_mode = st.selectbox("Metrics group", ["Offline demo", "Live Codex", "Live Gemini"], key="metrics_group")
metrics = session_metrics(reports, "fake" if metric_mode == "Offline demo" else "live",
                          provider=None if metric_mode == "Offline demo" else ("codex" if metric_mode == "Live Codex" else "gemini"))
def percentage(value):
    return "—" if value is None else f"{value:.0%}"
a,b,c,d = st.columns(4)
a.metric("Runs", metrics["runs"])
b.metric("First-attempt success", percentage(metrics["first_attempt_success_rate"]))
c.metric("Recovery rate", percentage(metrics["recovery_rate"]))
d.metric("Final failure rate", percentage(metrics["final_failure_rate"]))
st.caption("Recovery rate uses initially invalid runs as its denominator. A dash means there is no applicable sample.")
st.dataframe([{"Run": r.result.run_id[:8], "Mode": r.mode, "Provider": r.provider, "Scenario": r.scenario,
               "Status": r.result.status, "Attempts": len(r.result.attempts),
               "Tools": r.result.tool_calls, "Latency (ms)": round(r.result.duration_ms,1)}
              for r in reversed(reports)], hide_index=True, width="stretch")
