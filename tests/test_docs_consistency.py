"""Doc-consistency guard: every route in `src/api/main.py` must appear in the
README's API endpoint table, every `curl` example in the README's
"Example responses" section must hit a real route, every documented
example response must carry exactly the keys the API returns, and the
documented default model must agree with the model shown in the examples.

Docs and code drift silently apart (the 10-16 partial push proved it), so
these tests fail if a route is added/renamed in the API without updating the
README — or vice versa, if the README documents a route that no longer
exists. The curl guard additionally catches copy-paste drift in the
"Example responses" block (added 2026-10-18), the response-shape guard
catches doc drift when a response model gains or loses a field (2026-10-22),
and the model-agreement guard catches stale-model drift between the
documented default and the example responses (2026-10-23).
"""
import ast
import json
import os
import re

REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
MAIN_PY_PATH = os.path.join(REPO_ROOT, "src", "api", "main.py")
README_PATH = os.path.join(REPO_ROOT, "README.md")

# e.g. @app.get("/health") or @app.post("/ask", response_model=AskResponse)
ROUTE_RE = re.compile(
    r"""@app\.(get|post|put|delete|patch)\s*\(\s*["']([^"']+)["']"""
)

# a curl command line, e.g. "$ curl localhost:8000/health" or
# "curl -X POST http://localhost:8000/ask -H 'Content-Type: application/json'"
CURL_LINE_RE = re.compile(r"^\s*(?:\$\s*)?curl\b(.*)$")

# a token that looks like a URL target: scheme://host, host:port/path, or /path
TARGET_RE = re.compile(r"^(?:[\w.~-]+://|[\w.~-]+:\d+/|/)")


def _routes_in_main_py():
    with open(MAIN_PY_PATH, encoding="utf-8") as f:
        text = f.read()
    return {(m.group(1).upper(), m.group(2)) for m in ROUTE_RE.finditer(text)}


def _paths_in_readme_table():
    """Collect the Path column of the README's API endpoint table."""
    with open(README_PATH, encoding="utf-8") as f:
        lines = f.read().splitlines()
    # Locate the endpoint table: the markdown table whose header row declares
    # a Path column.
    for i, line in enumerate(lines):
        cells = [c.strip().lower() for c in line.strip().strip("|").split("|")]
        if "method" in cells and "path" in cells:
            paths = []
            for row in lines[i + 1 :]:
                row = row.strip()
                if not row.startswith("|"):
                    break
                cells = [c.strip() for c in row.strip().strip("|").split("|")]
                if all(set(c) <= set("-: ") for c in cells if c):
                    continue  # separator row
                paths.append(cells[1].strip("`"))
            return paths
    return []


def _curl_target_from_line(line):
    """First URL-like token on a curl command line, or None if the line is not
    a curl invocation (flags and -d payloads are skipped)."""
    m = CURL_LINE_RE.match(line)
    if not m:
        return None
    for token in m.group(1).split():
        token = token.strip().strip("\"'")
        if TARGET_RE.match(token):
            return token
    return None


def _url_path(target):
    """Path component of a curl target, e.g. localhost:8000/health -> /health."""
    t = target.strip().rstrip("/")
    if "://" in t:
        t = t.split("://", 1)[1]
    slash = t.find("/")
    path = t[slash:] if slash != -1 else "/"
    path = re.split(r"[?#]", path, maxsplit=1)[0]
    return path or "/"


def _curl_paths_in_example_responses(readme_text):
    """URL paths hit by `curl` commands inside the README's Example responses
    section (the block added 2026-10-18 under the API endpoint table). The
    section ends at the next level-2+ markdown heading; single-# shell comments
    inside the fenced blocks are NOT treated as headings."""
    lines = readme_text.splitlines()
    in_section = False
    paths = []
    for line in lines:
        stripped = line.strip()
        if not in_section:
            if stripped.startswith("Example responses"):
                in_section = True
            continue
        if re.match(r"^##+\s", stripped):  # next markdown heading ends the section
            break
        target = _curl_target_from_line(line)
        if target:
            paths.append(_url_path(target))
    return paths


def test_readme_has_endpoint_table():
    paths = _paths_in_readme_table()
    assert paths, "README must contain an API endpoint table with a Path column"


def test_every_route_documented_in_readme():
    routes = _routes_in_main_py()
    assert routes, "no routes found in src/api/main.py — regex may be stale"
    documented = _paths_in_readme_table()
    missing = [path for _method, path in sorted(routes) if path not in documented]
    assert not missing, (
        "routes in src/api/main.py missing from the README API endpoint table: "
        + ", ".join(missing)
    )


def test_no_documented_route_missing_from_code():
    routes = _routes_in_main_py()
    route_paths = {path for _method, path in routes}
    documented = _paths_in_readme_table()
    stale = [path for path in documented if path not in route_paths]
    assert not stale, (
        "README API endpoint table documents routes not present in "
        "src/api/main.py: " + ", ".join(stale)
    )


def test_readme_curl_examples_hit_real_routes():
    """Every curl example in the README's Example responses block must call a
    route that exists in src/api/main.py."""
    with open(README_PATH, encoding="utf-8") as f:
        readme_text = f.read()
    curl_paths = _curl_paths_in_example_responses(readme_text)
    assert curl_paths, (
        "README 'Example responses' section must contain at least one curl command"
    )
    route_paths = {path for _method, path in _routes_in_main_py()}
    bad = [p for p in curl_paths if p not in route_paths]
    assert not bad, (
        "README curl examples call routes not present in src/api/main.py: "
        + ", ".join(bad)
    )


def test_curl_example_guard_catches_nonexistent_route(tmp_path):
    """Drift detection: a README snippet whose curl points at a nonexistent
    route must be flagged by the guard."""
    snippet = tmp_path / "README.md"
    snippet.write_text(
        "Example responses:\n"
        "\n"
        "```bash\n"
        "curl localhost:8000/nonexistent-route\n"
        "```\n",
        encoding="utf-8",
    )
    route_paths = {path for _method, path in _routes_in_main_py()}
    bogus = [
        p
        for p in _curl_paths_in_example_responses(
            snippet.read_text(encoding="utf-8")
        )
        if p not in route_paths
    ]
    assert bogus == ["/nonexistent-route"], (
        "curl-route guard should flag a nonexistent route, got: %r" % bogus
    )


# --- response-shape guard (2026-10-22) -------------------------------------

# a backtick route line introducing a fenced example, e.g. `POST /ask` — ...
_EXAMPLE_ROUTE_RE = re.compile(r"^`([A-Z]+)\s+(/\S+?)`")

# a fenced code block boundary, e.g. ```json or ```
_FENCE_RE = re.compile(r"^```(\w*)\s*$")


def _list_element_model(annotation):
    """'list[Source]' -> 'Source'; anything else -> None."""
    if (
        isinstance(annotation, ast.Subscript)
        and isinstance(annotation.value, ast.Name)
        and annotation.value.id == "list"
        and isinstance(annotation.slice, ast.Name)
    ):
        return annotation.slice.id
    return None


def _response_shapes_in_main_py():
    """Map each route path to its response shape, parsed from src/api/main.py
    with ast: {path: (top_level_fields, {list_field: item_fields})}.

    Routes declared with response_model=X use X's annotated fields; routes
    without a model use the keys of the dict literal returned by the handler.
    Nested list[Model] fields map to that model's fields so example items can
    be checked too (sources -> Source, trace -> TraceStep).
    """
    with open(MAIN_PY_PATH, encoding="utf-8") as f:
        tree = ast.parse(f.read())

    models = {}  # class name -> {field: element model name or None}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(
            isinstance(b, ast.Name) and b.id == "BaseModel" for b in node.bases
        ):
            continue
        fields = {}
        for stmt in node.body:
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
                fields[stmt.target.id] = _list_element_model(stmt.annotation)
        models[node.name] = fields

    handlers = {}  # function name -> FunctionDef
    routes = {}  # path -> (response_model name or None, handler name)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        handlers[node.name] = node
        for dec in node.decorator_list:
            if (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Attribute)
                and isinstance(dec.func.value, ast.Name)
                and dec.func.value.id == "app"
                and dec.args
                and isinstance(dec.args[0], ast.Constant)
            ):
                model = None
                for kw in dec.keywords:
                    if kw.arg == "response_model" and isinstance(kw.value, ast.Name):
                        model = kw.value.id
                routes[dec.args[0].value] = (model, node.name)

    shapes = {}
    for path, (model, handler_name) in routes.items():
        if model and model in models:
            fields = models[model]
            nested = {
                f: list(models[elem])
                for f, elem in fields.items()
                if elem and elem in models
            }
            shapes[path] = (list(fields), nested)
        else:
            # No declared model: keys of the dict literal the handler returns.
            keys = []
            fn = handlers.get(handler_name)
            if fn is not None:
                for ret in ast.walk(fn):
                    if isinstance(ret, ast.Return) and isinstance(
                        ret.value, ast.Dict
                    ):
                        keys = [
                            k.value
                            for k in ret.value.keys
                            if isinstance(k, ast.Constant)
                            and isinstance(k.value, str)
                        ]
            shapes[path] = (keys, {})
    return shapes


def _example_payloads_in_readme(readme_text):
    """{route path: example response dict} for every documented example
    response in the README's 'Example responses' section.

    Two shapes are supported (both used in the section today):
    - curl lines followed by `# → {...}` comment lines (possibly multi-line);
    - a `METHOD /path` backtick line followed by a ```json fenced block.
    Malformed examples are skipped — the key check simply can't run on them.
    """
    lines = readme_text.splitlines()
    section = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if not in_section:
            if stripped.startswith("Example responses"):
                in_section = True
            continue
        if re.match(r"^##+\s", stripped):
            break
        section.append(line)

    payloads = {}
    pending_path = None
    comment_lines = []
    in_json_block = False
    block_lines = []

    def _parse(text):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

    def flush_comments():
        nonlocal pending_path, comment_lines
        if pending_path and comment_lines:
            payload = _parse("\n".join(comment_lines))
            if payload is not None:
                payloads[pending_path] = payload
        pending_path, comment_lines = None, []

    for line in section:
        stripped = line.strip()
        if in_json_block:
            if stripped.startswith("```"):
                in_json_block = False
                if pending_path:
                    payload = _parse("\n".join(block_lines))
                    if payload is not None:
                        payloads[pending_path] = payload
                    pending_path = None
                block_lines = []
            else:
                block_lines.append(line)
            continue
        m = _FENCE_RE.match(stripped)
        if m:
            if m.group(1) == "json" and pending_path:
                in_json_block = True
                block_lines = []
            else:
                flush_comments()
            continue
        m = _EXAMPLE_ROUTE_RE.match(stripped)
        if m:
            flush_comments()
            pending_path = m.group(2)
            continue
        target = _curl_target_from_line(line)
        if target:
            flush_comments()
            pending_path = _url_path(target)
            continue
        if pending_path and stripped.startswith("#"):
            body = stripped[1:].lstrip()
            if body.startswith("→"):
                comment_lines.append(body[1:].lstrip())
            elif comment_lines:
                # continuation line of a multi-line `# → {...}` payload
                comment_lines.append(body)
            continue
        if pending_path and not stripped:
            continue  # blank line inside/around an example
        flush_comments()
    flush_comments()
    return payloads


def _check_example_keys(payloads, shapes):
    """Compare documented example payloads against the shapes parsed from
    main.py. Returns a list of human-readable problems (empty == consistent)."""
    problems = []
    for path in sorted(payloads):
        payload = payloads[path]
        if path not in shapes:
            problems.append(
                f"{path}: example documents a route not in src/api/main.py"
            )
            continue
        if not isinstance(payload, dict):
            problems.append(f"{path}: example response is not a JSON object")
            continue
        fields, nested = shapes[path]
        extra = sorted(set(payload) - set(fields))
        missing = sorted(set(fields) - set(payload))
        if extra or missing:
            detail = []
            if extra:
                detail.append("documented but not returned: " + ", ".join(extra))
            if missing:
                detail.append("returned but not documented: " + ", ".join(missing))
            problems.append(f"{path}: response keys drift — " + "; ".join(detail))
        for list_field, item_fields in nested.items():
            for i, item in enumerate(payload.get(list_field) or []):
                if not isinstance(item, dict):
                    continue
                ie = sorted(set(item) - set(item_fields))
                im = sorted(set(item_fields) - set(item))
                if ie or im:
                    parts = []
                    if ie:
                        parts.append("documented but not in model: " + ", ".join(ie))
                    if im:
                        parts.append("in model but not documented: " + ", ".join(im))
                    problems.append(
                        f"{path}.{list_field}[{i}]: item keys drift — "
                        + "; ".join(parts)
                    )
    return problems


def test_example_response_keys_match_response_models():
    """Every documented example response must carry exactly the keys the API
    returns: top-level keys match the response_model fields (or the returned
    dict literal for model-less routes), and nested sources/trace items match
    their pydantic models. Fails when a model gains or loses a field without
    the docs being updated."""
    with open(README_PATH, encoding="utf-8") as f:
        readme_text = f.read()
    payloads = _example_payloads_in_readme(readme_text)
    assert set(payloads) >= {"/health", "/drugs", "/ask", "/ask-agent"}, (
        "README 'Example responses' must document parseable responses for all "
        f"four routes, got: {sorted(payloads)}"
    )
    shapes = _response_shapes_in_main_py()
    problems = _check_example_keys(payloads, shapes)
    assert not problems, (
        "example responses drifted from src/api/main.py:\n" + "\n".join(problems)
    )


def test_response_key_guard_catches_extra_key():
    """Drift detection: an example response carrying a key the model doesn't
    have must be flagged."""
    shapes = _response_shapes_in_main_py()
    assert "/ask" in shapes
    payloads = {
        "/ask": {
            "answer": "x",
            "sources": [],
            "model": "m",
            "indexed_drug": None,
            "rewritten_question": None,
            "bogus_field": 1,
        }
    }
    problems = _check_example_keys(payloads, shapes)
    assert len(problems) == 1 and "bogus_field" in problems[0], problems


def test_response_key_guard_catches_missing_key():
    """Drift detection: an example missing a model field must be flagged."""
    shapes = _response_shapes_in_main_py()
    payloads = {"/ask": {"answer": "x", "sources": [], "model": "m"}}
    problems = _check_example_keys(payloads, shapes)
    assert problems and "indexed_drug" in problems[0], problems


def test_response_key_guard_catches_nested_drift():
    """Drift detection: a sources item carrying a key the Source model doesn't
    have must be flagged."""
    shapes = _response_shapes_in_main_py()
    payloads = {
        "/ask": {
            "answer": "x",
            "sources": [
                {
                    "ref": 1,
                    "drug_name": "d",
                    "section_title": "s",
                    "score": 0.5,
                    "preview": "p",
                    "bogus_nested": 1,
                }
            ],
            "model": "m",
            "indexed_drug": None,
            "rewritten_question": None,
        }
    }
    problems = _check_example_keys(payloads, shapes)
    assert len(problems) == 1 and "bogus_nested" in problems[0], problems


# --- model-agreement guard (2026-10-23) -------------------------------------

# the documented default, e.g. "Default model:\n    `gemini-3.6-flash` (pinned ..."
_DEFAULT_MODEL_RE = re.compile(r"Default model:\s*`([^`]+)`")


def _default_model_in_readme(readme_text):
    """The model named by the README's 'Default model:' line, or None."""
    m = _DEFAULT_MODEL_RE.search(readme_text)
    return m.group(1).strip() if m else None


def _example_model(payloads, path):
    """The model string a documented example response shows for a route.

    /health nests it under the llm object; /ask and /ask-agent carry a
    top-level 'model' key. Returns None when the example is absent.
    """
    payload = payloads.get(path)
    if not isinstance(payload, dict):
        return None
    if path == "/health":
        llm = payload.get("llm")
        return llm.get("model") if isinstance(llm, dict) else None
    return payload.get("model")


def _check_model_agreement(readme_text):
    """Assert the documented default model matches the model each example
    response shows. Returns a list of human-readable problems (empty ==
    consistent)."""
    problems = []
    default = _default_model_in_readme(readme_text)
    if not default:
        return ["README has no parseable 'Default model:' line"]
    payloads = _example_payloads_in_readme(readme_text)
    for path in ("/health", "/ask", "/ask-agent"):
        shown = _example_model(payloads, path)
        if shown is None:
            problems.append(f"{path}: no example response (or no model in it)")
        elif shown != default:
            problems.append(
                f"{path}: example shows model {shown!r} but the documented "
                f"default model is {default!r}"
            )
    return problems


def test_documented_models_agree_with_default():
    """The README's 'Default model' line and the model shown in the
    /health, /ask and /ask-agent example responses must agree — the same
    stale-model drift family the response-shape guard now covers
    structurally (e.g. the 2.5-flash deprecation wave)."""
    with open(README_PATH, encoding="utf-8") as f:
        readme_text = f.read()
    default = _default_model_in_readme(readme_text)
    assert default, "README must name a default model"
    problems = _check_model_agreement(readme_text)
    assert not problems, "model drift between docs and examples:\n" + "\n".join(
        problems
    )


def test_model_agreement_guard_catches_health_drift():
    """Drift detection: a /health example showing a different model than the
    documented default must be flagged."""
    snippet = (
        "Notes:\n"
        "Default model: `gemini-3.6-flash` (pinned in `src/config.py`)\n"
        "\n"
        "Example responses:\n"
        "\n"
        "```bash\n"
        "curl localhost:8000/health\n"
        '# → {"status": "ok", "llm": {"provider": "gemini", '
        '"model": "gemini-2.5-flash", "configured": true}}\n'
        "```\n"
    )
    problems = _check_model_agreement(snippet)
    assert any("/health" in p and "gemini-2.5-flash" in p for p in problems), problems


def test_model_agreement_guard_catches_ask_drift():
    """Drift detection: a /ask example showing a different model than the
    documented default must be flagged."""
    snippet = (
        "Notes:\n"
        "Default model: `gemini-3.6-flash`\n"
        "\n"
        "Example responses:\n"
        "\n"
        "`POST /ask` — grounded answer:\n"
        "\n"
        "```json\n"
        '{"answer": "x", "model": "gpt-4o-mini"}\n'
        "```\n"
    )
    problems = _check_model_agreement(snippet)
    assert any("/ask" in p and "gpt-4o-mini" in p for p in problems), problems
