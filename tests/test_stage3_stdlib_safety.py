"""Stage 3 tests: standard library, --nan-trace, --record / replay, and --doc doctests."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from zen.cli import main as cli_main
from zen.doctest import run_doctests

REPO = Path(__file__).resolve().parent.parent


def _run_zen(code: str, cwd: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    p = cwd / "tmp_run.zen"
    p.write_text(code, encoding="utf-8")
    return subprocess.run(
        [sys.executable, "-m", "zen", "run", str(p), *extra_args],
        cwd=REPO,
        capture_output=True,
        text=True,
    )


def test_stdlib_json_files_path_regex_datetime_env_csv_process_stats(tmp_path: Path) -> None:
    data_file = tmp_path / "note.txt"
    csv_file = tmp_path / "rows.csv"
    sub = tmp_path / "sub"
    code = f"""
import json
import files
import path
import regex
import datetime
import env
import csv
import process
import stats

let obj = json.parse("{{\\"a\\": 1, \\"b\\": [2, 3]}}")
let js = json.stringify(obj)
assert obj["a"] == 1
assert "b" in js

files.mkdir("{sub.as_posix()}")
files.write("{data_file.as_posix()}", "hello")
files.append("{data_file.as_posix()}", " world")
assert files.exists("{data_file.as_posix()}")
assert files.read("{data_file.as_posix()}") == "hello world"
assert path.basename("{data_file.as_posix()}") == "note.txt"
assert path.stem("{data_file.as_posix()}") == "note"
assert path.ext("{data_file.as_posix()}") == ".txt"
assert path.join(["a", "b", "c.txt"]) == "a/b/c.txt"

assert regex.match("^hel+o", "hello")
assert regex.search("\\\\d+", "v0.30!") == "0"
assert regex.find_all("[0-9]+", "a1 b22 c3") == ["1", "22", "3"]
assert regex.replace("\\\\s+", "-", "a  b c") == "a-b-c"

let ts = datetime.parse("2026-11-26 12:00:00")
assert datetime.format(ts, "%Y-%m-%d") == "2026-11-26"
assert len(datetime.today()) == 10

env.set("ZEN_TEST_VAR", "zen3")
assert env.has("ZEN_TEST_VAR")
assert env.get("ZEN_TEST_VAR") == "zen3"
say "argv=" + str(env.argv()[1:])

csv.write("{csv_file.as_posix()}", [["name", "score"], ["ada", 10], ["zen", 20]])
let rows = csv.read("{csv_file.as_posix()}")
assert rows[1][0] == "ada" and rows[2][1] == "20"

let pr = process.run(["{Path(sys.executable).as_posix()}", "-c", "print('sub_ok')"])
assert pr["code"] == 0 and "sub_ok" in pr["stdout"]

assert stats.mean([1, 2, 3, 4]) == 2.5
assert stats.median([1, 3, 2]) == 2
assert abs(stats.var([2, 4]) - 1.0) < 1e-9
assert abs(stats.std([2, 4]) - 1.0) < 1e-9
say "STDLIB_OK"
"""
    res = _run_zen(code, tmp_path, "--", "alpha", "beta")
    assert res.returncode == 0, res.stderr
    assert "argv=['alpha', 'beta']" in res.stdout
    assert "STDLIB_OK" in res.stdout


def test_stdlib_http_against_local_server(tmp_path: Path) -> None:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = json.dumps({"method": "GET", "path": self.path}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length).decode("utf-8")
            body = json.dumps({"method": "POST", "received": json.loads(raw)}).encode("utf-8")
            self.send_response(201)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        code = f"""
import http
import json
let g = http.get("http://127.0.0.1:{port}/ping")
let gd = json.parse(g["body"])
assert g["status"] == 200 and gd["method"] == "GET" and gd["path"] == "/ping"

let p = http.post("http://127.0.0.1:{port}/echo", {{"msg": "hi"}})
let pd = json.parse(p["body"])
assert p["status"] == 201 and pd["method"] == "POST" and pd["received"]["msg"] == "hi"
say "HTTP_OK"
"""
        res = _run_zen(code, tmp_path)
        assert res.returncode == 0, res.stderr
        assert res.stdout.strip() == "HTTP_OK"
    finally:
        server.shutdown()
        server.server_close()


def test_nan_trace_stops_at_first_nan_or_inf(tmp_path: Path) -> None:
    code = """
import zennum as zn
let a = zn.array([1.0, 2.0])
let b = a * 1e200
let c = b * 1e200
say c
"""
    script = tmp_path / "nan_bad.zen"
    script.write_text(code.strip() + "\n", encoding="utf-8")
    p = subprocess.run(
        [sys.executable, "-m", "zen", "run", "--nan-trace", str(script)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 1
    assert "E014" in p.stderr
    assert "Inf" in p.stderr
    assert "line 4" in p.stderr


def test_record_and_replay_match_and_mismatch(tmp_path: Path) -> None:
    data = tmp_path / "vals.txt"
    data.write_text("42", encoding="utf-8")
    script = tmp_path / "repro.zen"
    script.write_text(
        f"""
import zennum as zn
zn.seed(77)
let d = read_file("{data.as_posix()}")
say f"d={{d}}|r={{zn.rand([2]).tolist()}}"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    manifest = tmp_path / "run.json"
    rec = subprocess.run(
        [sys.executable, "-m", "zen", "run", "--record", str(manifest), str(script)],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert manifest.is_file()
    m = json.loads(manifest.read_text(encoding="utf-8"))
    assert m["seed"] == 77
    assert m["zen_files"]
    assert m["data_files"]
    assert len(m["output_sha256"]) == 64

    rep1 = subprocess.run(
        [sys.executable, "-m", "zen", "replay", str(manifest)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert rep1.returncode == 0
    assert "MATCH" in rep1.stdout

    # Mutate input data -> replay must report MISMATCH
    data.write_text("99", encoding="utf-8")
    rep2 = subprocess.run(
        [sys.executable, "-m", "zen", "replay", str(manifest)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert rep2.returncode == 1
    assert "MISMATCH" in rep2.stdout


def test_doctests_cli_and_guide_examples(tmp_path: Path) -> None:
    doc_file = tmp_path / "doc_sample.zen"
    doc_file.write_text(
        """
//> say 1 + 1
// 2
//> let x = 5
//> say f"x={x * 2}"
// x=10
say "main"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    passed, failed, msgs = run_doctests([doc_file])
    assert passed == 2 and failed == 0, msgs

    p = subprocess.run(
        [sys.executable, "-m", "zen", "test", "--doc"],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert p.returncode == 0, p.stderr
    assert "ZEN doctests:" in p.stdout and "0 failed" in p.stdout
