"""Evidence bundle checks before sharing. Every credential below is a made-up fixture."""

import json
import zipfile

import pytest
from agentwatch_agent import redaction_check
from agentwatch_agent.cli import main
from agentwatch_agent.redacted_copy import write_redacted_copy
from agentwatch_agent.redaction_check import (
    CHECK_METHOD,
    CLEAN,
    REFUSED,
    BundleError,
    check_bundle,
    sensitive_kinds,
)

# Built from parts so no complete credential-shaped literal sits in the source.
GITHUB_TOKEN = "ghp_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q7r8"
PW = "PASS" + "WORD"
PLAIN = "agent" + "watch"
EMAIL = "alice@example.test"
HOME_PATH = "/Users/alice/project/stats.py"
KEY_HEADER = "-----BEGIN " + "RSA PRIVATE" + " KEY-----"
SECRETS = (GITHUB_TOKEN, PLAIN, EMAIL, "alice", KEY_HEADER)


def write(folder, relative, content):
    path = folder / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content, encoding="utf-8")
    return path


def clean_folder(root):
    folder = root / "bundle"
    write(
        folder,
        "attempts/a1/sampling-overhead.json",
        json.dumps({"run_id": "97fb140d-be6f-42f9-b31c-0d1ab305aabf", "median": 0.25}),
    )
    write(folder, "attempts/a1/live-journal.jsonl", '{"kind": "record"}\n\n{"kind": "close"}\n')
    write(folder, "notes.md", "Two attempts on the CPU container, POST /v1/chat/completions.\n")
    return folder


def locations(report):
    return [
        (finding["file"], finding["location"], finding["kinds"]) for finding in report["findings"]
    ]


def test_a_clean_folder_is_clean(tmp_path):
    report = check_bundle(clean_folder(tmp_path))
    assert report["verdict"] == CLEAN
    assert report["method"] == CHECK_METHOD
    assert report["files_scanned"] == 3
    assert report["findings"] == report["unscanned"] == report["json_read_as_text"] == []
    assert report["totals"] == {}


def test_findings_give_the_file_location_and_kind_but_never_the_value(tmp_path):
    folder = clean_folder(tmp_path)
    write(
        folder, "run/evidence.json", json.dumps({"steps": [{"reason": f"failed in {HOME_PATH}"}]})
    )
    write(folder, "run/journal.jsonl", '{"kind": "a"}\n' + json.dumps({"to": EMAIL}) + "\n")
    write(folder, "logs/server.log", f"started\npushing with {GITHUB_TOKEN}\n")
    write(folder, "keys.txt", f"{KEY_HEADER}\nnot-a-real-key\n")
    report = check_bundle(folder)
    assert report["verdict"] == REFUSED
    assert locations(report) == [
        ("keys.txt", "line 1", {"private_key": 1}),
        ("logs/server.log", "line 2", {"token": 1}),
        ("run/evidence.json", "/steps/0/reason", {"path": 1}),
        ("run/journal.jsonl", "line 2 /to", {"email": 1}),
    ]
    assert report["totals"] == {"email": 1, "path": 1, "private_key": 1, "token": 1}
    text = json.dumps(report)
    assert not any(secret in text for secret in SECRETS)


def test_hidden_files_are_checked_because_they_are_shared_too(tmp_path):
    folder = clean_folder(tmp_path)
    write(folder, ".env", f"POSTGRES_{PW}={PLAIN}\n")
    report = check_bundle(folder)
    assert locations(report) == [(".env", "line 1", {"secret_assignment": 1})]


def test_json_is_checked_after_parsing_so_escaped_paths_are_found(tmp_path):
    folder = clean_folder(tmp_path)
    # In the file the separators are escaped (`C:\\Users\\bob`), which a text scan would miss.
    write(folder, "evidence.json", json.dumps({"cwd": "C:\\Users\\bob\\run"}))
    assert locations(check_bundle(folder)) == [("evidence.json", "/cwd", {"path": 1})]


def test_every_value_of_a_repeated_json_key_is_checked(tmp_path):
    folder = clean_folder(tmp_path)
    # A normal parser keeps only the last value, so the earlier email would go unread.
    write(folder, "evidence.json", f'{{"note": "{EMAIL}", "note": "fine"}}')
    assert locations(check_bundle(folder)) == [("evidence.json", "/note", {"email": 1})]


def test_a_sensitive_key_is_reported_by_its_redacted_location(tmp_path):
    folder = clean_folder(tmp_path)
    write(folder, "evidence.json", json.dumps({"by_path": {HOME_PATH: 3}}))
    report = check_bundle(folder)
    assert locations(report) == [("evidence.json", "/by_path/<path> (key)", {"path": 1})]
    assert "alice" not in json.dumps(report)


def test_a_sensitive_file_name_is_a_finding_and_is_redacted_in_the_report(tmp_path):
    folder = clean_folder(tmp_path)
    write(folder, f"from-{EMAIL}.txt", "nothing here\n")
    report = check_bundle(folder)
    # The address rule takes the extension as part of the domain, so the whole name is hidden.
    assert locations(report) == [("<email>", "file name", {"email": 1})]
    assert EMAIL not in json.dumps(report)


def test_invalid_json_is_checked_as_text_and_listed(tmp_path):
    folder = clean_folder(tmp_path)
    write(folder, "broken.json", f'{{"to": "{EMAIL}"')
    write(folder, "mixed.jsonl", '{"kind": "a"}\nnot json\n')
    report = check_bundle(folder)
    assert locations(report) == [("broken.json", "line 1", {"email": 1})]
    assert report["json_read_as_text"] == ["broken.json", "mixed.jsonl"]


@pytest.mark.parametrize(
    ("name", "content", "reason"),
    [
        ("plot.png", b"\x89PNG\r\n\x1a\n\xff\xfe", "not UTF-8 text"),
        (".DS_Store", b"\x00\x00\x00\x01Bud1\xff", "not UTF-8 text"),
        ("nested.zip", None, "not UTF-8 text"),
    ],
)
def test_a_file_that_cannot_be_checked_refuses_the_bundle(tmp_path, name, content, reason):
    folder = clean_folder(tmp_path)
    if content is None:
        with zipfile.ZipFile(folder / name, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("inner.json", json.dumps({"to": EMAIL}))
    else:
        write(folder, name, content)
    report = check_bundle(folder)
    assert report["verdict"] == REFUSED
    assert report["findings"] == []
    assert report["unscanned"] == [{"file": name, "reason": reason}]


def test_symbolic_links_are_refused_rather_than_followed(tmp_path):
    folder = clean_folder(tmp_path)
    outside = write(tmp_path, "outside/secret.txt", f"{GITHUB_TOKEN}\n")
    (folder / "linked.txt").symlink_to(outside)
    (folder / "linked-dir").symlink_to(outside.parent, target_is_directory=True)
    report = check_bundle(folder)
    assert report["verdict"] == REFUSED
    assert [item["file"] for item in report["unscanned"]] == ["linked-dir", "linked.txt"]
    assert report["findings"] == []


def test_a_file_over_the_size_limit_is_not_read(tmp_path, monkeypatch):
    monkeypatch.setattr(redaction_check, "MAX_FILE_BYTES", 1024 * 1024)
    folder = clean_folder(tmp_path)
    write(folder, "big.log", "x" * (1024 * 1024 + 1))
    report = check_bundle(folder)
    assert report["unscanned"] == [{"file": "big.log", "reason": "larger than the 1 MiB limit"}]


def make_zip(path, members):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in members.items():
            archive.writestr(name, content)
    return path


def test_a_zip_bundle_is_checked_member_by_member(tmp_path):
    clean = make_zip(tmp_path / "clean.zip", {"a/run.json": '{"median": 0.25}', "a/": ""})
    assert check_bundle(clean)["verdict"] == CLEAN
    dirty = make_zip(
        tmp_path / "dirty.zip",
        {"a/run.json": json.dumps({"cwd": HOME_PATH}), "b/log.txt": f"ok\nby {EMAIL}\n"},
    )
    report = check_bundle(dirty)
    assert report["verdict"] == REFUSED
    assert locations(report) == [
        ("a/run.json", "/cwd", {"path": 1}),
        ("b/log.txt", "line 2", {"email": 1}),
    ]


def test_an_encrypted_zip_member_is_not_checked(tmp_path):
    path = make_zip(tmp_path / "locked.zip", {"secret.json": "{}"})
    # zipfile cannot write encrypted members, so set the encryption flag in both headers.
    data = bytearray(path.read_bytes())
    for signature, flag_offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        start = data.index(signature)
        data[start + flag_offset] |= 0x1
    path.write_bytes(bytes(data))
    report = check_bundle(path)
    assert report["unscanned"] == [{"file": "secret.json", "reason": "encrypted zip member"}]


def test_a_zip_member_whose_declared_size_is_wrong_is_refused_not_checked(tmp_path):
    path = make_zip(tmp_path / "crafted.zip", {"big.txt": f"{EMAIL}\n" + "y" * 4096})
    # Understate the member's size in the central directory, as a crafted archive could: the
    # read stops at the declared size and fails its CRC, so the member is refused unread.
    data = bytearray(path.read_bytes())
    start = data.index(b"PK\x01\x02")
    data[start + 24 : start + 28] = (16).to_bytes(4, "little")
    path.write_bytes(bytes(data))
    report = check_bundle(path)
    assert report["verdict"] == REFUSED
    assert report["unscanned"] == [{"file": "big.txt", "reason": "could not be read (BadZipFile)"}]


def test_a_zip_member_over_the_size_limit_is_not_read(tmp_path, monkeypatch):
    monkeypatch.setattr(redaction_check, "MAX_FILE_BYTES", 1024 * 1024)
    path = make_zip(tmp_path / "big.zip", {"big.txt": "y" * (1024 * 1024 + 1)})
    assert check_bundle(path)["unscanned"] == [
        {"file": "big.txt", "reason": "larger than the 1 MiB limit"}
    ]


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (lambda root: root / "missing", "does not exist"),
        (lambda root: (root / "empty").mkdir() or root / "empty", "contains no files"),
        (lambda root: write(root, "notes.txt", "hello"), "neither a folder nor a zip"),
        (lambda root: make_zip(root / "empty.zip", {"a/": ""}), "contains no files"),
    ],
)
def test_a_bundle_that_cannot_be_checked_at_all_is_an_error(tmp_path, make, message):
    with pytest.raises(BundleError, match=message):
        check_bundle(make(tmp_path))


def test_a_copy_made_by_redact_passes_the_check(tmp_path):
    source = tmp_path / "run"
    write(
        source,
        "agent-evidence.json",
        json.dumps(
            {
                "messages": [{"content": f"report to {EMAIL} from {HOME_PATH}"}],
                "by_path": {HOME_PATH: 1, "C:\\Users\\bob\\x.py": 2},
                "auth": f"Authorization: Bearer {GITHUB_TOKEN}",
            }
        ),
    )
    write(source, "live-journal.jsonl", json.dumps({"env": f"POSTGRES_{PW}={PLAIN}"}) + "\n")
    assert check_bundle(source)["verdict"] == REFUSED
    copy = tmp_path / "shared"
    write_redacted_copy(source, copy)
    report = check_bundle(copy)
    assert report["verdict"] == CLEAN, locations(report)
    assert report["files_scanned"] == 3  # the two copies and redaction.json


def test_a_private_key_header_alone_counts_even_without_its_end_line():
    assert sensitive_kinds(f"{KEY_HEADER}\nMIIE") == {"private_key": 1}
    assert sensitive_kinds("no key here") == {}


# --- Command line -----------------------------------------------------------------------------


def test_the_command_exits_0_for_a_clean_bundle_and_writes_the_report(tmp_path, capsys):
    report_path = tmp_path / "check.json"
    assert main(["check-redaction", str(clean_folder(tmp_path)), "--report", str(report_path)]) == 0
    assert "Clean" in capsys.readouterr().out
    assert json.loads(report_path.read_text(encoding="utf-8"))["verdict"] == CLEAN


def test_the_command_exits_1_and_refuses_without_printing_values(tmp_path, capsys):
    folder = clean_folder(tmp_path)
    write(folder, "logs/server.log", f"pushing with {GITHUB_TOKEN}\n")
    write(folder, "plot.png", b"\xff\xfe")
    assert main(["check-redaction", str(folder)]) == 1
    out = capsys.readouterr().out
    assert "FOUND  logs/server.log (line 1): token 1" in out
    assert "NOT CHECKED  plot.png: not UTF-8 text" in out
    assert "Refused" in out
    assert GITHUB_TOKEN not in out


def test_the_command_exits_2_when_the_bundle_or_report_is_unusable(tmp_path, capsys):
    assert main(["check-redaction", str(tmp_path / "missing")]) == 2
    folder = clean_folder(tmp_path)
    existing = write(tmp_path, "check.json", "{}")
    assert main(["check-redaction", str(folder), "--report", str(existing)]) == 2
    assert existing.read_text(encoding="utf-8") == "{}"
    assert "already exists" in capsys.readouterr().err
