"""Load the fixed, versioned coding tasks the agent runner executes.

A task is a small fixture project, reference documents for retrieval, and reviewed candidate
patches. `task_id` is derived from the task name and revision, and the content hash of every
task file is pinned in `task_fixtures/index.toml`: editing a task without bumping its revision
fails loading, so a task identity never silently covers different inputs.
"""

import hashlib
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

TASKS_ROOT = Path(__file__).parent / "task_fixtures"
_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_IGNORED_PARTS = {"__pycache__"}


class TaskDefinitionError(ValueError):
    """A task fixture is malformed or does not match its pinned identity."""


@dataclass(frozen=True)
class Patch:
    patch_id: str
    description: str
    file: str
    find: str
    replace: str


@dataclass(frozen=True)
class Task:
    name: str
    revision: int
    title: str
    instructions: str
    patches: tuple[Patch, ...]
    root: Path
    content_sha256: str

    @property
    def task_id(self) -> str:
        return f"{self.name}-r{self.revision}"

    @property
    def project_dir(self) -> Path:
        return self.root / "project"

    @property
    def docs_dir(self) -> Path:
        return self.root / "docs"

    def patch(self, patch_id: str) -> Patch | None:
        return next((p for p in self.patches if p.patch_id == patch_id), None)


def task_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not _IGNORED_PARTS.intersection(path.relative_to(root).parts)
        and path.suffix != ".pyc"
    )


def content_sha256(root: Path) -> str:
    """Hash every task file's relative path and bytes, independent of filesystem order."""
    digest = hashlib.sha256()
    for path in task_files(root):
        relative = path.relative_to(root).as_posix().encode()
        data = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big") + relative)
        digest.update(len(data).to_bytes(8, "big") + data)
    return digest.hexdigest()


def _string(table: dict, key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TaskDefinitionError(f"{where}: {key} must be a nonblank string")
    return value


def _load_patches(definition: dict, project: Path, where: str) -> tuple[Patch, ...]:
    raw = definition.get("patches")
    if not isinstance(raw, list) or not raw:
        raise TaskDefinitionError(f"{where}: at least one patch is required")
    patches = []
    for index, table in enumerate(raw):
        if not isinstance(table, dict):
            raise TaskDefinitionError(f"{where}: patches[{index}] must be a table")
        patch = Patch(
            patch_id=_string(table, "id", where),
            description=_string(table, "description", where),
            file=_string(table, "file", where),
            find=_string(table, "find", where),
            replace=_string(table, "replace", where),
        )
        if not _NAME.fullmatch(patch.patch_id):
            raise TaskDefinitionError(f"{where}: invalid patch id {patch.patch_id!r}")
        target = project / patch.file
        if Path(patch.file).is_absolute() or ".." in Path(patch.file).parts:
            raise TaskDefinitionError(f"{where}: patch {patch.patch_id} escapes the project")
        if not target.is_file() or target.read_text().count(patch.find) != 1:
            raise TaskDefinitionError(
                f"{where}: patch {patch.patch_id} must match its file exactly once"
            )
        patches.append(patch)
    if len({p.patch_id for p in patches}) != len(patches):
        raise TaskDefinitionError(f"{where}: patch ids must be unique")
    return tuple(patches)


def load_task(name: str, root: Path = TASKS_ROOT) -> Task:
    index = tomllib.loads((root / "index.toml").read_text())
    pinned = {
        entry.get("name"): entry for entry in index.get("task", []) if isinstance(entry, dict)
    }
    if name not in pinned:
        raise TaskDefinitionError(f"Unknown task {name!r}; available: {sorted(pinned)}")
    task_root = root / name
    where = f"task {name}"
    definition = tomllib.loads((task_root / "task.toml").read_text())
    revision = definition.get("revision")
    if definition.get("name") != name:
        raise TaskDefinitionError(f"{where}: task.toml name does not match its directory")
    if type(revision) is not int or revision < 1:
        raise TaskDefinitionError(f"{where}: revision must be a positive integer")
    if pinned[name].get("revision") != revision:
        raise TaskDefinitionError(f"{where}: revision differs from task_fixtures/index.toml")
    for directory in ("project", "docs"):
        if not (task_root / directory).is_dir():
            raise TaskDefinitionError(f"{where}: missing {directory}/ directory")
    actual = content_sha256(task_root)
    if pinned[name].get("content_sha256") != actual:
        raise TaskDefinitionError(
            f"{where}: content changed without a revision bump (now {actual}); "
            "bump the revision and update task_fixtures/index.toml"
        )
    return Task(
        name=name,
        revision=revision,
        title=_string(definition, "title", where),
        instructions=_string(definition, "instructions", where).strip(),
        patches=_load_patches(definition, task_root / "project", where),
        root=task_root,
        content_sha256=actual,
    )


def task_names(root: Path = TASKS_ROOT) -> list[str]:
    index = tomllib.loads((root / "index.toml").read_text())
    return sorted(
        entry["name"]
        for entry in index.get("task", [])
        if isinstance(entry, dict) and isinstance(entry.get("name"), str)
    )
