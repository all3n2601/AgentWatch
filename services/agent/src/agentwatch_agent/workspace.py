"""A disposable copy of a task's fixture project that the agent's tools may change."""

import shutil
import tempfile
from pathlib import Path

from agentwatch_agent.tasks import Patch, Task, task_files

MAX_READ_BYTES = 65_536


class WorkspaceError(Exception):
    """A tool request cannot be satisfied in this workspace; the message is model-safe."""


class Workspace:
    """Files are addressed only by the relative paths copied from the fixture.

    Reads are limited to that fixed allowlist, so symlinks, `..` and absolute paths cannot
    reach anything else. Changes happen only through reviewed task patches, each written
    against the original fixture: a file accepts one patch, so candidates never stack.
    """

    def __init__(self, task: Task, root: Path):
        self.task = task
        self.root = root
        self.files = frozenset(path.relative_to(root).as_posix() for path in task_files(root))
        self.applied_patches: list[str] = []
        self._patched_files: dict[str, str] = {}

    @classmethod
    def create(cls, task: Task, parent: Path | None = None) -> "Workspace":
        root = Path(tempfile.mkdtemp(prefix="agentwatch-agent-", dir=parent)) / "project"
        shutil.copytree(task.project_dir, root, ignore=shutil.ignore_patterns("__pycache__"))
        return cls(task, root)

    def read(self, relative: str) -> str:
        if relative not in self.files:
            raise WorkspaceError(
                f"Unknown file {relative!r}; readable files: {', '.join(sorted(self.files))}"
            )
        data = (self.root / relative).read_bytes()[: MAX_READ_BYTES + 1]
        if len(data) > MAX_READ_BYTES:
            raise WorkspaceError(f"{relative} exceeds the {MAX_READ_BYTES}-byte read limit")
        return data.decode("utf-8", errors="replace")

    def apply(self, patch: Patch) -> None:
        if patch.patch_id in self.applied_patches:
            raise WorkspaceError(f"Patch {patch.patch_id} is already applied")
        if patch.file in self._patched_files:
            raise WorkspaceError(
                f"Patch {patch.patch_id} cannot be applied: {patch.file} already has patch "
                f"{self._patched_files[patch.file]}; only one patch per file is allowed"
            )
        path = self.root / patch.file
        content = path.read_text()
        if content.count(patch.find) != 1:
            raise WorkspaceError(f"Patch {patch.patch_id} does not match {patch.file}")
        path.write_text(content.replace(patch.find, patch.replace, 1))
        self.applied_patches.append(patch.patch_id)
        self._patched_files[patch.file] = patch.patch_id

    def remove(self) -> bool:
        """Delete the workspace and report whether it is actually gone."""
        container = self.root.parent
        shutil.rmtree(container, ignore_errors=True)
        return not container.exists()
