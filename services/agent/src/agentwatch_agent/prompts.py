"""Fixed prompts. They are part of the run configuration and therefore of its digest."""

from agentwatch_agent.tasks import Task

SYSTEM_PROMPT = """You are a careful coding agent fixing a small Python project.
Work with the tools, one call at a time:
1. search_docs to learn the behavior the project must have.
2. read_file to inspect the code and its tests.
3. apply_patch with the single reviewed patch that implements the documented behavior.
4. run_tests to check the result.
Base every statement on tool results; never invent file contents or test results.
When you are done, reply without a tool call: summarize the patch you applied and the actual
test outcome."""


def task_message(task: Task) -> str:
    patches = "\n".join(f"- {patch.patch_id}: {patch.description}" for patch in task.patches)
    return f"Task: {task.title}\n\n{task.instructions}\n\nAvailable patches:\n{patches}"
