# Threat model

Trust boundaries, abuse cases, controls, and security-test ownership live here.

## Tool execution (A03)

**Asset:** the host running the agent, its files and credentials, the network it can reach,
and the integrity of recorded evidence.

**Trust boundary:** test suites run either as a host subprocess (reviewed fixtures only) or
in the isolated container runner ([ADR 0004](../adr/0004-container-isolation-for-tool-execution.md)).
Model output is untrusted; today it can only choose among reviewed patches (ADR 0002).

| Abuse case | Control | Test (`services/sandbox/tests/test_container.py`) |
| --- | --- | --- |
| Read host files or credentials | Private (`0700`) staging copy only, read-only; symlinks and special files are rejected before and after copying; unsafe mount paths are refused | `test_host_filesystem_is_invisible`, `test_symlinks_and_missing_projects_are_rejected`, `test_a_symlink_swapped_in_after_validation_is_rejected`, `test_staging_copy_is_private_to_the_host_user`, `test_a_temporary_directory_that_would_break_the_mount_is_refused` |
| Modify the workspace, image or host | Read-only mount and root; `/tmp` is a bounded `noexec` tmpfs; no `/dev/shm` (`--ipc none`) | `test_workspace_and_root_are_read_only_and_tmp_is_bounded_noexec` |
| Exfiltrate data or download code | `--network none`; `--pull never` with a pinned digest | `test_network_is_unavailable`, `test_invalid_requests_are_rejected_without_creating_anything` |
| Escalate privileges | uid 65534, empty capability bounding set, `no-new-privileges`, default seccomp | `test_runs_unprivileged_without_capabilities` |
| Read host secrets from the environment | No host environment is passed | `test_host_environment_is_not_inherited` |
| Exhaust memory, processes, CPU or file handles | Memory and swap cap, pids limit, CPU quota, nofile limit | `test_memory_limit_ends_the_suite_as_a_resource_limit`, `test_process_cpu_and_file_limits_apply` |
| Run forever or outlive the runner | Host deadline and cancellation kill, also during start-up; in-container backstop; deadline-based sweep of leftovers | `test_timeout_kills_the_container`, `test_backstop_ends_the_container_without_the_host`, `test_cancellation_kills_the_container`, `test_cancellation_before_start_creates_nothing`, `test_a_deadline_during_start_up_still_kills_the_container`, `test_only_expired_sandboxes_are_swept` |
| Disable the backstop or end the wrapper | The wrapper is PID 1, so signals it has no handler for are dropped; its SIGINT handler does nothing; an early `SIGALRM` is ignored | `test_stopping_pid_1_does_not_defeat_the_backstop`, `test_signals_from_the_suite_cannot_end_the_wrapper`, `test_an_early_alarm_cannot_fake_a_timeout` |
| Flood output to exhaust host memory or disk | The host drains output and stores only a bounded tail (4,000 bytes, 64 KiB for the report); a suite that writes more than 4 MiB is stopped | `test_an_output_flood_is_a_resource_limit_and_stays_bounded`, `test_captured_output_is_stored_only_up_to_its_bound` |
| Leak containers or disk | Remove and verify on every outcome; a failed release is raised, or attached to the error already being raised; an unreachable daemon is not reported as a leak of a container never created | `no_leaks` fixture on every Docker test; `test_failed_release_is_raised_with_the_result`, `test_a_release_that_raises_after_a_run_keeps_the_result`, `test_a_failing_release_does_not_replace_the_original_error`, `test_an_unreachable_daemon_reports_no_false_leak` |
| Forge the wrapper's report | The non-dumpable PID 1 wrapper keeps the nonce and its stdout out of the suite's reach | `test_the_suite_cannot_read_or_write_the_wrapper`, `test_a_suite_cannot_forge_a_passing_result` |
| Fake a pass by exiting 0 | **Not prevented.** Code under test owns its process and can call `os._exit(0)`. Acceptable while the model only selects reviewed patches (ADR 0002) | None; documented limitation |

**Out of scope:** kernel or container-runtime escapes (shared-kernel isolation; use gVisor or a
VM for hostile code), data the suite deliberately prints into its own output, and a suite that
reports its own success dishonestly (above).

**Owners:** A03 (Akshayaa) maintains the controls and tests. Benoy integrates the runner into
the agent and owns the run-level evidence.
