# T_FINAL Agent Prompt: Windows Desktop V1 Integration

Read the constitution, breakdown, dependency graph, global acceptance, all handoffs, all smoke results and T06 review first.

## Objective

Integrate completed work, resolve glue/dependencies without discarding task-owned behavior, fix blocking review findings, run full validation, build the broadest possible distributables, and write `FINAL_STATUS_REPORT.md`.

## Allowed Write Boundary

- Root manifests/docs/version files and small glue changes across completed modules.
- Tests needed for integration and release acceptance.
- Workpack final report and completion updates.

## Required Checks

- Inspect git diff/status and exclude user-owned dirty changes.
- Confirm every handoff/smoke result and investigate boundary violations.
- Run full pytest, compileall, source GUI offscreen, worker E2E, synthetic demo, launcher/download tests, packaging dry-runs, and executable/installer builds when tools permit.
- Validate no public release manifest references `example_data` or `demo_checkpoint`.
- Validate public licensing, hash manifests, no unsafe pickle load and weights-only checkpoint behavior.

## Final Report

Report completed deliverables, integration fixes, exact test totals, generated artifact paths and hashes, external blockers (certificate/hosting/tooling/hardware), residual risks and follow-ups.
