# Supervised labeler v7 runbook

Current state: the method, split, and CPU preflight are frozen. The 48-case
audit is sealed. Do not open its image IDs or pixels before the one-shot audit.

## GPU checkpoint

Do not start while another project is using the RTX 4090. The earlier v6 smoke
used about 6.8 GiB peak VRAM, but simultaneous compute would still interfere
with the other project.

When the GPU is genuinely idle:

```powershell
uv run python -m scripts.train_supervised_labeler smoke
```

The smoke is allowed to read one training batch only. It must write
`reports/supervised_labeler_v7_smoke.json` with:

- `status: smoke_passed`;
- `validation_images_read: 0`;
- `test_images_read: 0`;
- `untouched_audit_images_read: 0`.

Only after that evidence passes:

```powershell
uv run python -m scripts.train_supervised_labeler train
```

Training and epoch-by-epoch calibration use Train history only. The runner
opens the sealed 48-image audit once, after the final calibration-selected
checkpoint is fixed. It writes exact model/GT boxes to
`reports/supervised_labeler_v7_audit_evidence.json`; the review renderer uses
that sidecar rather than trying to recover boxes from colored pixels.

If the numeric gate passes:

```powershell
uv run python -m scripts.render_supervised_labeler_review
uv run python -m scripts.render_supervised_labeler_review_separated
```

The three separated pages show, left to right, dataset GT in green, model boxes
in magenta, and both overlaid. Generation remains locked until `kuotunyu`
reviews every case and reports zero problems. A numeric pass alone is not
approval.

After the owner states the decision, record it without editing JSON by hand:

```powershell
uv run python -m scripts.record_supervised_labeler_v7_review `
  --decision approve `
  --reviewed-on YYYY-MM-DD `
  --problem-cells "" `
  --note "All 48 cases reviewed."
```

For rejection, use `--decision reject` and supply the comma-separated problem
cells. The recorder verifies the checkpoint, split, raw figure, six review
pages, and exact box sidecar before writing owner-only evidence.
