# SafeSynth owner publishing runbook

> **Completed 2026-08-05:** `v1.0.0` is public and passed the final read-only
> acceptance gate. This runbook is retained as an auditable record of the exact
> owner-only publishing sequence; it is not an active checklist.

This runbook contains the **external write operations that must be performed by
the repository owner (`kuotunyu`) personally**. The preparation and verification
steps are safe to repeat. Do not paste access tokens into this repository or the
Codex chat.

## Prepared local artifacts

- GitHub source: this repository
- Dataset bundle: `${data_root}/publish/safesynth-hard-hat`
- Model bundle: `${data_root}/publish/safesynth-rtdetrv2-r18`
- Intended GitHub URL: <https://github.com/kuotunyu/SafeSynth>
- Intended dataset URL: <https://huggingface.co/datasets/steven0226/safesynth-hard-hat>
- Intended model URL: <https://huggingface.co/steven0226/safesynth-rtdetrv2-r18>

Run each section only after the controller reports that its read-only gate has
passed.

## 1. Re-check the prepared payloads

From a normal PowerShell opened in the SafeSynth repository:

```powershell
uv run python -m scripts.verify_hf_release
```

Continue only when the last line starts with `PASS:`. This check does not upload
or change either public service.

## 2. Save the reviewed local release

Still in the SafeSynth repository:

```powershell
git status --short
git add PLAN_PHASE2.md README.md docs/worklog.md instructions_for_me.md publishing scripts/prepare_hf_release.py scripts/verify_hf_release.py src/release/hf_bundle.py tests/test_hf_bundle.py docs/superpowers/plans/2026-08-04-repository-curation-history-slimming.md docs/superpowers/plans/2026-08-04-repository-curation-v5-tree-ref-safety.md
git diff --cached --check
git status --short
git commit -m "release: prepare SafeSynth public artifacts"
```

The controller will then re-check author identity, co-author trailers, secrets,
absolute paths, tests, and the exact commit before any remote is created.

## 3. Create and push the public GitHub repository

```powershell
gh auth status
gh repo create kuotunyu/SafeSynth --public --source . --remote origin --push --description "Controlled synthetic-data ablations for hard-hat detection"
gh run list --repo kuotunyu/SafeSynth --limit 5
```

Wait for CI to finish. If the newest run is still active, copy its numeric ID
from the first command and run:

```powershell
gh run watch RUN_ID --repo kuotunyu/SafeSynth --exit-status
```

Do not continue if CI is red.

## 4. Authenticate to Hugging Face

Use the current official CLI through `uvx`, then log in interactively. This runs
the CLI in an isolated environment and does not modify the SafeSynth Python
environment. The token should have write permission and should remain only in
the CLI credential store.

```powershell
uvx hf version
uvx hf auth login
uvx hf auth whoami
```

## 5. Upload the dataset and model

Create the two public repositories explicitly, then use the official `hf upload`
command. It uses Xet for large files and can be re-run after a network
interruption. The dataset command may take a while because it contains 6,152
images.

```powershell
$publishRoot = (uv run python -c "from src.data.paths import load_project_paths; print(load_project_paths().data_root / 'publish')").Trim()
$datasetBundle = Join-Path $publishRoot "safesynth-hard-hat"
$modelBundle = Join-Path $publishRoot "safesynth-rtdetrv2-r18"
$env:HF_XET_HIGH_PERFORMANCE = "1"
uvx hf repos create steven0226/safesynth-hard-hat --repo-type dataset --exist-ok
uvx hf repos create steven0226/safesynth-rtdetrv2-r18 --repo-type model --exist-ok
uvx hf upload steven0226/safesynth-hard-hat $datasetBundle . --repo-type dataset --commit-message "Release SafeSynth hard-hat dataset v1.0.0"
uvx hf upload steven0226/safesynth-rtdetrv2-r18 $modelBundle . --repo-type model --commit-message "Release SafeSynth RT-DETRv2-R18 v1.0.0"
Remove-Item Env:HF_XET_HIGH_PERFORMANCE
```

Re-running the same upload is the correct recovery action if it is interrupted;
already committed files and Xet chunks are reused.

## 6. Owner visual review

Open all three public pages and confirm:

1. GitHub shows only `kuotunyu` under Contributors.
2. GitHub Actions is green.
3. The dataset page shows CC0 1.0 and both COCO JSON files, 6,152 images,
   `records.jsonl`, and `release_manifest.json`.
4. The model page shows Apache-2.0 and loads `config.json`,
   `preprocessor_config.json`, and `model.safetensors`.
5. GitHub, dataset card, and model card link to one another.
6. No local drive path, token, private file, Test image, or training optimizer
   state is present on any public page.

## 7. Tag and create the GitHub release

Only after CI and both Hugging Face pages pass the review:

```powershell
git tag -a v1.0.0 -m "SafeSynth v1.0.0"
git push origin v1.0.0
gh release create v1.0.0 --repo kuotunyu/SafeSynth --title "SafeSynth v1.0.0" --notes-file publishing/RELEASE_NOTES_v1.0.0.md --verify-tag
```

Return the full output and the three public URLs to the controller for the final
read-only acceptance gate.
