"""Freeze the four prompt-only v10 cases without loading either GPU model."""

from __future__ import annotations

import json

from src.data.paths import PROJECT_ROOT
from src.synthetic.grounded_labeler import load_whole_image_config
from src.synthetic.whole_image import diagnostic_manifest

JSON_PATH = PROJECT_ROOT / "reports" / "whole_image_v10_preregistration.json"
MARKDOWN_PATH = PROJECT_ROOT / "reports" / "whole_image_v10_preregistration.md"


def main() -> None:
    config = load_whole_image_config()
    manifest = diagnostic_manifest(config)
    status = (
        "blocked because the zero-shot labeler failed its Train-only audit"
        if config["status"] == "zero_shot_labeler_audit_failed"
        else "waiting for labeler audit and kuotunyu approval"
    )
    JSON_PATH.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "# Whole-image v10 diagnostic preregistration",
        "",
        f"- Manifest SHA-256: `{manifest['manifest_sha256']}`",
        "- Cases: **4**",
        "- Validation/Test images read: **0 / 0**",
        "- FLUX images generated: **0**",
        f"- Status: **{status}**",
        "",
        "| Case | Scenario | Seed | Frozen prompt |",
        "|---:|---|---:|---|",
    ]
    for case in manifest["cases"]:
        prompt = str(case["prompt"]).replace("|", "\\|")
        lines.append(
            f"| {int(case['case_index']):02d} | {case['scenario']} | "
            f"{case['seed']} | {prompt} |"
        )
    lines.extend(
        [
            "",
            "No prompt or seed may be changed after output inspection.",
            "",
        ]
    )
    MARKDOWN_PATH.write_text(
        "\n".join(lines),
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
