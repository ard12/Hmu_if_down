#!/usr/bin/env python3
"""Generate IEC 62304 §8.2 compliant Model Changelog from ModelRegistry database."""

import argparse
import datetime
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hub.model_registry import ModelRegistry


def generate_changelog(db_path: str, output_path: str):
    registry = ModelRegistry(db_path=db_path)
    versions = registry.list_versions()

    lines = [
        "# IEC 62304 §8.2 — SaMD Machine Learning Model Changelog & Lineage Record",
        "",
        f"**Generated At**: {datetime.datetime.now(datetime.timezone.utc).isoformat()}",
        f"**Registry Database**: `{db_path}`",
        f"**Total Model Versions**: {len(versions)}",
        "",
        "## 1. Regulatory Change Control Statement",
        "",
        "In accordance with IEC 62304 Section 8.2 (Software Change Control) and FDA Good Machine",
        "Learning Practice (GMLP) Principle 8, this ledger provides an immutable audit trail of all",
        "machine learning model iterations, clinical safety metrics, and operational deployment states.",
        "",
        "| Version ID | Created At (UTC) | Algorithm | N Samples | Sensitivity | Specificity | Brier Score | Active | SHA-256 Hash | Notes |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    if not versions:
        lines.append("| *No models registered* | - | - | - | - | - | - | - | - | - |")
    else:
        for v in versions:
            dt_str = datetime.datetime.fromtimestamp(
                v["created_at"], tz=datetime.timezone.utc
            ).strftime("%Y-%m-%d %H:%M:%S")
            active_badge = "**ACTIVE**" if v["active"] else "Inactive"
            sha_short = v["sha256"][:12] + "..."
            v_id_short = v["version_id"][:8] + "..."
            sens_str = f"{v['sensitivity']:.2%}" if v["sensitivity"] > 0 else "N/A"
            spec_str = f"{v['specificity']:.2%}" if v["specificity"] > 0 else "N/A"
            brier_str = f"{v['brier']:.4f}" if v["brier"] > 0 else "N/A"
            notes = v.get("notes", "") or "-"
            lines.append(
                f"| `{v_id_short}` | {dt_str} | {v['algorithm']} | {v['training_n']} | {sens_str} | {spec_str} | {brier_str} | {active_badge} | `{sha_short}` | {notes} |"
            )

    lines.extend([
        "",
        "## 2. Integrity Verification",
        "",
    ])

    integrity = registry.verify_integrity()
    if integrity["ok"]:
        lines.append("- **Cryptographic Integrity**: ✅ 100% of model artifacts match recorded SHA-256 digests.")
    else:
        corrupt = ", ".join(integrity["corrupt_versions"])
        lines.append(f"- **Cryptographic Integrity**: ❌ CORRUPT ARTIFACTS DETECTED: {corrupt}")

    lines.append("")

    content = "\n".join(lines)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[OK] Generated model changelog at {output_path} ({len(versions)} versions)")


def main():
    parser = argparse.ArgumentParser(description="Generate IEC 62304 Model Changelog.")
    parser.add_argument(
        "--db", default="models/registry.db", help="Path to registry SQLite database"
    )
    parser.add_argument(
        "--output", default="docs/MODEL_CHANGELOG.md", help="Output markdown path"
    )
    args = parser.parse_args()
    generate_changelog(args.db, args.output)


if __name__ == "__main__":
    main()
