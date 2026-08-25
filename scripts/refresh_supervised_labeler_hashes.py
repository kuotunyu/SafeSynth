"""Refresh the allowlisted frozen-evidence hashes changed by path hygiene.

This utility is intentionally limited to the 2026-08-25 portable-path scrub.
It derives the dependency graph from the committed ``HEAD`` values, reuses the
repository's canonical mapping hash, and never reads checkpoints or evaluation
data.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import heapq
import json
import re
import subprocess
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from src.data.paths import PROJECT_ROOT
from src.release.public_paths import (
    LOCAL_ONLY_NOT_PUBLISHED,
    runtime_artifact_reference,
)
from src.synthetic.whole_image import canonical_mapping_sha256

HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")
PATH_FIELD_NAMES = frozenset(
    {
        "checkpoint_path",
        "data_root",
        "first_summary",
        "hf_home",
        "model_dir",
        "output_dir",
        "path",
        "run_dir",
        "second_summary",
        "source_run",
    }
)

# Exact structured files whose non-hash working-tree change is the approved
# portable-path scrub. Files outside this list cannot seed a refresh.
PRODUCTION_SCRUBBED_FILES = (
    "configs/paths.yaml",
    "configs/supervised_labeler_v12_gt_review.yaml",
    "configs/supervised_labeler_v13.yaml",
    "configs/supervised_labeler_v14.yaml",
    "configs/supervised_labeler_v15.yaml",
    "configs/supervised_labeler_v16.yaml",
    "configs/supervised_labeler_v17.yaml",
    "configs/supervised_labeler_v18.yaml",
    "configs/supervised_labeler_v19.yaml",
    "configs/supervised_labeler_v20.yaml",
    "configs/supervised_labeler_v21.yaml",
    "configs/supervised_labeler_v22.yaml",
    "configs/supervised_labeler_v23.yaml",
    "reports/colab_package.json",
    "reports/composition_reproducibility.json",
    "reports/filter_ledger.json",
    "reports/generative_model_preflight.json",
    "reports/grounded_labeler_model.json",
    "reports/h4_ablation_no_hard_negative.json",
    "reports/h4_artifact_gate.json",
    "reports/h4_artifact_gate_m13.json",
    "reports/h4_context_matched.json",
    "reports/h4_context_replacement.json",
    "reports/h4_guarded_input_preflight.json",
    "reports/h4_guarded_input_preflight_v2_failed.json",
    "reports/h4_guarded_input_preflight_v3_failed.json",
    "reports/h4_guarded_input_preflight_v4_failed.json",
    "reports/h4_paired_person_input_preflight_seed20260801_geometry_failed.json",
    "reports/h4_paired_person_input_preflight_seed20260802.json",
    "reports/h4_poisson_gate.json",
    "reports/h4_source_pair.json",
    "reports/supervised_labeler_model.json",
    "reports/supervised_labeler_training.json",
    "reports/supervised_labeler_v10_training.json",
    "reports/supervised_labeler_v11_training.json",
    "reports/supervised_labeler_v12_model_audit.json",
    "reports/supervised_labeler_v13_model_review_manifest.json",
    "reports/supervised_labeler_v13_review_diagnosis.json",
    "reports/supervised_labeler_v13_training.json",
    "reports/supervised_labeler_v14_model_review_manifest.json",
    "reports/supervised_labeler_v14_review_diagnosis.json",
    "reports/supervised_labeler_v14_training.json",
    "reports/supervised_labeler_v15_model_review_manifest.json",
    "reports/supervised_labeler_v15_training.json",
    "reports/supervised_labeler_v16_training.json",
    "reports/supervised_labeler_v17_model_review_manifest.json",
    "reports/supervised_labeler_v17_training.json",
    "reports/supervised_labeler_v18_model_review_manifest.json",
    "reports/supervised_labeler_v18_training.json",
    "reports/supervised_labeler_v19_model_review_manifest.json",
    "reports/supervised_labeler_v19_training.json",
    "reports/supervised_labeler_v20_training.json",
    "reports/supervised_labeler_v21_training.json",
    "reports/supervised_labeler_v22_model_review_manifest.json",
    "reports/supervised_labeler_v22_training.json",
    "reports/supervised_labeler_v23_model_review_manifest.json",
    "reports/supervised_labeler_v23_training.json",
    "reports/supervised_labeler_v2_training.json",
    "reports/supervised_labeler_v3_training.json",
    "reports/supervised_labeler_v4_training.json",
    "reports/supervised_labeler_v5_model.json",
    "reports/supervised_labeler_v5_training.json",
    "reports/supervised_labeler_v6_model.json",
    "reports/supervised_labeler_v6_training.json",
    "reports/supervised_labeler_v7_training.json",
    "reports/supervised_labeler_v8_training.json",
    "reports/supervised_labeler_v9_model.json",
    "reports/supervised_labeler_v9_training.json",
    "results/predictions_index.json",
    "results/rfdetr_predictions_index.json",
)

# Exact transitive reports/configs allowed to receive regenerated hashes.
PRODUCTION_UPPER_FILES = (
    "configs/supervised_labeler_v10.yaml",
    "configs/supervised_labeler_v11.yaml",
    "configs/supervised_labeler_v18_gt_review.yaml",
    "configs/supervised_labeler_v19_gt_review.yaml",
    "configs/supervised_labeler_v20_gt_review.yaml",
    "configs/supervised_labeler_v21_gt_review.yaml",
    "configs/supervised_labeler_v22_gt_review.yaml",
    "configs/supervised_labeler_v23_gt_review.yaml",
    "reports/supervised_labeler_v12_model_human_review.json",
    "reports/supervised_labeler_v12_review_diagnosis.json",
    "reports/supervised_labeler_v13_model_human_review.json",
    "reports/supervised_labeler_v14_model_human_review.json",
    "reports/supervised_labeler_v15_model_human_review.json",
    "reports/supervised_labeler_v15_review_diagnosis.json",
    "reports/supervised_labeler_v16_audit_diagnosis.json",
    "reports/supervised_labeler_v17_model_human_review.json",
    "reports/supervised_labeler_v17_review_diagnosis.json",
    "reports/supervised_labeler_v18_model_human_review.json",
    "reports/supervised_labeler_v18_review_diagnosis.json",
    "reports/supervised_labeler_v19_model_human_review.json",
    "reports/supervised_labeler_v19_review_diagnosis.json",
    "reports/supervised_labeler_v20_numeric_failure_diagnosis.json",
    "reports/supervised_labeler_v21_numeric_failure_diagnosis.json",
    "reports/supervised_labeler_v22_model_human_review.json",
    "reports/supervised_labeler_v22_review_diagnosis.json",
)

PRODUCTION_RUNTIME_REFERENCE_FIELDS = {
    **{
        path: frozenset({"checkpoint_path"})
        for path in PRODUCTION_SCRUBBED_FILES
        if path.startswith("reports/supervised_labeler") and path.endswith(".json")
    },
    "results/predictions_index.json": frozenset({"path"}),
    "results/rfdetr_predictions_index.json": frozenset({"path"}),
}


class HashRefreshError(RuntimeError):
    """Raised when the frozen evidence does not match the allowlisted schema."""


FieldPath = tuple[str | int, ...]


@dataclass(frozen=True)
class HashChange:
    """One deterministic hash-field update."""

    path: str
    field: str
    kind: str
    old: str
    new: str


@dataclass(frozen=True)
class RefreshResult:
    """Summary of a dry-run or applied refresh."""

    changed_files: tuple[str, ...]
    changes: tuple[HashChange, ...]
    dry_run: bool

    @property
    def change_count(self) -> int:
        return len(self.changes)


@dataclass(frozen=True)
class RuntimeReferenceResult:
    """Files whose reloadable marker was restored to a portable reference."""

    changed_files: tuple[str, ...]
    dry_run: bool


@dataclass
class _Document:
    path: str
    base_bytes: bytes
    current_bytes: bytes
    base_value: Any
    current_value: Any


@dataclass(frozen=True)
class _DigestSpec:
    kind: str
    digest: str
    self_field: FieldPath | None = None


@dataclass(frozen=True)
class _Edge:
    source: str
    target: str
    target_field: FieldPath
    kind: str
    self_field: FieldPath | None


def _git(root: Path, *arguments: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise HashRefreshError(f"git {' '.join(arguments)} failed: {detail}")
    return result.stdout


def _load_document(path: str, raw: bytes) -> Any:
    try:
        text = raw.decode("utf-8")
        if path.endswith(".json"):
            return json.loads(text)
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(text)
    except (UnicodeDecodeError, json.JSONDecodeError, yaml.YAMLError) as exc:
        raise HashRefreshError(f"cannot parse {path}: {exc}") from exc
    raise HashRefreshError(f"unsupported structured evidence file: {path}")


def _walk_scalars(value: Any, trail: FieldPath = ()) -> Iterable[tuple[FieldPath, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk_scalars(child, trail + (str(key),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_scalars(child, trail + (index,))
    else:
        yield trail, value


def _value_at(value: Any, field: FieldPath) -> Any:
    current = value
    for component in field:
        current = current[component]
    return current


def _set_value(value: Any, field: FieldPath, replacement: Any) -> None:
    current = value
    for component in field[:-1]:
        current = current[component]
    current[field[-1]] = replacement


def _field_name(field: FieldPath) -> str:
    return str(field[-1]) if field else ""


def _display_field(field: FieldPath) -> str:
    return ".".join(str(component) for component in field)


def _mapping_changes(
    before: Any,
    after: Any,
    trail: FieldPath = (),
) -> list[tuple[FieldPath, Any, Any]]:
    if type(before) is not type(after):
        return [(trail, before, after)]
    if isinstance(before, Mapping):
        changes: list[tuple[FieldPath, Any, Any]] = []
        keys = sorted(set(before) | set(after), key=lambda item: str(item))
        for key in keys:
            field = trail + (str(key),)
            if key not in before or key not in after:
                changes.append((field, before.get(key), after.get(key)))
            else:
                changes.extend(_mapping_changes(before[key], after[key], field))
        return changes
    if isinstance(before, list):
        if len(before) != len(after):
            return [(trail + ("<length>",), len(before), len(after))]
        changes = []
        for index, (left, right) in enumerate(zip(before, after, strict=True)):
            changes.extend(_mapping_changes(left, right, trail + (index,)))
        return changes
    return [] if before == after else [(trail, before, after)]


def _restore_runtime_value(
    source: Any,
    current: Any,
    *,
    allowed_fields: frozenset[str],
    project_root: Path,
    source_data_root: Path,
    trail: FieldPath = (),
) -> Any:
    if type(source) is not type(current):
        raise HashRefreshError(
            f"runtime reference schema changed at {_display_field(trail)}"
        )
    if isinstance(source, Mapping):
        if set(source) != set(current):
            raise HashRefreshError(
                f"runtime reference schema keys changed at {_display_field(trail)}"
            )
        return {
            key: _restore_runtime_value(
                source[key],
                current[key],
                allowed_fields=allowed_fields,
                project_root=project_root,
                source_data_root=source_data_root,
                trail=trail + (str(key),),
            )
            for key in current
        }
    if isinstance(source, list):
        if len(source) != len(current):
            raise HashRefreshError(
                f"runtime reference schema length changed at {_display_field(trail)}"
            )
        return [
            _restore_runtime_value(
                left,
                right,
                allowed_fields=allowed_fields,
                project_root=project_root,
                source_data_root=source_data_root,
                trail=trail + (index,),
            )
            for index, (left, right) in enumerate(zip(source, current, strict=True))
        ]
    if (
        _field_name(trail) in allowed_fields
        and current == LOCAL_ONLY_NOT_PUBLISHED
    ):
        if not isinstance(source, str):
            raise HashRefreshError(
                f"runtime reference schema is not a string at {_display_field(trail)}"
            )
        return runtime_artifact_reference(
            source,
            project_root=project_root,
            data_root=source_data_root,
        )
    return current


def restore_runtime_references(
    project_root: Path,
    *,
    source_revision: str,
    runtime_reference_fields: Mapping[str, frozenset[str]],
    dry_run: bool,
) -> RuntimeReferenceResult:
    """Restore exact reloadable fields from pre-scrub evidence, or fail closed."""

    root = project_root.resolve()
    source_paths = _load_document(
        "configs/paths.yaml",
        _git(root, "show", f"{source_revision}:configs/paths.yaml"),
    )
    if not isinstance(source_paths, Mapping):
        raise HashRefreshError("source configs/paths.yaml schema is not a mapping")
    raw_data_root = source_paths.get("data_root")
    if not isinstance(raw_data_root, str) or not Path(raw_data_root).is_absolute():
        raise HashRefreshError("source configs/paths.yaml has no absolute data_root")
    source_data_root = Path(raw_data_root).resolve()

    changed: list[str] = []
    rendered: dict[str, bytes] = {}
    for path in sorted(runtime_reference_fields):
        target = root / path
        if not target.is_file() or not path.endswith(".json"):
            raise HashRefreshError(f"runtime reference file is missing or unsupported: {path}")
        source_value = _load_document(
            path,
            _git(root, "show", f"{source_revision}:{path}"),
        )
        current_bytes = target.read_bytes()
        current_value = _load_document(path, current_bytes)
        restored = _restore_runtime_value(
            source_value,
            current_value,
            allowed_fields=runtime_reference_fields[path],
            project_root=root,
            source_data_root=source_data_root,
        )
        newline = "\r\n" if b"\r\n" in current_bytes else "\n"
        restored_bytes = _json_bytes(restored, newline=newline)
        if restored_bytes != current_bytes:
            changed.append(path)
            rendered[path] = restored_bytes

    if not dry_run:
        for path in changed:
            (root / path).write_bytes(rendered[path])
    return RuntimeReferenceResult(changed_files=tuple(changed), dry_run=dry_run)


def _is_hash_field(field: FieldPath) -> bool:
    name = _field_name(field).casefold()
    return "sha" in name or "hash" in name


def _verify_semantic_invariants(documents: Mapping[str, _Document]) -> None:
    for path, document in documents.items():
        for field, _, _ in _mapping_changes(
            document.base_value,
            document.current_value,
        ):
            if _field_name(field) in PATH_FIELD_NAMES or _is_hash_field(field):
                continue
            raise HashRefreshError(
                f"{path}:{_display_field(field)} changed a non-path/hash field"
            )


def _json_bytes(value: Any, *, newline: str = "\n") -> bytes:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if newline != "\n":
        text = text.replace("\n", newline)
    return text.encode("utf-8")


def _self_hash_fields(value: Any) -> tuple[FieldPath, ...]:
    if not isinstance(value, Mapping):
        return ()
    fields = []
    for key, embedded in value.items():
        field = (str(key),)
        if not _is_hash_field(field) or not isinstance(embedded, str):
            continue
        if HEX_SHA256.fullmatch(embedded) is None:
            continue
        canonical = dict(value)
        canonical.pop(key)
        if canonical_mapping_sha256(canonical) == embedded:
            fields.append(field)
    return tuple(fields)


def _digest_specs(document: _Document) -> tuple[_DigestSpec, ...]:
    specs = [
        _DigestSpec(
            kind="file-byte",
            digest=hashlib.sha256(document.base_bytes).hexdigest(),
        )
    ]
    if isinstance(document.base_value, Mapping):
        specs.append(
            _DigestSpec(
                kind="canonical-full",
                digest=canonical_mapping_sha256(document.base_value),
            )
        )
        for field in _self_hash_fields(document.base_value):
            specs.append(
                _DigestSpec(
                    kind="canonical-without-self",
                    digest=str(_value_at(document.base_value, field)),
                    self_field=field,
                )
            )
    unique = {
        (spec.kind, spec.digest, spec.self_field): spec for spec in specs
    }
    return tuple(unique.values())


def _load_all_documents(root: Path, base_revision: str) -> dict[str, _Document]:
    tracked = _git(root, "ls-files", "-z").split(b"\0")
    documents = {}
    for raw_path in tracked:
        if not raw_path:
            continue
        path = raw_path.decode("utf-8")
        if not path.endswith((".json", ".yaml", ".yml")):
            continue
        if not path.startswith(("configs/", "reports/", "results/")):
            continue
        target = root / path
        if not target.is_file():
            raise HashRefreshError(f"tracked evidence file is missing: {path}")
        base_bytes = _git(root, "show", f"{base_revision}:{path}")
        current_bytes = target.read_bytes()
        documents[path] = _Document(
            path=path,
            base_bytes=base_bytes,
            current_bytes=current_bytes,
            base_value=_load_document(path, base_bytes),
            current_value=_load_document(path, current_bytes),
        )
    return documents


def _build_graph(
    documents: Mapping[str, _Document],
    scrubbed_files: Sequence[str],
    allowed_upper_files: Sequence[str],
) -> tuple[set[str], tuple[_Edge, ...]]:
    scrubbed = set(scrubbed_files)
    upper = set(allowed_upper_files)
    expected = scrubbed | upper
    missing = expected - documents.keys()
    if missing:
        raise HashRefreshError(
            "allowlisted evidence files are missing: " + ", ".join(sorted(missing))
        )

    occurrences: dict[str, list[tuple[str, FieldPath]]] = defaultdict(list)
    for path, document in documents.items():
        for field, value in _walk_scalars(document.base_value):
            if isinstance(value, str) and HEX_SHA256.fullmatch(value):
                occurrences[value].append((path, field))

    affected = set(scrubbed)
    queue = deque(sorted(scrubbed))
    edges: set[_Edge] = set()
    while queue:
        source = queue.popleft()
        for spec in _digest_specs(documents[source]):
            for target, target_field in occurrences.get(spec.digest, []):
                if target == source and target_field == spec.self_field:
                    continue
                if target not in expected or not _is_hash_field(target_field):
                    raise HashRefreshError(
                        "unknown hash reference "
                        f"{target}:{_display_field(target_field)} -> {source}"
                    )
                edge = _Edge(
                    source=source,
                    target=target,
                    target_field=target_field,
                    kind=spec.kind,
                    self_field=spec.self_field,
                )
                if edge in edges:
                    continue
                edges.add(edge)
                if target not in affected:
                    affected.add(target)
                    queue.append(target)

    discovered_upper = affected - scrubbed
    if discovered_upper != upper:
        missing_edges = upper - discovered_upper
        unexpected = discovered_upper - upper
        details = []
        if missing_edges:
            details.append("unreachable allowlist=" + ",".join(sorted(missing_edges)))
        if unexpected:
            details.append("unexpected=" + ",".join(sorted(unexpected)))
        raise HashRefreshError("hash dependency inventory changed: " + "; ".join(details))
    return affected, tuple(
        sorted(
            edges,
            key=lambda edge: (
                edge.source,
                edge.target,
                _display_field(edge.target_field),
                edge.kind,
            ),
        )
    )


def _topological_order(affected: set[str], edges: Sequence[_Edge]) -> tuple[str, ...]:
    outgoing: dict[str, set[str]] = defaultdict(set)
    indegree = {path: 0 for path in affected}
    for edge in edges:
        if edge.target in outgoing[edge.source]:
            continue
        outgoing[edge.source].add(edge.target)
        indegree[edge.target] += 1
    ready = [path for path, degree in indegree.items() if degree == 0]
    heapq.heapify(ready)
    ordered = []
    while ready:
        source = heapq.heappop(ready)
        ordered.append(source)
        for target in sorted(outgoing[source]):
            indegree[target] -= 1
            if indegree[target] == 0:
                heapq.heappush(ready, target)
    if len(ordered) != len(affected):
        raise HashRefreshError("hash dependency graph contains a cycle")
    return tuple(ordered)


def _yaml_node_at(node: Node, field: FieldPath) -> Node:
    current = node
    for component in field:
        if isinstance(current, MappingNode):
            match = None
            for key_node, value_node in current.value:
                if isinstance(key_node, ScalarNode) and key_node.value == str(component):
                    match = value_node
                    break
            if match is None:
                raise HashRefreshError(
                    f"YAML field not found while rendering: {_display_field(field)}"
                )
            current = match
        elif isinstance(current, SequenceNode) and isinstance(component, int):
            try:
                current = current.value[component]
            except IndexError as exc:
                raise HashRefreshError(
                    f"YAML index not found while rendering: {_display_field(field)}"
                ) from exc
        else:
            raise HashRefreshError(
                f"unknown YAML schema at {_display_field(field)}"
            )
    return current


def _render_yaml_hash_changes(
    original: bytes,
    before: Any,
    after: Any,
) -> bytes:
    text = original.decode("utf-8")
    root_node = yaml.compose(text)
    if root_node is None:
        raise HashRefreshError("cannot render an empty YAML document")
    replacements = []
    for field, old, new in _mapping_changes(before, after):
        if not _is_hash_field(field):
            raise HashRefreshError(
                f"refusing non-hash YAML rewrite at {_display_field(field)}"
            )
        if not isinstance(old, str) or not isinstance(new, str):
            raise HashRefreshError(
                f"hash field is not a string at {_display_field(field)}"
            )
        node = _yaml_node_at(root_node, field)
        start = node.start_mark.index
        end = node.end_mark.index
        segment = text[start:end]
        if segment.count(old) != 1:
            raise HashRefreshError(
                f"cannot locate exact YAML scalar at {_display_field(field)}"
            )
        replacements.append((start, end, segment.replace(old, new, 1)))
    for start, end, replacement in sorted(replacements, reverse=True):
        text = text[:start] + replacement + text[end:]
    return text.encode("utf-8")


def _render_document(document: _Document, value: Any) -> bytes:
    if document.path.endswith(".json"):
        newline = "\r\n" if b"\r\n" in document.current_bytes else "\n"
        return _json_bytes(value, newline=newline)
    return _render_yaml_hash_changes(
        document.current_bytes,
        document.current_value,
        value,
    )


def refresh_hash_chain(
    project_root: Path,
    *,
    scrubbed_files: Sequence[str],
    allowed_upper_files: Sequence[str],
    dry_run: bool,
    base_revision: str = "HEAD",
) -> RefreshResult:
    """Refresh one allowlisted path-scrub hash chain or fail closed."""

    root = project_root.resolve()
    documents = _load_all_documents(root, base_revision)
    allowlisted = set(scrubbed_files) | set(allowed_upper_files)
    _verify_semantic_invariants(
        {path: documents[path] for path in sorted(allowlisted)}
    )
    affected, edges = _build_graph(
        documents,
        scrubbed_files,
        allowed_upper_files,
    )
    order = _topological_order(affected, edges)
    outgoing: dict[str, list[_Edge]] = defaultdict(list)
    for edge in edges:
        outgoing[edge.source].append(edge)

    values = {
        path: copy.deepcopy(documents[path].current_value) for path in affected
    }
    rendered = {path: documents[path].current_bytes for path in affected}
    changes: list[HashChange] = []
    updated_fields: dict[tuple[str, FieldPath], str] = {}

    def replace_hash(
        path: str,
        field: FieldPath,
        desired: str,
        kind: str,
    ) -> None:
        current = _value_at(values[path], field)
        if not isinstance(current, str) or HEX_SHA256.fullmatch(current) is None:
            raise HashRefreshError(
                f"invalid SHA-256 scalar at {path}:{_display_field(field)}"
            )
        previous = updated_fields.get((path, field))
        if previous is not None and previous != desired:
            raise HashRefreshError(
                f"conflicting hash sources for {path}:{_display_field(field)}"
            )
        updated_fields[(path, field)] = desired
        if current == desired:
            return
        _set_value(values[path], field, desired)
        changes.append(
            HashChange(
                path=path,
                field=_display_field(field),
                kind=kind,
                old=current,
                new=desired,
            )
        )

    for source in order:
        for self_field in _self_hash_fields(documents[source].base_value):
            canonical = dict(values[source])
            canonical.pop(str(self_field[0]))
            replace_hash(
                source,
                self_field,
                canonical_mapping_sha256(canonical),
                "canonical-without-self",
            )
        rendered[source] = _render_document(documents[source], values[source])
        for edge in outgoing[source]:
            if edge.kind == "file-byte":
                desired = hashlib.sha256(rendered[source]).hexdigest()
            elif edge.kind == "canonical-full":
                desired = canonical_mapping_sha256(values[source])
            elif edge.kind == "canonical-without-self" and edge.self_field:
                desired = str(_value_at(values[source], edge.self_field))
            else:
                raise HashRefreshError(
                    f"unknown digest kind for {source}: {edge.kind}"
                )
            replace_hash(
                edge.target,
                edge.target_field,
                desired,
                edge.kind,
            )

    changed_files = tuple(
        path
        for path in order
        if _render_document(documents[path], values[path])
        != documents[path].current_bytes
    )
    if not dry_run:
        for path in changed_files:
            (root / path).write_bytes(_render_document(documents[path], values[path]))
    return RefreshResult(
        changed_files=changed_files,
        changes=tuple(changes),
        dry_run=dry_run,
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report deterministic changes without writing files",
    )
    parser.add_argument(
        "--restore-runtime-references-from",
        default=None,
        metavar="REVISION",
        help="restore allowlisted reloadable references from a pre-scrub revision",
    )
    parser.add_argument(
        "--base-revision",
        default="HEAD",
        help="committed revision containing the pre-scrub evidence",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="repository root (defaults to the current SafeSynth checkout)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parse_args(argv)
    try:
        if arguments.restore_runtime_references_from is not None:
            restored = restore_runtime_references(
                arguments.project_root,
                source_revision=arguments.restore_runtime_references_from,
                runtime_reference_fields=PRODUCTION_RUNTIME_REFERENCE_FIELDS,
                dry_run=arguments.dry_run,
            )
            if restored.changed_files:
                action = "would restore" if restored.dry_run else "restored"
                for path in restored.changed_files:
                    print(f"runtime-reference: {action} {path}")
            else:
                print("runtime-reference: no changes")
            return 0
        result = refresh_hash_chain(
            arguments.project_root,
            scrubbed_files=PRODUCTION_SCRUBBED_FILES,
            allowed_upper_files=PRODUCTION_UPPER_FILES,
            dry_run=arguments.dry_run,
            base_revision=arguments.base_revision,
        )
    except HashRefreshError as exc:
        print(f"hash-refresh: FAIL: {exc}")
        return 1
    action = "would update" if result.dry_run else "updated"
    for change in result.changes:
        print(
            f"{action} {change.path}:{change.field} "
            f"[{change.kind}] {change.old} -> {change.new}"
        )
    if result.change_count == 0:
        print("hash-refresh: no changes")
    else:
        print(
            f"hash-refresh: files={len(result.changed_files)} "
            f"changes={result.change_count} dry_run={result.dry_run}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
