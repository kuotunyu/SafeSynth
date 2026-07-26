import json
from pathlib import Path

import pytest
from PIL import Image

from src.data.voc_to_coco import (
    DataInvariantError,
    convert_voc_dataset,
    detect_coordinate_offset,
    discover_pairs,
    verify_coco_schema,
    write_canonical_json,
)


def write_example(
    root: Path,
    stem: str,
    *,
    label: str = "Helmet",
    bbox: tuple[str, str, str, str] = ("0", "1", "12", "15"),
    xml_size: tuple[int, int] = (20, 20),
    difficult: str = "0",
) -> None:
    image_dir = root / "images"
    annotation_dir = root / "annotations"
    image_dir.mkdir(parents=True, exist_ok=True)
    annotation_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 20), "white").save(image_dir / f"{stem}.png")
    xmin, ymin, xmax, ymax = bbox
    (annotation_dir / f"{stem}.xml").write_text(
        f"""<annotation>
  <size><width>{xml_size[0]}</width><height>{xml_size[1]}</height></size>
  <object>
    <name>{label}</name>
    <truncated>1</truncated>
    <difficult>{difficult}</difficult>
    <bndbox>
      <xmin>{xmin}</xmin><ymin>{ymin}</ymin>
      <xmax>{xmax}</xmax><ymax>{ymax}</ymax>
    </bndbox>
  </object>
</annotation>
""",
        encoding="utf-8",
        newline="\n",
    )


def convert_fixture(root: Path):
    return convert_voc_dataset(
        root,
        classes=("helmet", "head", "person"),
        kaggle_handle="owner/data",
        kaggle_version=1,
        archive_sha256="a" * 64,
    )


def test_detect_coordinate_offset_records_unexpected_minimum() -> None:
    assert detect_coordinate_offset(0)[0] == 0
    assert detect_coordinate_offset(1)[0] == 1
    offset, message = detect_coordinate_offset(2)
    assert offset == 0
    assert "unexpected" in message


def test_discover_pairs_uses_stems_not_enumeration_order(tmp_path: Path) -> None:
    write_example(tmp_path, "b")
    write_example(tmp_path, "a")

    pairs = discover_pairs(tmp_path)

    assert [xml.stem for xml, _ in pairs] == ["a", "b"]


def test_discover_pairs_reports_symmetric_difference(tmp_path: Path) -> None:
    write_example(tmp_path, "paired")
    Image.new("RGB", (20, 20), "white").save(tmp_path / "images" / "orphan.png")

    with pytest.raises(DataInvariantError, match="Missing XML"):
        discover_pairs(tmp_path)


def test_conversion_normalizes_labels_flags_and_paths(tmp_path: Path) -> None:
    write_example(tmp_path, "sample", difficult="1")

    coco, stats, audit = convert_fixture(tmp_path)

    assert audit["coordinate_offset"] == 0
    assert coco["images"][0]["file_name"] == "images/sample.png"
    annotation = coco["annotations"][0]
    assert annotation["category_id"] == 1
    assert annotation["bbox"] == [0.0, 1.0, 12.0, 14.0]
    assert annotation["area"] == 168.0
    assert annotation["iscrowd"] == 0
    assert annotation["difficult"] == 1
    assert annotation["truncated"] == 1
    assert stats.label_counts == {"helmet": 1}
    verify_coco_schema(coco)


def test_one_based_conversion_only_offsets_minima(tmp_path: Path) -> None:
    write_example(tmp_path, "sample", bbox=("1", "1", "20", "20"))

    coco, _, audit = convert_fixture(tmp_path)

    assert audit["coordinate_offset"] == 1
    assert coco["annotations"][0]["bbox"] == [0.0, 0.0, 20.0, 20.0]


def test_conversion_records_round_swap_clip_and_size_mismatch(tmp_path: Path) -> None:
    write_example(
        tmp_path,
        "sample",
        bbox=("21.2", "-2.4", "2.2", "12.6"),
        xml_size=(19, 20),
    )

    coco, stats, _ = convert_fixture(tmp_path)

    assert coco["annotations"][0]["bbox"] == [2.0, 0.0, 18.0, 13.0]
    assert stats.corrections["float_coordinates_rounded"] == 1
    assert stats.corrections["x_coordinates_swapped"] == 1
    assert stats.corrections["boxes_clipped_to_image"] == 1
    assert stats.xml_size_mismatches == ["sample.xml"]


def test_conversion_rejects_unknown_labels(tmp_path: Path) -> None:
    write_example(tmp_path, "sample", label="vest")

    with pytest.raises(DataInvariantError, match="Unknown labels"):
        convert_fixture(tmp_path)


def test_canonical_json_is_stable_and_ascii(tmp_path: Path) -> None:
    destination = tmp_path / "manifest.json"
    value = {"z": "安全帽", "a": [2, 1]}

    write_canonical_json(destination, value)
    first = destination.read_bytes()
    write_canonical_json(destination, value)

    assert destination.read_bytes() == first
    assert first == b'{"a":[2,1],"z":"\\u5b89\\u5168\\u5e3d"}\n'
    assert json.loads(first) == value
