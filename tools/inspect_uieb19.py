"""
    @Project: UIE4_VN
    @Author: paxton
    @FileName： inspect_uieb19.py
    @Date：2026/8/24 22:45
    @OS：
    @Email: None
"""
from pathlib import Path
from collections import Counter
import argparse


IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp",
    ".tif", ".tiff", ".webp"
}


def get_all_files(folder: Path):
    """递归获取目录下所有文件。"""
    if not folder.exists():
        return []

    return sorted(
        [p for p in folder.rglob("*") if p.is_file()],
        key=lambda x: str(x).lower()
    )


def get_image_files(folder: Path):
    """递归获取目录下所有图像文件。"""
    return [
        p for p in get_all_files(folder)
        if p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def relative_list(files, root):
    """转换为相对于数据集根目录的路径。"""
    return [str(p.relative_to(root)) for p in files]


def build_stem_map(files):
    """
    使用不带扩展名的文件名作为 sample ID。

    例如：
        input/123.jpg
        GT/123.png

    会认为是同一对。
    """
    mapping = {}

    for path in files:
        stem = path.stem

        if stem not in mapping:
            mapping[stem] = []

        mapping[stem].append(path)

    return mapping


def analyze_pairing(input_files, gt_files):
    """分析 input 与 GT 的配对情况。"""

    input_map = build_stem_map(input_files)
    gt_map = build_stem_map(gt_files)

    input_ids = set(input_map.keys())
    gt_ids = set(gt_map.keys())

    matched = sorted(input_ids & gt_ids)
    only_input = sorted(input_ids - gt_ids)
    only_gt = sorted(gt_ids - input_ids)

    duplicate_input = {
        k: v for k, v in input_map.items()
        if len(v) > 1
    }

    duplicate_gt = {
        k: v for k, v in gt_map.items()
        if len(v) > 1
    }

    return {
        "matched": matched,
        "only_input": only_input,
        "only_gt": only_gt,
        "duplicate_input": duplicate_input,
        "duplicate_gt": duplicate_gt,
    }


def extension_statistics(files):
    return Counter(p.suffix.lower() for p in files)


def write_section_header(f, title):
    f.write("\n")
    f.write("=" * 80 + "\n")
    f.write(title + "\n")
    f.write("=" * 80 + "\n")


def inspect_split(
    f,
    dataset_root: Path,
    split_name: str,
    input_dir: Path,
    gt_dir: Path,
):
    write_section_header(f, f"{split_name} SPLIT")

    f.write(f"Input directory : {input_dir}\n")
    f.write(f"GT directory    : {gt_dir}\n\n")

    if not input_dir.exists():
        f.write("[ERROR] Input directory does not exist.\n")

    if not gt_dir.exists():
        f.write("[ERROR] GT directory does not exist.\n")

    input_files = get_image_files(input_dir)
    gt_files = get_image_files(gt_dir)

    pairing = analyze_pairing(input_files, gt_files)

    f.write("SUMMARY\n")
    f.write("-" * 80 + "\n")
    f.write(f"Input images             : {len(input_files)}\n")
    f.write(f"GT images                : {len(gt_files)}\n")
    f.write(f"Matched sample IDs       : {len(pairing['matched'])}\n")
    f.write(f"Input without GT         : {len(pairing['only_input'])}\n")
    f.write(f"GT without input         : {len(pairing['only_gt'])}\n")
    f.write(
        f"Duplicate input IDs      : "
        f"{len(pairing['duplicate_input'])}\n"
    )
    f.write(
        f"Duplicate GT IDs         : "
        f"{len(pairing['duplicate_gt'])}\n"
    )

    f.write("\nInput extension statistics:\n")
    for ext, count in sorted(extension_statistics(input_files).items()):
        f.write(f"  {ext}: {count}\n")

    f.write("\nGT extension statistics:\n")
    for ext, count in sorted(extension_statistics(gt_files).items()):
        f.write(f"  {ext}: {count}\n")

    # ---------------------------------------------------------
    # Pairing problems
    # ---------------------------------------------------------
    f.write("\n")
    f.write("PAIRING CHECK\n")
    f.write("-" * 80 + "\n")

    if (
        not pairing["only_input"]
        and not pairing["only_gt"]
        and not pairing["duplicate_input"]
        and not pairing["duplicate_gt"]
    ):
        f.write("[OK] All samples are one-to-one paired by filename stem.\n")

    if pairing["only_input"]:
        f.write("\nInput images without corresponding GT:\n")
        for sample_id in pairing["only_input"]:
            for path in build_stem_map(input_files)[sample_id]:
                f.write(
                    f"  {path.relative_to(dataset_root)}\n"
                )

    if pairing["only_gt"]:
        f.write("\nGT images without corresponding input:\n")
        for sample_id in pairing["only_gt"]:
            for path in build_stem_map(gt_files)[sample_id]:
                f.write(
                    f"  {path.relative_to(dataset_root)}\n"
                )

    if pairing["duplicate_input"]:
        f.write("\nDuplicate input sample IDs:\n")
        for sample_id, paths in sorted(
            pairing["duplicate_input"].items()
        ):
            f.write(f"  [{sample_id}]\n")
            for path in paths:
                f.write(
                    f"    {path.relative_to(dataset_root)}\n"
                )

    if pairing["duplicate_gt"]:
        f.write("\nDuplicate GT sample IDs:\n")
        for sample_id, paths in sorted(
            pairing["duplicate_gt"].items()
        ):
            f.write(f"  [{sample_id}]\n")
            for path in paths:
                f.write(
                    f"    {path.relative_to(dataset_root)}\n"
                )

    # ---------------------------------------------------------
    # Matched pairs
    # ---------------------------------------------------------
    f.write("\n")
    f.write("MATCHED PAIRS\n")
    f.write("-" * 80 + "\n")

    input_map = build_stem_map(input_files)
    gt_map = build_stem_map(gt_files)

    for idx, sample_id in enumerate(pairing["matched"], start=1):

        # 正常情况下每个 sample_id 只有一个文件
        input_path = input_map[sample_id][0]
        gt_path = gt_map[sample_id][0]

        f.write(
            f"{idx:04d}\t"
            f"{sample_id}\t"
            f"{input_path.relative_to(dataset_root)}\t"
            f"{gt_path.relative_to(dataset_root)}\n"
        )

    # ---------------------------------------------------------
    # Complete input list
    # ---------------------------------------------------------
    f.write("\n")
    f.write("ALL INPUT FILES\n")
    f.write("-" * 80 + "\n")

    for idx, path in enumerate(input_files, start=1):
        f.write(
            f"{idx:04d}\t"
            f"{path.relative_to(dataset_root)}\n"
        )

    # ---------------------------------------------------------
    # Complete GT list
    # ---------------------------------------------------------
    f.write("\n")
    f.write("ALL GT FILES\n")
    f.write("-" * 80 + "\n")

    for idx, path in enumerate(gt_files, start=1):
        f.write(
            f"{idx:04d}\t"
            f"{path.relative_to(dataset_root)}\n"
        )

    return {
        "input": len(input_files),
        "gt": len(gt_files),
        "matched": len(pairing["matched"]),
        "only_input": len(pairing["only_input"]),
        "only_gt": len(pairing["only_gt"]),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Inspect existing UIEB19 Train/Val split."
    )

    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to UIEB19 dataset root."
    )

    parser.add_argument(
        "--output",
        type=str,
        default="UIEB19_split_report.txt",
        help="Output report path."
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    output = Path(args.output).resolve()

    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root does not exist: {root}"
        )

    train_input = root / "Train" / "input"
    train_gt = root / "Train" / "GT"

    val_input = root / "Val" / "input"
    val_gt = root / "Val" / "GT"

    with output.open("w", encoding="utf-8") as f:

        f.write("UIEB19 DATASET SPLIT REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Dataset root: {root}\n")

        train_stats = inspect_split(
            f=f,
            dataset_root=root,
            split_name="TRAIN",
            input_dir=train_input,
            gt_dir=train_gt,
        )

        val_stats = inspect_split(
            f=f,
            dataset_root=root,
            split_name="VAL",
            input_dir=val_input,
            gt_dir=val_gt,
        )

        write_section_header(f, "GLOBAL SUMMARY")

        total_input = (
            train_stats["input"]
            + val_stats["input"]
        )

        total_gt = (
            train_stats["gt"]
            + val_stats["gt"]
        )

        total_matched = (
            train_stats["matched"]
            + val_stats["matched"]
        )

        f.write(f"Train input   : {train_stats['input']}\n")
        f.write(f"Train GT      : {train_stats['gt']}\n")
        f.write(f"Train pairs   : {train_stats['matched']}\n")
        f.write("\n")

        f.write(f"Val input     : {val_stats['input']}\n")
        f.write(f"Val GT        : {val_stats['gt']}\n")
        f.write(f"Val pairs     : {val_stats['matched']}\n")
        f.write("\n")

        f.write(f"Total input   : {total_input}\n")
        f.write(f"Total GT      : {total_gt}\n")
        f.write(f"Total pairs   : {total_matched}\n")

        f.write("\n")

        if (
            train_stats["only_input"] == 0
            and train_stats["only_gt"] == 0
            and val_stats["only_input"] == 0
            and val_stats["only_gt"] == 0
        ):
            f.write(
                "[OK] No missing input/GT pairs detected.\n"
            )
        else:
            f.write(
                "[WARNING] Missing input/GT pairs detected. "
                "See split sections above.\n"
            )

    print("=" * 70)
    print("UIEB19 inspection completed.")
    print(f"Dataset root : {root}")
    print(f"Report       : {output}")
    print("-" * 70)
    print(
        f"Train: input={train_stats['input']}, "
        f"GT={train_stats['gt']}, "
        f"pairs={train_stats['matched']}"
    )
    print(
        f"Val  : input={val_stats['input']}, "
        f"GT={val_stats['gt']}, "
        f"pairs={val_stats['matched']}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()