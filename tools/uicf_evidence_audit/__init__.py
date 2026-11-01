"""Read-only, CPU-only consolidation of existing UICF evidence."""

BASELINE_NAME = "v16_PlainUNet_ColorQuery_UICF_PreBackbone_seed3520_20260824_004528"
COUNTS = {"train": 3466, "validation": 385, "test": 428}
VARIANTS = ("original", "spatial_mean", "shift_xy", "channel_cycle_1",
            "shift_y", "shift_x", "channel_cycle_2", "zero_field")
METRICS = ("psnr", "ssim", "e00")
REPORT_FILES = ("EVIDENCE_CHAIN_AUDIT_CN.md", "spatial_correspondence_audit.csv",
                "functional_intervention_audit.csv", "evidence_claim_matrix.csv",
                "audit_summary.json", "source_manifest.json")
MEAN_ATOL = 1e-9


def combined_status(statuses):
    values = set(statuses)
    return "FAIL" if "FAIL" in values else (
        "UNVERIFIED" if not values or values - {"PASS"} else "PASS")
