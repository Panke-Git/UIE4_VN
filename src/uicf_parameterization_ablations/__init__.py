"""Isolated V16 correction-parameterization controls; never replaces V16 factories."""

VARIANTS = ("direct_residual", "without_anchor", "fixed_anchor")
FORMULAS = {
    "direct_residual": "I_c = I + predicted_residual",
    "without_anchor": "I_c = I + R * I",
    "fixed_anchor": "I_c = I + R * (I - 0.5)",
}


def version(variant: str) -> str:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown correction parameterization: {variant!r}")
    return f"uicf_parameterization_{variant}"
