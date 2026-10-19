"""Explicit finite-gradient AMP scale ceiling, with no skipped or retried updates."""
import math


def validate_scale_policy(initial, maximum):
    initial = float(initial)
    if not math.isfinite(initial) or initial <= 0:
        raise ValueError("amp_init_scale must be finite and positive")
    if maximum is not None:
        maximum = float(maximum)
        if not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("amp_max_scale must be finite and positive")
        if initial > maximum:
            raise ValueError("amp_init_scale must be <= amp_max_scale; specify both explicitly")


def cap_scale_state(state, maximum):
    """Return an independent public GradScaler state_dict, capped after a successful update."""
    result = dict(state)
    if not result or maximum is None:
        return result
    validate_scale_policy(min(float(result["scale"]), float(maximum)), maximum)
    if not math.isfinite(float(result["scale"])) or float(result["scale"]) <= 0:
        raise FloatingPointError("Saved GradScaler scale is invalid")
    if float(result["scale"]) > float(maximum):
        result["scale"] = float(maximum)
        result["_growth_tracker"] = 0
    return result


def cap_scaler(scaler, maximum):
    """Use public state_dict APIs; preserve scale/growth state on ordinary resume."""
    before = scaler.get_scale()
    state = scaler.state_dict()
    capped = cap_scale_state(state, maximum)
    if capped != state:
        scaler.load_state_dict(capped)
    return before != scaler.get_scale()
