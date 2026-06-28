"""Cadence classification shared by discovery, parsing, and validation."""


def classify_cadence(seconds, minimum=110.0, maximum=130.0):
    if seconds is None or seconds <= 0:
        return "unknown"
    if minimum <= float(seconds) <= maximum:
        return "high_cadence"
    if float(seconds) < minimum:
        return "faster_than_configured"
    return "ffi_or_long_cadence"


def is_high_cadence(seconds, minimum=110.0, maximum=130.0):
    return classify_cadence(seconds, minimum, maximum) == "high_cadence"
