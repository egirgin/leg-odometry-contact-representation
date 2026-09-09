"""Contact detection: the shared ABC plus the two detectors this repo implements."""

from __future__ import annotations

from leg_odom.contact.base import BaseContactDetector, ContactDetectorStepInput, ContactEstimate

# hmm_gmm imports leg_odom.features, which imports this module for ContactDetectorStepInput --
# load it lazily so `import leg_odom.contact` alone doesn't hit that cycle.
_LAZY = frozenset({"GmmHmmContactDetector", "build_gmm_hmm_detectors_from_cfg"})


def __getattr__(name: str):
    if name in _LAZY:
        from leg_odom.contact import hmm_gmm

        return getattr(hmm_gmm, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseContactDetector",
    "ContactDetectorStepInput",
    "ContactEstimate",
    "GmmHmmContactDetector",
    "build_gmm_hmm_detectors_from_cfg",
]
