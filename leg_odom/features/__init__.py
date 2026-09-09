"""The shared 5D instant feature vector used by both contact detectors."""

from leg_odom.features.instant_spec import (
    ALLOWED_INSTANT_FEATURE_FIELDS,
    DEFAULT_INSTANT_FEATURE_FIELDS,
    INSTANT_FEATURE_SPEC_VERSION,
    InstantFeatureSpec,
    build_timeline_features_for_leg,
    flatten_history_window,
    instant_vector_from_step,
    parse_instant_feature_fields,
    sliding_windows_flat,
)

__all__ = [
    "ALLOWED_INSTANT_FEATURE_FIELDS",
    "DEFAULT_INSTANT_FEATURE_FIELDS",
    "INSTANT_FEATURE_SPEC_VERSION",
    "InstantFeatureSpec",
    "build_timeline_features_for_leg",
    "flatten_history_window",
    "instant_vector_from_step",
    "parse_instant_feature_fields",
    "sliding_windows_flat",
]
