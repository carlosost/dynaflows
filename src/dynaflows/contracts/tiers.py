"""Model tiers. ADR-006: tier is assigned by calls-per-run multiplier."""

from __future__ import annotations

from enum import StrEnum


class Tier(StrEnum):
    """A cost/capability band, not a model.

    The mapping from tier to concrete model chain lives in config/models.toml
    so that swapping a model is a config edit, never a code change.
    """

    SMALL = "small"
    MID = "mid"
    MID_HIGH = "mid_high"
    FRONTIER = "frontier"
