"""Noise generation, heightmaps, mountains, and region partitioning."""

from .noise import (
    fbm,
    ridged_noise,
    domain_warp,
    gradient_mask,
    terrace,
    normalize as noise_normalize,
    remap,
    fbm_3d,
    ridged_noise_3d,
)
from .heightmap import (
    sample_noise_field,
    sample_noise_field_region,
    sample_noise_field_3d,
    smooth_field,
    blend_fields,
    clamp_field,
    normalize_field,
)
from .mountains import (
    MountainConfig,
    generate_mountains,
    MOUNTAIN_RANGE,
    ALPINE_PEAKS,
    ROLLING_HILLS,
    MESA_PLATEAU,
)
from .regions import (
    RegionMap,
    RegionValidation,
    partition_angular,
    partition_flood_fill,
    partition_voronoi,
    partition_noise,
    assign_field,
    assign_biome,
    regions_to_overlay,
    validate_region_map,
)

__all__ = [
    "fbm", "ridged_noise", "domain_warp", "gradient_mask", "terrace",
    "noise_normalize", "remap", "fbm_3d", "ridged_noise_3d",
    "sample_noise_field", "sample_noise_field_region", "sample_noise_field_3d",
    "smooth_field", "blend_fields", "clamp_field", "normalize_field",
    "MountainConfig", "generate_mountains",
    "MOUNTAIN_RANGE", "ALPINE_PEAKS", "ROLLING_HILLS", "MESA_PLATEAU",
    "RegionMap", "RegionValidation",
    "partition_angular", "partition_flood_fill", "partition_voronoi", "partition_noise",
    "assign_field", "assign_biome", "regions_to_overlay", "validate_region_map",
]
