"""Noise generation, heightmaps, mountains, region partitioning, and climate."""

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
from .temperature import (
    compute_temperature,
    generate_temperature_field,
    LATITUDE_WEIGHT,
    LAPSE_RATE,
)
from .moisture import (
    compute_ocean_distance,
    generate_moisture_field,
    OCEAN_PROXIMITY_WEIGHT,
    ELEVATION_PENALTY,
    MAX_OCEAN_DISTANCE,
)
from .classification import (
    classify_tile,
    generate_terrain_field,
    TERRAIN_TYPES,
    OCEAN,
    SNOW,
    TUNDRA,
    MOUNTAINS,
    DESERT,
    WETLAND,
    HILLS,
    PLAINS,
)
from .features import (
    detect_coast,
    detect_lakes,
    place_forests,
    generate_features,
    get_features,
    add_feature,
    COAST,
    LAKE,
    FOREST,
    FEATURE_TYPES,
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
    "compute_temperature", "generate_temperature_field",
    "LATITUDE_WEIGHT", "LAPSE_RATE",
    "compute_ocean_distance", "generate_moisture_field",
    "OCEAN_PROXIMITY_WEIGHT", "ELEVATION_PENALTY", "MAX_OCEAN_DISTANCE",
    "classify_tile", "generate_terrain_field", "TERRAIN_TYPES",
    "OCEAN", "SNOW", "TUNDRA", "MOUNTAINS", "DESERT", "WETLAND", "HILLS", "PLAINS",
    "detect_coast", "detect_lakes", "place_forests", "generate_features",
    "get_features", "add_feature",
    "COAST", "LAKE", "FOREST", "FEATURE_TYPES",
    "RegionMap", "RegionValidation",
    "partition_angular", "partition_flood_fill", "partition_voronoi", "partition_noise",
    "assign_field", "assign_biome", "regions_to_overlay", "validate_region_map",
]
