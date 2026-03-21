# Backward-compatibility shim — canonical location: polygrid.terrain.noise
from .terrain.noise import *  # noqa: F401,F403
from .terrain.noise import _init_noise, _init_noise3  # noqa: F401 — private, used by tests
