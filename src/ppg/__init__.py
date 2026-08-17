"""Profile Photo Generator - local, photorealistic synthetic avatars."""

import os as _os

__version__ = "0.1.0"

# Set before anything can create a CUDA context. Expandable segments stop the
# allocator from fragmenting, which is what produces "tried to allocate 26 MiB,
# 174 MiB free" errors on a card that is also driving a desktop session.
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Bumped whenever a change would make previously cached images stale
# (different sampler, different realism cue block, different defaults).
# It is part of every cache key, so bumping it invalidates the whole store.
PIPELINE_VERSION = 2
