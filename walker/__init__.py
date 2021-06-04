"""isort:skip_file
"""
from ._install import install

CACHE_DIR = install()

from .sampler import Sampler
