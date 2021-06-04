"""isort:skip_file
"""
from ._install import install, print, rule

CACHE_DIR = install()
rule("Finished set-up.")

from .sampler import Sampler, print
