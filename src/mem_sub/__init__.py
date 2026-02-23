from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("mem_sub") # Match the 'name' in pyproject.toml
except PackageNotFoundError:
    # Package is not installed, perhaps running from source
    __version__ = "0.0.0-dev"