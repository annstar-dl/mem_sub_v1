try:
    from ._version import version as __version__
except ImportError:
    # Fallback for when the file hasn't been generated yet
    __version__ = "0+unknown"
