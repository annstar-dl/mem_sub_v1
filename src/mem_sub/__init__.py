try:
    # 1. Try to get the LIVE version from Git (for your development)
    from setuptools_scm import get_version
    # This looks for the .git folder relative to this file
    __version__ = get_version(root='../..', relative_to=__file__)
except (ImportError, LookupError):
    # 2. Fallback to the INSTALLED version (for your users/cluster)
    try:
        from importlib.metadata import version, PackageNotFoundError
        __version__ = version("mem_sub")
    except PackageNotFoundError:
        __version__ = "0+unknown"