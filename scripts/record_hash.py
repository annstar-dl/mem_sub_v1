import datetime
import argparse
import yaml
import os
import subprocess
import sys
import inspect
from mem_sub import __version__
from pathlib import Path
from setuptools_scm import get_version

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def get_repository_path() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode('ascii').strip()

def save_yaml_file(save_path: str, data: dict) -> None:
    with open(save_path, 'w') as f:
        yaml.dump(data, f)


def check_version_sync_boolean():
    """
    Returns True if the version is verified or if a check is not possible (Standard install).
    Returns False ONLY if a Git mismatch is found (Editable install out of sync).
    """
    # 1. Locate the .git folder relative to this file
    # Path: src/mem_sub/__init__.py -> src/mem_sub -> src -> Project Root
    package_root = Path(__file__).parent.parent.parent
    git_dir = package_root / ".git"

    # 2. If no .git folder, we assume a stable, standard installation.
    if not git_dir.exists():
        return True

    # 3. If .git exists, try to compare live vs. installed
    try:

        live_version = get_version(root=package_root)

        # If they match, we are synced.
        return live_version == __version__
    except Exception:
        # If setuptools_scm isn't found, we can't verify,
        # so we return True to avoid false alarms.
        return True


def get_conda_env_info() -> dict:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "unknown")
    conda_prefix = os.environ.get("CONDA_PREFIX", "unknown")
    is_synced = check_version_sync_boolean()
    return {
        "conda_env_name": conda_env,
        "conda_prefix": conda_prefix,
        "mem_sub": {"version" : __version__,
                    "status": "Verified" if is_synced else "WARNING: Stale Installation"},
    }

def create_metadata(caller_frame=None, script_args=None) -> None:
    d = {}
    frame = caller_frame if caller_frame else inspect.stack()[1]
    d['initiating module'] = frame.filename
    d['git_revision_hash'] = get_git_revision_hash()
    d['timestamp'] = datetime.datetime.now().isoformat()
    d['repo_path'] = get_repository_path()
    d['conda'] = get_conda_env_info()
    if script_args is not None:
        d['script_args'] = script_args
    return d

def check_git_status():
    # 'git status --porcelain' returns an empty string if nothing is changed
    #status = subprocess.check_output(['git', 'status', '--porcelain', '--ignore-submodules']).decode('utf-8').strip()
    status = subprocess.check_output(['git', 'status', '--porcelain']).decode('utf-8').strip()
    print(f"DEBUG: Status output is: '{status}'")  # The quotes will show if there's a hidden newline

    if status:
        raise Exception("Error: Uncommitted changes detected! Commit your changes first or delete them!")

def save_metadata(save_path, script_args=None) -> None:
    """
    Save metadata to yaml file of the save_folder
    Args:
    :param save_path: path to the dir with the data
    :param save_seg_dir: binary. True if save the segmentation folder and False otherwise
    returns: None
        """
    check_git_status()
    caller_frame = inspect.stack()[1]
    d = create_metadata(caller_frame=caller_frame, script_args=script_args)
    if os.path.exists(save_path):
        metadata_different = compare_metadata(save_path, script_args=script_args)
        if metadata_different:
            raise Exception(
                f"Metadata file {save_path} and current project are not the same! Restore the project state or use new folder.")
        else:
            print(f"Metadata file already exists, but metadata files are the same (except for timestamp).")
    save_yaml_file(save_path, d)

def load_yaml_file(fpath) -> dict:
    with open(fpath, 'r') as f:
        return yaml.safe_load(f)

def compare_metadata(old_yml_path, script_args=None) -> None:
    caller_frame = inspect.stack()[1]
    d_new = create_metadata(caller_frame=caller_frame, script_args=script_args)
    d_old = load_yaml_file(old_yml_path)
    files_different = False
    for k, v in d_old.items():
        if k == "timestamp":
            continue
        if d_new[k] != v:
            print(f"Metadata field '{k}' is different! Old value: {v}, new value: {d_new[k]}")
            files_different = True
    return files_different




if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-sp",'--save_path', help="Path to saved results", required=True)
    args.add_argument("-fname", '--fname', help="Yml file name", default="exp_config")
    args = args.parse_args()
    save_path = os.path.join(args.save_path,args.fname+".yml")
    save_metadata(save_path)


