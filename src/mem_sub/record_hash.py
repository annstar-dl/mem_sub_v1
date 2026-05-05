import datetime
import argparse
import yaml
import os
import subprocess
from pathlib import Path
from importlib.metadata import distribution, PackageNotFoundError
import json
import mem_sub
import inspect
from mem_sub import __version__

def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def get_repository_path() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode('ascii').strip()

def save_yaml_file(save_path: str, data: dict) -> None:
    with open(save_path, 'w') as f:
        yaml.dump(data, f)


def get_package_integrity():
    try:
        dist = distribution("mem_sub")

        # 1. Detect if it's an editable installation
        is_editable = False
        direct_url_content = dist.read_text("direct_url.json")
        if direct_url_content:
            url_data = json.loads(direct_url_content)
            is_editable = url_data.get("dir_info", {}).get("editable", False)

        # Get physical location of the running code
        active_path = Path(mem_sub.__file__).resolve().parent

        # 2. Implementation of your logic
        if is_editable:
            # DEVELOPMENT MODE: Report the link (path) to the package
            return {
                "install_type": "editable",
                "source_link": str(active_path),
                "version": mem_sub.__version__,
                "status": "Path-Linked (Development)"
            }
        else:
            # PRODUCTION MODE: Check that environment metadata matches the code
            pip_version = dist.version
            internal_version = mem_sub.__version__

            if pip_version != internal_version:
                return {
                    "install_type": "standard",
                    "status": "MISMATCH",
                    "error": f"Pip reports {pip_version} but code is {internal_version}"
                }

            return {
                "install_type": "standard",
                "version": pip_version,
                "status": "Verified Standard Install"
            }

    except PackageNotFoundError:
        return {"status": "NOT_INSTALLED", "error": "Package 'mem_sub' not found in active environment"}

def get_conda_env_info() -> dict:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "unknown")
    conda_prefix = os.environ.get("CONDA_PREFIX", "unknown")
    return {
        "conda_env_name": conda_env,
        "conda_prefix": conda_prefix,
        "mem_sub": get_package_integrity(),
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
        metadata_different = compare_metadata(save_path, new_metadata=d)
        if metadata_different:
            raise Exception(
                f"Metadata file {save_path} and current project are not the same! Restore the project state or use new folder.")
        else:
            print(f"Metadata file already exists, but metadata files are the same (except for timestamp and initiating module).")
    save_yaml_file(save_path, d)

def load_yaml_file(fpath) -> dict:
    with open(fpath, 'r') as f:
        return yaml.safe_load(f)

def compare_metadata(old_yml_path, new_metadata) -> bool:
    d_old = load_yaml_file(old_yml_path)
    files_different = False
    for k, v in d_old.items():
        if k == "timestamp" or k == "initiating module":
            continue
        if new_metadata.get(k) != v:
            print(f"Metadata field '{k}' is different! Old value: {v}, new value: {new_metadata.get(k)}")
            files_different = True
    return files_different




if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-sp",'--save_path', help="Path to saved results", required=True)
    args.add_argument("-fname", '--fname', help="Yml file name", default="exp_config")
    args = args.parse_args()
    save_path = os.path.join(args.save_path,args.fname+".yml")
    save_metadata(save_path)


