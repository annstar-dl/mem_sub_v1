import datetime
import argparse
import yaml
import os
import subprocess
import sys
import inspect
from verify_subpackege_version import verify_package_version


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def get_repository_path() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode('ascii').strip()

def save_yaml_file(save_path: str, data: dict) -> None:

    if not os.path.isfile(save_path):
        with open(save_path, 'w') as f:
            yaml.dump(data, f)
    else:
        raise Exception(f"File '{os.path.basename(save_path)}' already exists!. Delete the experiment folder or change the save path to avoid overwriting.")



def get_conda_env_info() -> dict:
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "unknown")
    conda_prefix = os.environ.get("CONDA_PREFIX", "unknown")

    return {
        "conda_env_name": conda_env,
        "conda_prefix": conda_prefix,
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
        print("Error: Uncommitted changes detected! Commit your changes first or delete them!")
        sys.exit(1)  # Exit with error code

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
    save_yaml_file(save_path, d)

def load_yaml_file(fpath) -> dict:
    with open(fpath, 'r') as f:
        return yaml.load(f)

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
    if not files_different:
        print("Metadata files are the same (except for timestamp).")
    return files_different




if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-sp",'--save_path', help="Path to saved results", required=True)
    args.add_argument("-fname", '--fname', help="Yml file name", default="exp_config")
    args.add_argument("-c","--compare_metadata", help="compare_metadata", action="store_true")
    args = args.parse_args()
    d = create_metadata()
    save_path = os.path.join(args.save_path,args.fname+".yml")
    if args.compare_metadata:
       files_differenet = compare_metadata(save_path, d)
       if files_differenet:
           raise Exception(f"Metadata file {save_path} and current project are not the same! Restore the project state or use new folder.")
    else:
        save_yaml_file(args.save_path, d)


