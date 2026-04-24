import datetime
import subprocess
import argparse
import yaml
import os


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()

def get_repository_path() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).decode('ascii').strip()

def save_yaml_file(path: str, data: dict) -> None:
    path = os.path.join(path, f'seg_config.yml')
    if not os.path.isfile(path):
        with open(path, 'w') as f:
            yaml.dump(data, f)

def load_yaml_file() -> dict:
    with open("seg_parameters.yml", 'r') as f:
        return yaml.load(f)

def create_metadata(args: argparse.Namespace) -> None:
    d = {}
    d['git_revision_hash'] = get_git_revision_hash()
    d['timestamp'] = datetime.datetime.now().isoformat()
    if args.save_seg_dir:
        d['seg_model'] = load_yaml_file()['seg_model']
    d['repo_path'] = get_repository_path()
    return d

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("-sp",'--save_path', help="Path to saved results", required=True)
    args.add_argument("--save_seg_dir", action="store_true", help="Path to repo directory")
    args = args.parse_args()
    d = create_metadata(args)
    save_yaml_file(args.save_path, d)


