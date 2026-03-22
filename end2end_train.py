import re
from pathlib import Path

from ruamel.yaml import YAML

from rocnovo.common.io import normalize_path
from rocnovo.module.clip import train as train_clip
from rocnovo.module.denovo import train as train_denovo

def search_best_clip_model(clip_model_save_folder: Path):
    def extract_metrics(file_path):
        match = re.search(r"step=(\d+).*accuracy=(\d+(?:\.\d+)?)", file_path.name)
        if match:
            step = int(match.group(1))
            accuracy = float(match.group(2))
            return (accuracy, step)
        
        return (-1.0, -1)

    ckpt_files = list(clip_model_save_folder.glob("*.ckpt"))
    if not ckpt_files:
        return None
    
    sorted_files = sorted(ckpt_files, key=extract_metrics)
    return normalize_path(sorted_files[-1])

yaml = YAML()
yaml.preserve_quotes = True
yaml.indent(mapping=2, sequence=4, offset=2)

def read_yaml(path: str | Path):
    with open(path, "r") as f:
        denovo_base_config = yaml.load(f)
    
    return denovo_base_config

def generate_denovo_config(clip_config_path: str | Path, denovo_config_path: str | Path):
    save_dir = normalize_path("./yamls/massiveKB_dummy")
    save_dir.mkdir(parents=True, exist_ok=True)
    clip_base_config = read_yaml(clip_config_path)
    denovo_base_config = read_yaml(denovo_config_path)

    denovo_base_config["clip_checkpoint_path"] = str(search_best_clip_model(
        normalize_path(clip_base_config["trainer"]["model_save_folder"])
    ))

    yaml.dump(denovo_base_config, denovo_config_path)

train_clip(normalize_path("./yamls/massiveKB_dummy/clip.yaml"))

clip_path = normalize_path("./yamls/massiveKB_dummy/clip.yaml")
denovo_path = normalize_path("./yamls/massiveKB_dummy/denovo.yaml")

generate_denovo_config(clip_path, denovo_path)
train_denovo(denovo_path)