from pathlib import Path
from ruamel.yaml import YAML
import pdb
import json

def prune_to_matching_key(d: dict, target_key: str) -> dict:
    """
    Recursively searches `d` for any subdictionary containing `target_key`.
    Returns a dictionary preserving ONLY the nested parent keys leading to 
    the match, excluding all unmatching sibling keys.
    """
    if not isinstance(d, dict):
        return {}

    result = {}

    # Case 1: Target key exists at this level
    if target_key in d:
        return {target_key: d[target_key]}

    # Case 2: Recurse into child dictionaries
    for key, value in d.items():
        if isinstance(value, dict):
            child_result = prune_to_matching_key(value, target_key)
            if child_result:
                # Keep ONLY this parent key and its matching branch
                result[key] = child_result

    return result

def load_folder_yamls(search_dict = dict) -> dict:
    # Get the directory of the current script file
    if not search_dict:
        return None
    current_dir = Path(__file__).resolve().parent / search_dict['path']
    
    # Initialize ruamel.yaml instance
    yaml = YAML(typ='safe')  # 'safe' loads standard dictionaries/lists
    
    folder_yamls = {}
    result = {}
    
    # Find all .yaml and .yml files directly inside the subdirectory
    for yaml_file in current_dir.iterdir():
        if yaml_file.is_file() and yaml_file.suffix.lower() in ('.yaml', '.yml'):
            with open(yaml_file, 'r', encoding='utf-8') as f:
                # Key is file name without extension, value is parsed YAML content
                folder_yamls[yaml_file.stem] = yaml.load(f)

    # Only include the folder if it contained at least one YAML file
    if folder_yamls:
        if not search_dict['path']:
            result[search_dict['key']] = folder_yamls
        else:
            result[search_dict['path'].split('/')[-1]] = folder_yamls

    return prune_to_matching_key(result, search_dict['key'])