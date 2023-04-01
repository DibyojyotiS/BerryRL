import os
import json
import multiprocessing as mp
from itertools import product
from subprocess import run
from config import GRID_SEARCH_CONFIG, BASE_CONFIG, MAX_PARALLEL
from copy import deepcopy

def make_grid_search_configs(base_config, grid_search_config:dict):

    def editConfig(config, path:str, val):
        path = path.split(".")
        root = config
        for k in path[:-1]: 
            root = root[k]
        root[path[-1]] = val
        config["WANDB"]["notes"] = \
            config["WANDB"]["notes"] + f"\n{path}={val}"
        config["run_name_prefix"] = config["run_name_prefix"] + f" {val}" 

    params = [*zip(*grid_search_config.items())]
    for args in product(*params[1]):
        config_copy = deepcopy(base_config)
        [editConfig(config_copy, *pathAndVal) for pathAndVal in zip(params[0], args)]
        yield config_copy

def jsonSerializeForCLI(run_config):
    json_serl_config = json.dumps(run_config)
    assert "<space>" not in json_serl_config
    assert "<dblQuotes>" not in json_serl_config
    json_serl_config = json_serl_config.replace(" ", '<space>').replace('"', "<dblQuotes>")
    return json_serl_config

def worker(cmd):
    try:
        from subprocess import CREATE_NEW_CONSOLE
        run(cmd, creationflags=CREATE_NEW_CONSOLE)
    except Exception as e:
        print("Can't create a new console, retrying without CREATE_NEW_CONSOLE")
        try:
            run(cmd)
        except Exception as e2:
            print("trying with shell=True")
            run(cmd, shell=True)

if __name__ == "__main__":
    relaive_pth = os.path.split(__file__)[0]
    json_serl_configs = [jsonSerializeForCLI(c) 
        for c in make_grid_search_configs(BASE_CONFIG, GRID_SEARCH_CONFIG)]
    commands = [
        f"python {relaive_pth}/train.py --run-config=\"{json_serl_config}\""
        for json_serl_config in json_serl_configs
    ]
    
    if len(commands) > 0:
        with mp.Pool(min(MAX_PARALLEL, len(commands))) as pool:
            pool.map(worker, commands)
        print("grid-search finished!")
    else:
        print("nothing to grid-search")