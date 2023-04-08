import os
import json
import multiprocessing as mp
from itertools import product
from subprocess import run
from config import GRID_SEARCH_CONFIG, BASE_CONFIG, MAX_PARALLEL, prepareConfig
from copy import deepcopy


def serializeForCLI(run_config):
    json_serl_config = json.dumps(run_config)
    assert "<space>" not in json_serl_config
    assert "<dblQuotes>" not in json_serl_config
    json_serl_config = json_serl_config.replace(" ", '<space>').replace('"', "<dblQuotes>")
    return json_serl_config

def makeSubprocessCommands(base_config:dict, grid_search_config:dict, relaive_pth_to_trainpy:str):

    def editConfig(config, path:str, val):
        splitedPath = path.split(".")
        root = config
        for k in splitedPath[:-1]: 
            root = root[k]
        assert splitedPath[-1] in root, f"{path} doesnot exist!"
        root[splitedPath[-1]] = val
        config["WANDB"]["notes"] = \
            config["WANDB"]["notes"] + f"\n{splitedPath}={val}"
        config["run_name_prefix"] = config["run_name_prefix"] + f" {val}" 

    commandTemplate = f"python {relaive_pth_to_trainpy}/train.py --run-config=\"{{}}\""
    params = [*zip(*grid_search_config.items())]
    for args in product(*params[1]):
        config_copy = deepcopy(base_config)
        [editConfig(config_copy, *pathAndVal) for pathAndVal in zip(params[0], args)]
        yield commandTemplate.format(serializeForCLI(prepareConfig(config_copy)))

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
    commands = [*makeSubprocessCommands(BASE_CONFIG, GRID_SEARCH_CONFIG, relaive_pth)]
    
    if len(commands) > 0:
        with mp.Pool(min(MAX_PARALLEL, len(commands))) as pool:
            pool.map(worker, commands)
        print("grid-search finished!")
    else:
        print("nothing to grid-search")