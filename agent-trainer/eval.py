from DRLagents import *
from DRLagents.agents.DDQN import greedyAction
from config import CONFIG
from berry_field import BerryFieldEnv
from berry_field.envs.analysis.visualization import picture_episode
from berry_field.envs.analysis.visualization import juice_plot
from agent import Agent

# set all seeds
set_seed(CONFIG["seed"])

LOG_DIR = None
TORCH_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    evalEnv = BerryFieldEnv()
    agent = Agent(
        berry_env_FIELD_SIZE=evalEnv.FIELD_SIZE,
        berry_env_HALFDIAGOBS=evalEnv.HALFDIAGOBS,
        berry_env_REWARD_RATE=evalEnv.REWARD_RATE,
        berry_env_DRAIN_RATE=evalEnv.DRAIN_RATE,
        torch_device=TORCH_DEVICE,
        **CONFIG["AGENT"]
    )
    evalEnv = agent.getPerceivedEnvironment(evalEnv)

    nnet = agent.nnet()
    nnet.load_state_dict(torch.load(
        ".temp\\nearby-berry-memory\\0.0.0\\2022-11-21 19-42-48\\trainLogs\models\episode-174\onlinemodel_statedict.pt"
    ))
    nnet = nnet.eval()

    buffer = None; optim = None; tstrat = None
    estrat = greedyAction()

    ddqn_trainer = DDQN(evalEnv, nnet, tstrat, optim, buffer,
                        log_dir=LOG_DIR, device=TORCH_DEVICE)

    try:ddqn_trainer.evaluate(estrat, render=False)
    except KeyboardInterrupt as ex: pass

    agent_stats = agent.get_stats()
    env_stats = evalEnv.get_analysis()
    print(
        f"actions: {agent_stats['action_stats']}\n",
        f"nberries: {env_stats['berries_picked']}\n",
        f"env_steps: {env_stats['env_steps']}"
    )

    juice_plot(
        sampled_juice=env_stats["sampled_juice"],
        total_steps=env_stats["env_steps"],
        max_steps=evalEnv.MAX_STEPS,
        title="juice",
        nberries_picked=env_stats["berries_picked"],
    )

    picture_episode(
        berry_boxes=env_stats["berry_boxes"],
        patch_boxes=env_stats["patch_boxes"],
        sampled_path=env_stats["sampled_path"],
        nberries_picked=env_stats['berries_picked'],
        total_steps=env_stats['env_steps'],
        field_size=evalEnv.FIELD_SIZE
    )



