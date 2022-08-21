from berry_field.envs import BerryFieldEnv
from torch import nn, device, load
from Agent import Agent
from DRLagents import greedyAction, DDQN

PATH_TEMPLATE = "..\\trainLogs\\onlinemodel_weights_episode_{}.pth"
EPISODES = [226, 242, 257, 263]

class Ensemble(nn.Module):
    def __init__(self, agent:Agent, episodes, TORCH_DEVICE=device('cpu')) -> None:
        super(Ensemble, self).__init__()
        self.models= nn.ModuleList()
        for episode in episodes:
            model = agent.makeNet(TORCH_DEVICE)
            model.load_state_dict(load(PATH_TEMPLATE.format(episode)))
            self.models.append(model)
    
    def __call__(self, state):
        q_vals = self.models[0](state)
        for model in self.models[1:]: 
            q_vals += model(state)
        return q_vals/len(self.models)

if __name__ == "__main__":
    berry_env = BerryFieldEnv()
    agent = Agent(berry_env, mode='eval', debug=True, noise=0.025, persistence=0.7)

    buffer = None; optim = None; tstrat = None
    ensemble = Ensemble(agent, EPISODES).eval()
    estrat = greedyAction(ensemble)
    ddqn_trainer = DDQN(berry_env, ensemble, tstrat, optim, buffer, batchSize=256, skipSteps=10,
                        make_state=agent.makeState, make_transitions=agent.makeStateTransitions,
                        gamma=0.9, MaxTrainEpisodes=50, user_printFn=None, printFreq=1, log_dir=None)
    ddqn_trainer.evaluate(estrat, render=True)
    agent.showDebug(ensemble)