class AgentAnalyticsProcessor:
    def __init__(self) -> None:
        pass

    def __call__(self, info:dict):
        agent_analytics = info["raw_stats"]["agent"]
        info["agent"] = {
            "action_stats": agent_analytics["env_adapter"]["action_stats"],
            "memory_stats": agent_analytics["memory_manager"]
        }
        return info