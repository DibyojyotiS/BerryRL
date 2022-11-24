class AgentAnalyticsProcessor:
    def __init__(self) -> None:
        pass

    def __call__(self, info:dict):
        agent_analytics = info["raw_stats"]["agent"]
        info["agent"] = {
            "action_stats": agent_analytics["action_stats"]
        }
        return info