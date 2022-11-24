class QvalueStats:
    def __init__(self) -> None:
        pass

    def __call__(self, info_dict):
        max_qvalues = info_dict["qvalue_stats"]["max"]
        min_qvalues = info_dict["qvalue_stats"]["min"]
        info_dict["qvalue_stats"]["max"]={
            f"action_{action}": qval for action, qval in enumerate(max_qvalues)
        }
        info_dict["qvalue_stats"]["min"]={
            f"action_{action}": qval for action, qval in enumerate(min_qvalues)
        }
        return info_dict