def max_qvalues_stats_callback(info_dict):
    max_qvalues = info_dict["train"]["qvalue_stats"]
    info_dict["train"]["qvalue_stats"]={
        f"action_{action}": qval for action, qval in enumerate(max_qvalues)
    }
    print("Max q-values:- ", "".join([f"{i}: {x:.3f}" for i,x in enumerate(max_qvalues)]))
    return info_dict