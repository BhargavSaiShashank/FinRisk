def compute_reward(action, label, rho):
    """
    Independent reward function implementation for full visibility.
    rho = minority_class_size / majority_class_size
    """
    if action == 1 and label == 1: return 1.0     # TP (High risk correctly ID'd)
    if action == 1 and label == 0: return -1.0    # FP (False alarm)
    if action == 0 and label == 0: return rho     # TN (Low risk correctly ID'd)
    if action == 0 and label == 1: return -rho    # FN (High risk missed)
    return 0.0

def calculate_rho(labels):
    """
    Labels are 0 and 1.
    rho = count_1 / count_0
    """
    count_1 = (labels == 1).sum()
    count_0 = (labels == 0).sum()
    if count_0 == 0: return 1.0
    return count_1 / count_0
