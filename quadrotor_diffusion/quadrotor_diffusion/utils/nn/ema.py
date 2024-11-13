def update_average(old, new, beta):
    """
    Comptes weighted average
    """
    if old is None:
        return new
    return old * beta + (1 - beta) * new


def update_model_average(ma_model, current_model, beta):
    """
    Updates moving average model in place using current model and beta
    """
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = update_average(old_weight, up_weight, beta)
