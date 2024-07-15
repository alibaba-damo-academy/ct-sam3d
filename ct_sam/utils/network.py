
def partial_weight_update(model, pretrained_state):
    model_state = model.state_dict()
    state_dict = {k: v for k, v in pretrained_state.items() if k in model_state.keys()}
    model_state.update(state_dict)
    model.load_state_dict(model_state)
    return model
