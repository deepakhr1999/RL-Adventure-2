import torch


def rollout(env, model, device, sample=True, vis=False):
    state = env.reset()
    if vis:
        env.render()
    done = False
    total_reward = 0
    all_states = []
    all_actions = []
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        all_states.append(state)
        with torch.no_grad():
            dist, _ = model(state)
        if sample:
            action = dist.sample().cpu().numpy()[0]
        else:
            action = dist.probs.cpu().numpy().argmax(axis=1)[0]
        all_actions.append(action)

        next_state, reward, done, _ = env.step(action)
        state = next_state
        if vis:
            env.render()
        total_reward += reward
    return total_reward, all_states, all_actions
