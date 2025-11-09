import torch

def solution(env, policies):
    """
    Combined solution that works with both LinearPolicy and neural network policies
    """
    
    # --- Run one greedy rollout ---
    dones = [False] * env.n_agents
    states = env.reset()
    env.render()   
    max_steps = 20 
    for step in range(max_steps):
        actions = []
        for i in range(env.n_agents):
            # Detect policy type and get greedy action accordingly
            if hasattr(policies[i], '_phi'):
                # LinearPolicy case - use _phi method
                phi = policies[i]._phi(states[i])
                logits = phi @ policies[i].W
                probs = torch.softmax(logits, dim=0)
                action = torch.argmax(probs, dim=0).item()
            else:
                # Neural network policy case - call directly
                state_tensor = torch.tensor(states[i], dtype=torch.float32).unsqueeze(0)
                probs = policies[i](state_tensor)
                action = torch.argmax(probs, dim=1).item()
                print("Agent:", i)
                print("State: ", state_tensor)
                print("Prob: ", probs)
            
            actions.append(action)

        next_states, step_rewards, dones = env.step(actions)
        states = next_states
        env.render()
        if any(dones):
            break