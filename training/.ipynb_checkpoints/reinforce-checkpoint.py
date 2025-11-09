import numpy as np
import torch
from collections import deque, defaultdict
from utils.convergence import check_convergence
from utils.greedy_solution import solution
from policies.linear_policy import LinearPolicy
from policies.policy_net import PolicyNet

def reinforce_multi_rwd2go_alt(env, policies, optimizers, n_episodes=30000, max_t=20, gamma=0.9, batch_size=128, print_every=25):
    
    # Add convergence parameters
    window = 100
    min_episodes = 3*window + 1        # minimum episodes before checking convergence
    max_rounds = 200
    
    scores_deque = deque(maxlen=window)
    # Track average return (already in scores) and rewards-to-go
    scores = []   # average episodic returns per agent
    values_log = [[] for _ in range(env.n_agents)]  # track rewards-to-go estimates

    # For WITHIN-ROUND convergence tracking
    round_values = [[] for _ in range(env.n_agents)]
    round_convergence_threshold = 7.5e-3  # can tune this
    
    round_count = 0
    all_agents_stable = False

    # Enhanced parameters for both methods
    entropy_coef = [0.7, 0.7]
    entropy_decay = [0.995, 0.995]
    min_entropy_coef = 0.0

    
    while round_count < max_rounds and not all_agents_stable:  
        
        round_count += 1
        entropy_coef[0] = 0.7
        print(f"\n{'='*50}")
        print(f"=== Round {round_count} ===")
        print(f"{'='*50}")
        
        # Alternate through each agent
        for phase in range(env.n_agents):
            print(f"\n--- Training Agent {phase} ---")
            
            episode_count = 0
            agent_converged = False  # WITHIN-ROUND convergence flag
            
            while episode_count < n_episodes and not agent_converged:
                episode_count += 1

                avg_rtg = 0
                # Accumulate batch loss for current agent
                batch_loss = 0.0
                batch_count = 0
                batch_value_logs = defaultdict(list)
                batch_rewards = []
                batch_entropies = []
                
                # ---- collect batch of episodes ----
                for _ in range(batch_size):
                    states = env.reset()
            
                    saved_log_probs = [[] for _ in range(env.n_agents)]
                    rewards = [[] for _ in range(env.n_agents)]
                    saved_entropies = [[] for _ in range(env.n_agents)]
                    epi_states = [[] for _ in range(env.n_agents)]
                    dones = [False] * env.n_agents
            
                    # episode
                    for t in range(max_t):
                        actions, log_probs, entropies = [], [], []
                        for i in range(env.n_agents):
                            a, lp, entropy = policies[i].act(states[i])
                            actions.append(a)
                            log_probs.append(lp)
                            epi_states[i].append(states[i])
                            
                            entropies.append(entropy)
            
                        next_states, step_rewards, dones = env.step(actions)
                        for i in range(env.n_agents):
                            saved_log_probs[i].append(log_probs[i])
                            rewards[i].append(step_rewards[i])
                            saved_entropies[i].append(entropies[i])  # Save per step
            
                        states = next_states
                        if any(dones):
                            for i in range(env.n_agents):
                                epi_states[i].append(states[i])
                            break
        
                    # --- Process current agent's trajectory ---
                    if len(rewards[phase]) > 0:
                        # rewards-to-go (Monte Carlo return)
                        discounts = [gamma**k for k in range(len(rewards[phase]) + 1)]
                        rewards_to_go = [
                            sum(discounts[j] * rewards[phase][j+t] for j in range(len(rewards[phase]) - t))
                            for t in range(len(rewards[phase]))
                        ]
                        rewards_to_go = torch.tensor(rewards_to_go, dtype=torch.float32)

                        # Log average rewards-to-go as proxy for V(s)
                        avg_rtg += rewards_to_go.mean().item()
                        
                        # --- policy loss for current agent ---
                        pol_terms = []
                        for lp, G, entropy in zip(saved_log_probs[phase], rewards_to_go, saved_entropies[phase]):
                            if isinstance(lp, torch.Tensor):
                                pol_terms.append(-lp * G)
                                batch_entropies.append(entropy)
        
                        if pol_terms:
                            ep_loss = torch.stack(pol_terms).sum()
                            batch_loss += ep_loss
                            batch_count += 1

        
                    # logging rewards per episode
                    episode_rewards = [sum(r) for r in rewards]
                    batch_rewards.append(episode_rewards)
                    
                values_log[phase].append(avg_rtg/batch_count)
                # Calculate mean entropy across the entire batch
                if batch_entropies:
                    mean_entropy = torch.stack(batch_entropies).mean()
                else:
                    mean_entropy = torch.tensor(0.0)
                
                # Update current agent's policy
                if batch_count > 0:
                    loss = (batch_loss / batch_count) - entropy_coef[phase] * mean_entropy
                    optimizers[phase].zero_grad()
                    loss.backward()
                    #torch.nn.utils.clip_grad_norm_(policies[phase].parameters(), max_norm=1.0)
                    optimizers[phase].step()

                    if phase == 1 and episode_count > 24:
                        temp = check_convergence(values_log, phase, window, episode_count, min_episodes)
                        break
                
                # Decay entropy coefficient periodically
                if episode_count % 5 == 0:
                    entropy_coef[phase] = max(min_entropy_coef, entropy_coef[phase] * entropy_decay[phase])
                
                # Update scores
                avg_batch_rewards = np.mean(batch_rewards, axis=0)
                scores_deque.append(avg_batch_rewards)
                scores.append(avg_batch_rewards)

                agent_converged = check_convergence(values_log, phase, window, episode_count, min_episodes)
                                
                if agent_converged:
                    print(f"Agent {phase} converged after {episode_count} episodes")
                    break

                # ---- print progress ----
                if episode_count % print_every == 0:
                    avg_rewards = np.mean(scores_deque, axis=0) if len(scores_deque) > 0 else [0]*env.n_agents
                    msg = f" Agent{phase} Episode {episode_count}"
                    msg += f" avgR={avg_rewards[phase]:.4f}"
                    print(msg)
                    print("COEF: ", entropy_coef[phase])
                    print(" ")


        # BETWEEN-ROUND convergence check (after all agents complete this round)
        solution(env, policies) 
        print(f"\n--- End of Round {round_count}: Computing Round Averages ---")


        # === Reset the first agent's LinearPolicy after each round ===
        if isinstance(policies[0], LinearPolicy):
        #if False:
            print("\nResetting Agent 0 (LinearPolicy) parameters for the next round...")
            state_size = policies[0].W.shape[0] - 1
            action_size = policies[0].W.shape[1]
            maze_size = policies[0].maze_size
            policies[0] = LinearPolicy(state_size=state_size, action_size=action_size, maze_size=maze_size, epsilon=0.0, min_epsilon=0.0)
            optimizers[0] = torch.optim.Adam(policies[0].parameters(), lr=0.01)

        elif isinstance(policies[0], PolicyNet) and False:
            print("\nResetting Agent 0 (PolicyNet) parameters for the next round...")
            state_size = policies[0].fc1.in_features
            action_size = policies[0].fc2.out_features
            hidden_size = policies[0].fc1.out_features

            # Reinitialize a new PolicyNet with same structure
            policies[0] = PolicyNet(
                state_size=state_size,
                action_size=action_size,
                hidden_size=hidden_size,
            )
            optimizers[0] = torch.optim.Adam(policies[0].parameters(), lr=0.01)
        
        for phase in range(env.n_agents):
            if len(values_log[phase]) >= window:
                mean_V_round = np.mean(values_log[phase][-window:])  # last window as representative of this round
                round_values[phase].append(mean_V_round)   
        
        if round_count > 4:
            all_stable = True
            for phase in range(env.n_agents):
                prev_mean = round_values[phase][-2]
                curr_mean = round_values[phase][-1]
                delta_round = abs(curr_mean - prev_mean)
                print(f"   Agent {phase}: ΔV_round = {delta_round:.6e}")
                
                if delta_round < round_convergence_threshold:
                    print(f" Agent {phase} stabilized between Rounds {round_count-1} and {round_count}")
                else:
                    all_stable = False
                    print(f" Agent {phase} still changing between rounds (Δ={delta_round:.6e})")
        
            if all_stable:
                print(f"\n All agents' VALUE functions stabilized between rounds — training converged.")
                all_agents_stable = True
                break

        
        print(f"{'='*50}")

    
    return scores, values_log
