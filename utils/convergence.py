import numpy as np
import matplotlib.pyplot as plt


def check_convergence(values_log, phase, window, episode_count, min_episodes):
    #After policy update, check convergence
    if episode_count >= min_episodes:

        prev_window = values_log[phase][-window-1 : -1]
        curr_window = values_log[phase][-window:]
    
        prev_mean = np.mean(prev_window)
        curr_mean = np.mean(curr_window)   
        delta_mean = abs(curr_mean - prev_mean)

        # Define convergence criteria
        if delta_mean < 1e-5:
            print(f"Agent {phase} VALUE function converged (stable over last {window})")
            plt.plot(values_log[phase])
            plt.title(f"Value Func-{phase}")
            plt.ylabel("Expected Discounted Reward")
            plt.show()
            return True
            
    return False