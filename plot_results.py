import matplotlib.pyplot as plt
import os

def load_scores(filename):
    """Safely load training logs."""
    if not os.path.exists(filename):
        print(f"[WARNING] File not found: {filename}")
        return [], []
    
    episodes = []
    rewards = []
    with open(filename) as f:
        next(f, None)  # skip header safely
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                episodes.append(int(parts[0]))
                rewards.append(float(parts[1]))
    return episodes, rewards


# --------------------------
# HYBRID POLICIES
# --------------------------

eps_h_ep, eps_h_rw = load_scores('hybrid_epsilon_training_log.txt')
greedy_h_ep, greedy_h_rw = load_scores('hybrid_greedy_training_log.txt')
softmax_h_ep, softmax_h_rw = load_scores('softmax_training_log.txt')  # your hybrid-softmax replacement


# --------------------------
# NON-HYBRID POLICIES
# --------------------------

eps_ep, eps_rw = load_scores('epsilon_training_log.txt')
greedy_ep, greedy_rw = load_scores('greedy_training_log.txt')
baseline_ep, baseline_rw = load_scores('training_log.txt')  # standard Q-learning baseline


# --------------------------
# PLOTTING
# --------------------------

plt.figure(figsize=(14, 10))

# ======== SUBPLOT 1: HYBRID ========
plt.subplot(2, 1, 1)
plt.plot(eps_h_ep, eps_h_rw, label='Hybrid ε–Greedy', linewidth=2)
plt.plot(greedy_h_ep, greedy_h_rw, label='Hybrid Greedy', linewidth=2)
plt.plot(softmax_h_ep, softmax_h_rw, label='Hybrid Softmax', linewidth=2)

plt.title('Hybrid Policies (A* + Approx Q-Learning)')
plt.ylabel('Score')
plt.grid(True)
plt.legend()

# ======== SUBPLOT 2: NON-HYBRID ========
plt.subplot(2, 1, 2)
plt.plot(eps_ep, eps_rw, '--', label='ε–Greedy', linewidth=2)
plt.plot(greedy_ep, greedy_rw, '--', label='Greedy', linewidth=2)
plt.plot(baseline_ep, baseline_rw, '--', label='Q-Learning Baseline', linewidth=2)

plt.title('Non-Hybrid Policies (Standard Approx Q-Learning)')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
