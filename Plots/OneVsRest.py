import matplotlib.pyplot as plt
import numpy as np

# Data (percentages)
rounds = np.arange(1, 11)
x4_scores = np.array([0.9644, 0.9948, 0.9976, 0.9973, 0.9956, 0.9970, 0.9974, 0.9917, 0.9953, 0.9975]) * 100
x20_scores = np.array([1, 0.9999, 0.9999, 0.9999, 0.9999, 1, 0.9999, 0.9998, 0.9999, 0.9999]) * 100
x41_scores = np.array([0.9133, 0.9252, 0.9398, 0.9401, 0.9218, 0.926, 0.9403, 0.9143, 0.9043, 0.9221]) * 100

# Scale x-axis by 0.5 to reduce spacing by 50%
rounds_scaled = rounds * 0.5

plt.figure(figsize=(10, 6))

plt.plot(rounds_scaled, x4_scores, marker='o', linestyle='-', color='blue', label='X4')
plt.plot(rounds_scaled, x20_scores, marker='s', linestyle='--', color='green', label='X20')
plt.plot(rounds_scaled, x41_scores, marker='^', linestyle='-.', color='red', label='X41')

# Annotate last points
plt.annotate(f'{x4_scores[-1]:.2f}%', (rounds_scaled[-1], x4_scores[-1]),
             textcoords="offset points", xytext=(0,-25), ha='center', color='blue', fontsize=18)  # Below the point
plt.annotate(f'{x20_scores[-1]:.2f}%', (rounds_scaled[-1], x20_scores[-1]),
             textcoords="offset points", xytext=(0,10), ha='center', color='green', fontsize=18)  # Above the point
plt.annotate(f'{x41_scores[-1]:.2f}%', (rounds_scaled[-1], x41_scores[-1]),
             textcoords="offset points", xytext=(0,10), ha='center', color='red', fontsize=18)  # Above the point

plt.xlabel('Round ID', fontsize=20)
plt.ylabel('F1-score (%)', fontsize=20)
#plt.title('F1-score over 10 Training Rounds for Satellites X4, X20, and X41', fontsize=24)

# Custom xticks at scaled positions, but labeled by original round IDs
plt.xticks(rounds_scaled, rounds, fontsize=18)
plt.yticks(fontsize=18)
plt.ylim(89, 101)
plt.grid(alpha=0.3)

plt.legend(fontsize=18)
plt.tight_layout()
plt.show()
