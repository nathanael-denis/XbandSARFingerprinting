import matplotlib.pyplot as plt

# Increase font sizes by 50%
plt.rcParams.update({
    'font.size': 15,           # Global font size
    'axes.labelsize': 15,      # X/Y label size
    'xtick.labelsize': 15,     # X tick size
    'ytick.labelsize': 15,     # Y tick size
    'legend.fontsize': 15,     # Legend text size
})

# Data from provided tables for 10 rounds
rounds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
mobilenetv2_f1 = [0.7396, 0.7396, 0.7410, 0.7410, 0.7396, 0.7426, 0.7362, 0.7362, 0.7362, 0.7396]
mobilenetv3_f1 = [0.7344, 0.7344, 0.7344, 0.7336, 0.7336, 0.7375, 0.7375, 0.7375, 0.7375, 0.7375]

# Plotting
plt.plot(rounds, mobilenetv2_f1, marker='o', color="#3c94e6", label='MobileNetv2')
plt.plot(rounds, mobilenetv3_f1, marker='s', color='#0c9b37', label='MobileNetV3-large')
plt.xlabel('Training Round')
plt.ylabel('F1-Score')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('f1_score_plot.pdf')
plt.show()
