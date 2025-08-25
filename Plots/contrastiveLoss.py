import matplotlib.pyplot as plt

# SupCon loss data for 125 epochs
epochs = list(range(1, 126))
supcon_losses = [
    2.1909, 1.8156, 1.7010, 1.6407, 1.5933, 1.5615, 1.5307, 1.5136, 1.4884, 1.4767,
    1.4617, 1.4484, 1.4399, 1.4194, 1.4144, 1.4036, 1.3936, 1.3802, 1.3866, 1.3730,
    1.3727, 1.3566, 1.3584, 1.3478, 1.3491, 1.3299, 1.3274, 1.3256, 1.3197, 1.3198,
    1.3055, 1.3055, 1.2953, 1.2967, 1.2952, 1.2839, 1.2787, 1.2647, 1.2673, 1.2535,
    1.2584, 1.2492, 1.2464, 1.2419, 1.2342, 1.2298, 1.2183, 1.2282, 1.2168, 1.2144,
    1.2163, 1.1999, 1.1918, 1.1937, 1.1820, 1.1883, 1.1766, 1.1697, 1.1827, 1.1650,
    1.1591, 1.1494, 1.1511, 1.1459, 1.1406, 1.1338, 1.1392, 1.1305, 1.1376, 1.1222,
    1.1218, 1.1203, 1.1148, 1.1114, 1.1033, 1.1025, 1.1011, 1.0948, 1.0979, 1.0957,
    1.0852, 1.0834, 1.0818, 1.0759, 1.0763, 1.0707, 1.0701, 1.0614, 1.0561, 1.0704,
    1.0536, 1.0612, 1.0560, 1.0505, 1.0416, 1.0499, 1.0441, 1.0428, 1.0352, 1.0383,
    1.0306, 1.0312, 1.0285, 1.0247, 1.0283, 1.0261, 1.0194, 1.0180, 1.0154, 1.0137,
    1.0104, 1.0129, 1.0049, 1.0024, 1.0033, 1.0014, 1.0004, 0.9975, 1.0018, 0.9944,
    0.9951, 0.9913, 0.9883, 0.9846, 0.9859
]

# F1-score data for specific epochs
f1_epochs = [40, 50, 100, 125]
f1_scores = [0.7678, 0.7632, 0.7479333333, 0.7383]

# Create the plot with dual y-axes
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot SupCon loss on the primary y-axis
ax1.plot(epochs, supcon_losses, marker='o', linestyle='-', color='#589fc9', markersize=4, label='SupCon Loss')
ax1.set_xlabel('Epoch', fontsize=16)
ax1.set_ylabel('SupCon Loss', fontsize=16, color="#000000")
ax1.tick_params(axis='y', labelcolor='#589fc9',labelsize=16)
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_xticks(range(0, 151, 25))  # Show ticks at intervals of 25 epochs, up to 150
ax1.set_ylim(0.9, 2.3)  # Adjust y-axis for SupCon loss range

# Create secondary y-axis for F1-score
ax2 = ax1.twinx()
ax2.plot(f1_epochs, f1_scores, marker='s', linestyle='-', color='#52c474', markersize=14, label='F1-Score')
ax2.set_ylabel('F1-Score', fontsize=16, color="#000000")
ax2.tick_params(axis='y', labelcolor='#52c474', labelsize=16)
ax2.set_ylim(0.7, 0.8)  # Adjust y-axis for F1-score range

# Add labels directly on F1-score points
for i, (epoch, f1) in enumerate(zip(f1_epochs, f1_scores)):
    ax2.annotate(f'{f1:.4f}', (epoch, f1), textcoords="offset points", xytext=(0, 10), ha='center', color='#52c474', fontsize=15)

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Set title and layout
#plt.title('SupCon Loss and F1-Score over Epochs for SCULLY Methodology', fontsize=18)
plt.tight_layout()

# Save the plot
plt.savefig('supcon_loss_f1_plot.png', dpi=300)
plt.close()