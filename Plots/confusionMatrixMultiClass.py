import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Loading and preparing the confusion matrix data
df = pd.read_csv('confusion_matrix_40103630.csv', index_col=0)

# Ensuring the correct order of labels (X1 to X51, excluding missing labels)
all_labels = [f'X{i}' for i in range(1, 52)]
present_labels = [label for label in all_labels if label in df.index]
df = df.loc[present_labels, present_labels]

# Normalizing by row (true labels) to show proportions
df_normalized = df.div(df.sum(axis=1), axis=0)

# Creating the heatmap
plt.figure(figsize=(8, 6))  # Compact size for double-column article
sns.heatmap(df_normalized, cmap='BuGn', cbar=True, square=True, 
            xticklabels=present_labels, yticklabels=present_labels, 
            annot=False, vmin=0, vmax=1,  # Full intensity scale
            cbar_kws={'shrink': 0.8, 'pad': 0.02})  # Closer colorbar

# Setting title only (no axis labels for compactness)
#plt.title('Normalized Confusion Matrix Heatmap')

# Adjusting layout for better fit
plt.tight_layout()

# Saving the plot
plt.savefig('confusion_matrix_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()