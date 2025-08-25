import pandas as pd
import numpy as np

# Read the confusion matrix from CSV
df = pd.read_csv('confusion_matrix_40103630.csv', index_col=0)

# Function to calculate F1-score for a given class
def calculate_f1_score(conf_matrix, class_name):
    TP = conf_matrix.loc[class_name, class_name]
    FP = conf_matrix[class_name].sum() - TP
    FN = conf_matrix.loc[class_name].sum() - TP
    
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1 * 100  # Convert to percentage

# Calculate F1-scores for all classes
f1_scores = {}
for class_name in df.index:
    f1_scores[class_name] = calculate_f1_score(df, class_name)

# Calculate macro-averaged F1-score
macro_f1 = np.mean(list(f1_scores.values()))

# Display F1-scores
print("F1-Scores for each satellite (%):")
for class_name, f1 in sorted(f1_scores.items(), key=lambda x: x[1], reverse=True):
    print(f"{class_name}: {f1:.1f}%")
print(f"\nMacro-averaged F1-score: {macro_f1:.1f}%")