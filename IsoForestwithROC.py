import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("dataset/KDDTrain+.txt", header=None)

# Add column names (43 columns)
columns = [f'feature_{i}' for i in range(41)] + ['label', 'difficulty']
df.columns = columns

# Optional: drop difficulty if not needed
# df = df.drop(columns=['difficulty'])

# Binary classification: normal vs anomaly
df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Encode categorical features
categorical_columns = [1, 2, 3]  # protocol_type, service, flag
for col in categorical_columns:
    le = LabelEncoder()
    df[f'feature_{col}'] = le.fit_transform(df[f'feature_{col}'])

# Feature Scaling
X = df.drop(columns=['label'])
y = df['label']
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Fit Isolation Forest
model = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
model.fit(X_scaled)
y_pred = model.predict(X_scaled)

# Map Isolation Forest output to binary
y_pred_mapped = np.where(y_pred == -1, 1, 0)

# Evaluation
from sklearn.metrics import classification_report, confusion_matrix
print("Confusion Matrix:\n", confusion_matrix(y, y_pred_mapped))
print("\nClassification Report:\n", classification_report(y, y_pred_mapped))

# Visualize Confusion Matrix
conf_matrix = confusion_matrix(y, y_pred_mapped)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=['Normal','Anomaly'], yticklabels=['Normal','Anomaly'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


#Plotting and saving in a folder

import os
print("Current Working Directory:", os.getcwd())

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Compute confusion matrix
conf_matrix = confusion_matrix(y, y_pred_mapped)

# Print evaluation
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y, y_pred_mapped))

# Create results folder
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Save confusion matrix as CSV
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Normal', 'Anomaly'], columns=['Normal', 'Anomaly'])
conf_matrix_df.to_csv(os.path.join(results_dir, "confusion_matrix.csv"))

# Plot and save confusion matrix image
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Normal','Anomaly'], yticklabels=['Normal','Anomaly'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.close()


from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assume results_dir already created:
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Compute confusion matrix
conf_matrix = confusion_matrix(y, y_pred_mapped)

# Print evaluation
print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_report(y, y_pred_mapped))

# Save confusion matrix as CSV
conf_matrix_df = pd.DataFrame(conf_matrix, index=['Normal', 'Anomaly'], columns=['Normal', 'Anomaly'])
conf_matrix_df.to_csv(os.path.join(results_dir, "confusion_matrix.csv"))

# Plot and save confusion matrix image
plt.figure(figsize=(6, 4))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(results_dir, "confusion_matrix.png"))
plt.close()

# --- ROC-AUC Curve ---
# Isolation Forest gives -1 for anomalies, 1 for normal
# We can use the decision_function for scores
scores = model.decision_function(X_scaled)
fpr, tpr, thresholds = roc_curve(y, scores * -1)  # invert score to align with labels
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(results_dir, "roc_auc_curve.png"))
plt.close()

# --- Dummy Training/Testing Loss & Accuracy Plots ---
# Since Isolation Forest does not provide these, create example plots to show how to save them:

epochs = np.arange(1, 11)
train_loss = np.random.uniform(0.4, 0.1, size=10)  # simulated decreasing loss
test_loss = np.random.uniform(0.5, 0.15, size=10)
train_acc = np.random.uniform(0.6, 0.95, size=10)  # simulated increasing accuracy
test_acc = np.random.uniform(0.55, 0.9, size=10)

# Plot Loss
plt.figure()
plt.plot(epochs, train_loss, label='Train Loss')
plt.plot(epochs, test_loss, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Testing Loss')
plt.legend()
plt.savefig(os.path.join(results_dir, "loss_plot.png"))
plt.close()

# Plot Accuracy
plt.figure()
plt.plot(epochs, train_acc, label='Train Accuracy')
plt.plot(epochs, test_acc, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Testing Accuracy')
plt.legend()
plt.savefig(os.path.join(results_dir, "accuracy_plot.png"))
plt.close()