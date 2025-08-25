# Step 1: Import Libraries
import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

# Step 2: Load and preprocess data
df = pd.read_csv("Dell_systems.csv")
features = ['gwp_use_ratio', 'gwp_manufacturing_ratio', 'lifetime', 'screen_size', 'weight']
target = 'gwp_total'

# Normalize
scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])

# Remove outliers (IQR method)
def remove_outliers(data, cols):
    for col in cols:
        Q1, Q3 = data[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        data = data[(data[col] >= Q1 - 1.5 * IQR) & (data[col] <= Q3 + 1.5 * IQR)]
    return data

df = remove_outliers(df, features)

# Step 3: Define the PyTorch Model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.fc(x)

# Step 4: Federated Learning with K-Fold CV + Early Stopping
results = []
EPOCHS = 800
PATIENCE = 10
K = 5

for subcat in df['subcategory'].unique():
    df_sub = df[df['subcategory'] == subcat].copy()
    if len(df_sub) < K + 1:
        continue

    df_sub.reset_index(drop=True, inplace=True)
    X = df_sub[features].values
    y = df_sub[target].values
    kf = KFold(n_splits=K, shuffle=True, random_state=42)

    fold_metrics = []

    for train_idx, val_idx in kf.split(X):
        X_train = torch.tensor(X[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y[train_idx].reshape(-1, 1), dtype=torch.float32)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32)
        y_val = y[val_idx]

        model = Net()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

        best_loss = np.inf
        patience_counter = 0

        for epoch in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            output = model(X_train)
            loss = criterion(output, y_train)
            loss.backward()
            optimizer.step()

            # Early stopping
            model.eval()
            with torch.no_grad():
                val_output = model(X_val).numpy().flatten()
                val_loss = mean_squared_error(y_val, val_output)

            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                best_model = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= PATIENCE:
                    break

        # Load best model
        model.load_state_dict(best_model)
        y_val_pred = model(X_val).detach().numpy().flatten()

        fold_metrics.append({
            'MAE': mean_absolute_error(y_val, y_val_pred),
            'RMSE': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'R2': r2_score(y_val, y_val_pred),
            'Actual': y_val.mean(),
            'Predicted': y_val_pred.mean(),
            'AbsError': abs(y_val.mean() - y_val_pred.mean())
        })

    df_metrics = pd.DataFrame(fold_metrics)
    actual = df_metrics['Actual'].mean()
    predicted = df_metrics['Predicted'].mean()
    abs_error = df_metrics['AbsError'].mean()
    pct_error = abs_error / actual * 100
    accuracy = 100 - pct_error

    results.append({
        'Subcategory': subcat,
        'Samples': len(df_sub),
        'Actual GWP': round(actual, 2),
        'Predicted GWP': round(predicted, 2),  
        'Absolute Error': round(abs_error, 2),
        'Percentage Error (%)': round(pct_error, 2),
        'Accuracy (%)': round(accuracy, 2),
        'Train MAE': round(df_metrics['MAE'].mean(), 2),
        'Train RMSE': round(df_metrics['RMSE'].mean(), 2),
        'Train R2': round(df_metrics['R2'].mean(), 3)
    })

# Create results DataFrame
results_df = pd.DataFrame(results)

# Enforce order and rounding
cols_to_round = [
    'Actual GWP', 'Predicted GWP', 'Absolute Error',
    'Percentage Error (%)', 'Accuracy (%)',
    'Train MAE', 'Train RMSE', 'Train R2'
]
results_df = results_df[[
    'Subcategory', 'Samples'] + cols_to_round
]
results_df[cols_to_round] = results_df[cols_to_round].round(2)

for col in ['Actual GWP', 'Predicted GWP', 'Absolute Error', 'Percentage Error (%)', 'Accuracy (%)', 'Train MAE', 'Train RMSE', 'Train R2']:
    results_df[col] = results_df[col].round(2)

import matplotlib.pyplot as plt
import seaborn as sns

# --- Accuracy Bar Plot ---
plt.figure(figsize=(8, 6))
ax1 = sns.barplot(data=results_df, x='Subcategory', y='Accuracy (%)', palette='Blues_d')
ax1.set_title("Prediction Accuracy by Subcategory\n(Federated NN with K-Fold)", fontsize=13)
ax1.set_ylim(0, 100)
ax1.bar_label(ax1.containers[0], fmt="%.2f", padding=3)
plt.tight_layout()
plt.savefig("federated_accuracy_plot.png", dpi=300)
plt.close()

# --- Absolute Error Plot ---
plt.figure(figsize=(8, 6))
ax2 = sns.barplot(data=results_df, x='Subcategory', y='Absolute Error', palette='Oranges_d')
ax2.set_title("Prediction Error by Subcategory\n(Federated NN with K-Fold)", fontsize=13)
ax2.bar_label(ax2.containers[0], fmt="%.2f", padding=3)
plt.tight_layout()
plt.savefig("federated_error_plot.png", dpi=300)
plt.close()

# --- Table Summary Plot ---
fig, ax = plt.subplots(figsize=(16, 3))  # wider figure
ax.axis('off')

# Convert all float columns to formatted string with 2 decimal places
formatted_df = results_df.copy()
for col in formatted_df.columns:
    if formatted_df[col].dtype in ['float64', 'float32']:
        formatted_df[col] = formatted_df[col].map(lambda x: f"{x:.2f}")

# Prepare table data
columns = formatted_df.columns.tolist()
cell_data = formatted_df.values.tolist()

# Create the table
table = ax.table(cellText=cell_data,
                 colLabels=columns,
                 cellLoc='center',
                 colLoc='center',
                 loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.4, 1.7)  # Increase width for better spacing

# Bold header
for key, cell in table.get_celld().items():
    if key[0] == 0:
        cell.set_text_props(weight='bold')

plt.tight_layout()
plt.savefig("federated_results_table_fixed.png", dpi=300)
plt.close()
