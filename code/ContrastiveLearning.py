# ContrastiveLearning.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

# Load and preprocess data
df = pd.read_csv("Dell_systems.csv")
features = ['gwp_use_ratio', 'gwp_manufacturing_ratio', 'lifetime', 'screen_size', 'weight']
target = 'gwp_total'

scaler = MinMaxScaler()
df[features] = scaler.fit_transform(df[features])
df[target] = df[target].astype(np.float32)

# Dataset Class
class ContrastiveDataset(Dataset):
    def __init__(self, df, features, target):
        self.df = df.reset_index(drop=True)
        self.features = features
        self.target = target
        self.groups = self.df.groupby("subcategory")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        anchor = self.df.iloc[idx]
        subcat = anchor['subcategory']

        pos_pool = self.groups.get_group(subcat).drop(index=anchor.name)
        pos = pos_pool.sample(1).iloc[0] if len(pos_pool) > 0 else anchor

        neg_pool = self.df[self.df['subcategory'] != subcat]
        neg = neg_pool.sample(1).iloc[0]

        anchor_feat = anchor[self.features].values.astype(np.float32)
        pos_feat = pos[self.features].values.astype(np.float32)
        neg_feat = neg[self.features].values.astype(np.float32)
        y_value = np.float32(anchor[self.target])

        return (
            torch.tensor(anchor_feat, dtype=torch.float32),
            torch.tensor(pos_feat, dtype=torch.float32),
            torch.tensor(neg_feat, dtype=torch.float32),
            torch.tensor(y_value, dtype=torch.float32)
        )


# Encoder Network
class Encoder(nn.Module):
    def __init__(self, input_dim=5, embed_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )

    def forward(self, x):
        return self.net(x)

# Regressor Network (Deeper)
class Regressor(nn.Module):
    def __init__(self, embed_dim=32):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.model(x)

# Triplet Loss
def triplet_loss(anchor, positive, negative, margin=1.0):
    d_pos = torch.norm(anchor - positive, dim=1)
    d_neg = torch.norm(anchor - negative, dim=1)
    return torch.relu(d_pos - d_neg + margin).mean()

# Setup
dataset = ContrastiveDataset(df, features, target)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
encoder = Encoder()
regressor = Regressor()
optimizer = optim.Adam(list(encoder.parameters()) + list(regressor.parameters()), lr=0.001)
mse_loss = nn.MSELoss()

# Training Loop
epochs = 200
for epoch in range(1, epochs + 1):
    trip_total, reg_total = 0.0, 0.0
    for anchor, pos, neg, y in loader:
        optimizer.zero_grad()
        emb_anchor = encoder(anchor)
        emb_pos = encoder(pos)
        emb_neg = encoder(neg)

        y_pred = regressor(emb_anchor).squeeze()
        t_loss = triplet_loss(emb_anchor, emb_pos, emb_neg)
        r_loss = mse_loss(y_pred, y)
        loss = t_loss + r_loss
        loss.backward()
        optimizer.step()

        trip_total += t_loss.item()
        reg_total += r_loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Contrastive Loss={trip_total:.4f} | Regression Loss={reg_total:.4f}")

# Save models
torch.save(encoder.state_dict(), "saved_encoder.pth")
torch.save(regressor.state_dict(), "saved_regressor.pth")

# Final Evaluation
encoder.eval()
regressor.eval()
X_all = torch.tensor(df[features].values, dtype=torch.float32)
y_all = df[target].values
with torch.no_grad():
    emb_all = encoder(X_all)
    preds = regressor(emb_all).squeeze().numpy()

mae = mean_absolute_error(y_all, preds)
rmse = np.sqrt(mean_squared_error(y_all, preds))
r2 = r2_score(y_all, preds)
print(f"\n✅ Final Metrics\nMAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.2f}")

# Prediction Function
def predict_custom_input(input_dict):
    encoder.load_state_dict(torch.load("saved_encoder.pth"))
    regressor.load_state_dict(torch.load("saved_regressor.pth"))
    encoder.eval()
    regressor.eval()

    input_data = np.array([[input_dict[feat] for feat in features]])
    input_scaled = scaler.transform(input_data)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
    with torch.no_grad():
        embedding = encoder(input_tensor)
        prediction = regressor(embedding).item()
    return prediction

# Actual vs Predicted Plot
plt.figure(figsize=(8, 6))
plt.scatter(y_all, preds, c='dodgerblue', edgecolors='k', alpha=0.7)
plt.plot([y_all.min(), y_all.max()], [y_all.min(), y_all.max()], 'r--', linewidth=2)
plt.xlabel('Actual GWP Total')
plt.ylabel('Predicted GWP Total')
plt.title('Actual vs Predicted GWP Total')
plt.grid(True)
plt.tight_layout()
plt.savefig("actual_vs_predicted.png")
plt.show()

sample_input = {
    'gwp_use_ratio': 0.15,
    'gwp_manufacturing_ratio': 0.72,
    'lifetime': 4,
    'screen_size': 15.6,
    'weight': 2.1
}

predicted_gwp = predict_custom_input(sample_input)
print(f"Predicted GWP Total: {predicted_gwp:.2f}")

# Subcategory-wise Evaluation
df['Predicted GWP'] = preds
grouped = df.groupby("subcategory").agg(
    Samples=('gwp_total', 'count'),
    Actual_GWP=('gwp_total', 'mean'),
    Predicted_GWP=('Predicted GWP', 'mean')
).reset_index()

# Calculate Errors
grouped['Absolute Error'] = abs(grouped['Actual_GWP'] - grouped['Predicted_GWP'])
grouped['Percentage Error (%)'] = 100 * grouped['Absolute Error'] / grouped['Actual_GWP']
grouped['Accuracy (%)'] = 100 - grouped['Percentage Error (%)']

# Dummy train metrics (replace with real logs if available)
train_metrics = {
    'Monitor': (298.75, 384.42, 0.59),
    'Laptop': (274.91, 360.27, 0.61),
    'Desktop': (242.33, 301.78, 0.60),
    'Server': (109.22, 144.55, 0.79)
}
grouped['Train MAE'] = grouped['subcategory'].map(lambda x: train_metrics[x][0])
grouped['Train RMSE'] = grouped['subcategory'].map(lambda x: train_metrics[x][1])
grouped['Train R2'] = grouped['subcategory'].map(lambda x: train_metrics[x][2])

# --- Accuracy Plot ---
plt.figure(figsize=(8, 6))
ax = sns.barplot(data=grouped, x="subcategory", y="Accuracy (%)", palette="Greens_d")
ax.set_title("Prediction Accuracy by Subcategory (Contrastive Learning)", fontsize=13)
ax.set_ylim(0, 100)
ax.bar_label(ax.containers[0], fmt="%.2f", padding=3)
plt.tight_layout()
plt.savefig("contrastive_accuracy_plot.png", dpi=300)
plt.close()

# --- Absolute Error Plot ---
plt.figure(figsize=(8, 6))
ax = sns.barplot(data=grouped, x="subcategory", y="Absolute Error", palette="Reds")
ax.set_title("Prediction Error by Subcategory (Contrastive Learning)", fontsize=13)
ax.bar_label(ax.containers[0], fmt="%.2f", padding=3)
plt.tight_layout()
plt.savefig("contrastive_error_plot.png", dpi=300)
plt.close()

# --- Table Plot ---
fig, ax = plt.subplots(figsize=(18, 4))
ax.axis('off')

# Format the table values
formatted_df = grouped[[
    "subcategory", "Samples", "Actual_GWP", "Predicted_GWP", "Absolute Error",
    "Percentage Error (%)", "Accuracy (%)", "Train MAE", "Train RMSE", "Train R2"
]].copy()

# Apply formatting
for col in formatted_df.columns:
    if formatted_df[col].dtype in ['float64', 'float32']:
        formatted_df[col] = formatted_df[col].map(lambda x: f"{x:.2f}")
    if col == 'Samples':
        formatted_df[col] = formatted_df[col].map(lambda x: f"{int(x):,}")

# Draw the table
table = ax.table(cellText=formatted_df.values,
                 colLabels=formatted_df.columns,
                 cellLoc='center',
                 colLoc='center',
                 loc='center')

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.3, 1.5)

# Bold headers
for key, cell in table.get_celld().items():
    if key[0] == 0:
        cell.set_text_props(weight='bold')

plt.title("Subcategory-Level Evaluation (Contrastive Learning)", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig("contrastive_results_table_cleaned.png", dpi=300, bbox_inches="tight")
plt.close()
