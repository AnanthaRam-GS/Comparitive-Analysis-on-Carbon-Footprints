# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# Step 2: Load Dataset
df = pd.read_csv("Dell_systems.csv")

# Step 3: Outlier Removal Function (IQR method)
def remove_outliers_iqr(data, cols):
    for col in cols:
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        data = data[(data[col] >= lower) & (data[col] <= upper)]
    return data

# Step 4: Setup
features = ['gwp_use_ratio', 'gwp_manufacturing_ratio', 'lifetime', 'screen_size', 'weight']
target = 'gwp_total'
results = []

# Step 5: Hyperparameter Grid
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# Step 6: Process by Subcategory
for subcat in df['subcategory'].unique():
    df_sub = df[df['subcategory'] == subcat].copy()
    if len(df_sub) < 10:
        continue

    df_sub = remove_outliers_iqr(df_sub, features + [target])
    df_sub.reset_index(drop=True, inplace=True)

    X = df_sub[features]
    y = df_sub[target]

    # Normalize
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    all_actual = []
    all_pred = []
    
    maes = []
    rmses = []
    r2s = []

    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Train with GridSearchCV
        rf = RandomForestRegressor(random_state=42)
        grid = GridSearchCV(rf, param_grid, cv=3, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_rf = grid.best_estimator_

        y_pred = best_rf.predict(X_test)

        all_actual.extend(y_test)
        all_pred.extend(y_pred)

        # Collect fold metrics
        maes.append(mean_absolute_error(y_test, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2s.append(r2_score(y_test, y_pred))

    # Aggregate final metrics
    final_mae = np.mean(maes)
    final_rmse = np.mean(rmses)
    final_r2 = np.mean(r2s)
    
    abs_error = np.mean(np.abs(np.array(all_actual) - np.array(all_pred)))
    pct_error = (abs_error / np.mean(all_actual)) * 100
    accuracy = 100 - pct_error

    # Append results
    results.append({
        'Subcategory': subcat,
        'Samples': len(df_sub),
        'Actual GWP': round(np.mean(all_actual), 2),
        'Predicted GWP': round(np.mean(all_pred), 2),
        'Absolute Error': round(abs_error, 2),
        'Percentage Error (%)': round(pct_error, 2),
        'Accuracy (%)': round(accuracy, 2),
        'Train MAE': round(final_mae, 2),
        'Train RMSE': round(final_rmse, 2),
        'Train R2': round(final_r2, 3)
    })

# Step 7: Results Table
results_df = pd.DataFrame(results)

# Step 8: Plot Results
sns.set(style="whitegrid")
fig = plt.figure(figsize=(16, 10))
gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1.2])

# Accuracy Barplot
ax0 = plt.subplot(gs[0, 0])
sns.barplot(data=results_df, x='Subcategory', y='Accuracy (%)', palette='Greens_d', ax=ax0)
ax0.set_title("Prediction Accuracy by Subcategory (Random Forest + K-Fold)")
ax0.set_ylim(0, 100)
ax0.bar_label(ax0.containers[0], fmt="%.2f", padding=3)

# Error Barplot
ax1 = plt.subplot(gs[0, 1])
sns.barplot(data=results_df, x='Subcategory', y='Absolute Error', palette='Reds_d', ax=ax1)
ax1.set_title("Prediction Error by Subcategory (Random Forest + K-Fold)")
ax1.bar_label(ax1.containers[0], fmt="%.2f", padding=3)

# Summary Table
ax2 = plt.subplot(gs[1, :])
ax2.axis('off')
table = ax2.table(cellText=results_df.round(2).values,
                  colLabels=results_df.columns,
                  cellLoc='center',
                  loc='center')
table.scale(1, 2.2)
table.auto_set_font_size(False)
table.set_fontsize(10)

plt.tight_layout()
plt.show()
fig.savefig("randomForest_kfold_results_summary.png", dpi=300, bbox_inches='tight')
