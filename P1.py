############################################################
# STEP 1: DATA PROCESSING
############################################################
import pandas as pd

# Read CSV data into DataFrame
df = pd.read_csv("Project 1 Data.csv")

print("STEP 1: Data Processing")
print(df)

############################################################
# STEP 2: DATA VISUALIZATION
############################################################
import matplotlib.pyplot as plt
import numpy as np

print("\nSTEP 2: Data Visualization")

print("\nStatistical Analysis:")
print(df.describe())

# Histogram Plots
df[['X', 'Y', 'Z', 'Step']].hist(bins=30, figsize=(10, 4))
plt.suptitle("Histograms (X, Y, Z, Step)")
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("histograms.png")
plt.close()

# Boxplots
fig, ax = plt.subplots(figsize=(8, 4))
ax.boxplot([df['X'], df['Y'], df['Z']], labels=['X', 'Y', 'Z'])
ax.set_title("Boxplots of X, Y, Z")
ax.set_ylabel("Value")
plt.tight_layout()
plt.savefig("boxplots.png")
plt.close()

# Scatter Plots for (X,Y), (X,Z), (Y,Z)
color_map = plt.cm.get_cmap("tab20", df['Step'].nunique())
steps = np.sort(df['Step'].unique())

for (x, y) in [('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]:
    plt.figure(figsize=(6, 5))
    for i, Step in enumerate(steps):
        subset = df[df['Step'] == Step]
        plt.scatter(subset[x], subset[y], s=15, color=color_map(i), label=f"Step {Step}")
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f"{x} vs {y} colored by Step")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
    plt.tight_layout()
    plt.savefig(f"scatter_{x}_{y}.png")
    plt.close()

print("Step 2 visualization figures saved in current directory.")

############################################################
# STEP 3: CORRELATION ANALYSIS
############################################################
import seaborn as sns

print("\nSTEP 3: Correlation Analysis")

corr = df[['X', 'Y', 'Z', 'Step']].corr(method='pearson')
print(corr)

sns.heatmap(corr, annot=True, fmt=".3f", cmap='vlag', center=0)
plt.title("Pearson Correlation Matrix (X, Y, Z, Step)")
plt.tight_layout()
plt.savefig("correlation_matrix.png")
plt.close()

print("Correlation heatmap saved in current directory.")

############################################################
# STEP 4: CLASSIFICATION MODEL DEVELOPMENT / ENGINEERING
############################################################
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

print("\nSTEP 4: Model Development")

# Train-Test Split
X = df[['X', 'Y', 'Z']].values
y = df['Step'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Define models and parameter grids
models = {
    "KNN": (Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier())]),
            {"clf__n_neighbors": [3, 5, 7]}),

    "SVC": (Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True))]),
            {"clf__C": [0.1, 1, 10], "clf__kernel": ["rbf", "linear"]}),

    "RandomForest": (RandomForestClassifier(random_state=42),
                     {"n_estimators": [100, 200], "max_depth": [None, 10]}),

    "GradientBoosting": (GradientBoostingClassifier(random_state=42),
                         {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1]})
}

# Train and tune each model
best_models = {}
for name, (model, params) in models.items():
    print(f"\nTuning {name}...")
    grid = GridSearchCV(model, param_grid=params, scoring="f1_macro", cv=5, n_jobs=-1)
    grid.fit(X_train, y_train)
    print(f"Best {name} parameters:", grid.best_params_)
    print(f"Best {name} F1 (CV):", grid.best_score_)
    best_models[name] = grid.best_estimator_

print("\nAll models trained and tuned successfully.")

############################################################
# STEP 5: MODEL PERFORMANCE ANALYSIS
############################################################
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score, accuracy_score)

print("\nSTEP 5: MODEL PERFORMANCE ANALYSIS")

# Evaluate each trained model
results = []
for name, model in best_models.items():
    print(f"\nEvaluating {name}...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    })

    print(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f}")

# Summarize all results in a table
results_df = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
print("\n=== Model Performance Summary ===")
print(results_df)

# Identify best model
best_model_name = results_df.iloc[0]['Model']
best_model = best_models[best_model_name]
print(f"\nBest Performing Model: {best_model_name}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

############################################################
# STEP 6: STACKED MODEL PERFORMANCE ANALYSIS
############################################################
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

print("\nSTEP 6: STACKED MODEL PERFORMANCE ANALYSIS")

# Define stacked model using two best individual models
stack_model = StackingClassifier(
    estimators=[
        ('rf', best_models['RandomForest']),
        ('svc', best_models['SVC'])
    ],
    final_estimator=LogisticRegression(max_iter=2000, random_state=42),
    n_jobs=-1
)

# Train the stacked model
stack_model.fit(X_train, y_train)
y_stack_pred = stack_model.predict(X_test)

# Evaluate performance
stack_accuracy = accuracy_score(y_test, y_stack_pred)
stack_precision = precision_score(y_test, y_stack_pred, average='macro', zero_division=0)
stack_recall = recall_score(y_test, y_stack_pred, average='macro', zero_division=0)
stack_f1 = f1_score(y_test, y_stack_pred, average='macro', zero_division=0)

print(f"Stacked Model Accuracy:  {stack_accuracy:.3f}")
print(f"Stacked Model Precision: {stack_precision:.3f}")
print(f"Stacked Model Recall:    {stack_recall:.3f}")
print(f"Stacked Model F1 Score:  {stack_f1:.3f}")

# Confusion matrix
cm_stack = confusion_matrix(y_test, y_stack_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm_stack, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix - Stacked Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig("_confusion_matrix_stacked.png")
plt.close()

# Summary table for reporting
stack_results = pd.DataFrame([{
    "Model": "Stacked (RF + SVC)",
    "Accuracy": stack_accuracy,
    "Precision": stack_precision,
    "Recall": stack_recall,
    "F1 Score": stack_f1
}])

print("\nStacked Model Performance Summary:")
print(stack_results)

# Optionally, store in dictionary for Step 7
stacked_model = stack_model

############################################################
# STEP 7: MODEL EVALUATION
############################################################
import joblib

print("\nSTEP 7: MODEL EVALUATION")

# Save the best-performing model
model_filename = f"best_model_{best_model_name}.joblib"
joblib.dump(best_model, model_filename)

# Save the stacked model
joblib.dump(stacked_model, "stacked_model.joblib")

# Predict given coordinates
coords = np.array([
    [9.375, 3.0625, 1.51],
    [6.995, 5.125, 0.3875],
    [0.0, 3.0625, 1.93],
    [9.4, 3.0, 1.8],
    [9.4, 3.0, 1.3]
])

# Predictions using the best model
pred_best = best_model.predict(coords)
print("\nPredictions using Best Model:")
for pt, p in zip(coords, pred_best):
    print(f"{pt.tolist()} -> Predicted Step: {int(p)}")

# Predictions using the stacked model
pred_stack = stacked_model.predict(coords)
print("\nPredictions using Stacked Model:")
for pt, p in zip(coords, pred_stack):
    print(f"{pt.tolist()} -> Predicted Step: {int(p)}")