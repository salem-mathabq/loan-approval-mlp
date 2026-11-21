"""
STINK3014 Assignment 01 - MLP Loan Approval
Run: python solution.py
This script runs the baseline, optimiser comparison, hyperparameter tuning,
regularisation and dropout experiments, evaluates a final model, and saves
artifacts for the Streamlit app.
"""
import os, time, json, random, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

warnings.filterwarnings("ignore")
SEED = 42
DATA_PATH = "loan_approval_dataset.csv"
OUT = "outputs"
os.makedirs(OUT, exist_ok=True)

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ---------- 1. Load and prepare data ----------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
for c in ["education", "self_employed", "loan_status"]:
    df[c] = df[c].astype(str).str.strip()

print("\nDATASET SHAPE:", df.shape)
print("\nDATA TYPES:\n", df.dtypes)
print("\nMISSING VALUES:\n", df.isna().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\nTARGET DISTRIBUTION:\n", df["loan_status"].value_counts())

# Remove exact duplicate rows and the non-predictive identifier.
df = df.drop_duplicates().copy()
y = df["loan_status"].map({"Approved": 1, "Rejected": 0})
if y.isna().any():
    raise ValueError("Unexpected loan_status labels. Check the CSV values.")

# Encode the two categorical inputs. Strip spaces before mapping.
df["education"] = df["education"].map({"Graduate": 1, "Not Graduate": 0})
df["self_employed"] = df["self_employed"].map({"Yes": 1, "No": 0})
if df[["education", "self_employed"]].isna().any().any():
    raise ValueError("Unexpected categorical values in education/self_employed.")

feature_cols = [
    "no_of_dependents", "education", "self_employed", "income_annum",
    "loan_amount", "loan_term", "cibil_score", "residential_assets_value",
    "commercial_assets_value", "luxury_assets_value", "bank_asset_value"
]
X = df[feature_cols].astype("float32")
y = y.astype("int32")

# First reserve 30% as held-out test set, stratified by target.
X_train_all, X_test, y_train_all, y_test = train_test_split(
    X, y, test_size=0.30, random_state=SEED, stratify=y
)
# Take validation only from the 70% training portion.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_all, y_train_all, test_size=0.20, random_state=SEED,
    stratify=y_train_all
)

# Fit scaler on training data only, then apply to validation/test.
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype("float32")
X_val_s = scaler.transform(X_val).astype("float32")
X_test_s = scaler.transform(X_test).astype("float32")

joblib.dump(scaler, os.path.join(OUT, "scaler.joblib"))
with open(os.path.join(OUT, "feature_columns.json"), "w") as f:
    json.dump(feature_cols, f, indent=2)

print(f"\nTrain={len(X_train)}, validation={len(X_val)}, test={len(X_test)}")
print("Target encoding: Approved=1, Rejected=0")
print("Features used:", feature_cols)

# ---------- Helpers ----------
def make_optimizer(name, lr):
    name = name.lower()
    if name == "sgd":
        return tf.keras.optimizers.SGD(learning_rate=lr)
    if name == "momentum":
        return tf.keras.optimizers.SGD(
            learning_rate=lr, momentum=0.9, nesterov=False
        )
    if name == "adagrad":
        return tf.keras.optimizers.Adagrad(
            learning_rate=lr, initial_accumulator_value=0.1, epsilon=1e-7
        )
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(
            learning_rate=lr, rho=0.9, epsilon=1e-7
        )
    if name == "adam":
        return tf.keras.optimizers.Adam(
            learning_rate=lr, beta_1=0.9, beta_2=0.99, epsilon=1e-7
        )
    raise ValueError("Unknown optimizer: " + name)

def make_model(hidden=(32, 16), optimizer="adam", lr=0.001,
               regularizer=None, reg_strength=0.0, dropout=0.0):
    reg = None
    if regularizer == "l1":
        reg = tf.keras.regularizers.L1(reg_strength)
    elif regularizer == "l2":
        reg = tf.keras.regularizers.L2(reg_strength)
    elif regularizer == "elastic":
        # Elastic Net = alpha*L1 + (1-alpha)*L2; alpha=1.0 is pure L1.
        alpha = 1.0
        reg = tf.keras.regularizers.L1L2(
            l1=reg_strength * alpha, l2=reg_strength * (1-alpha)
        )

    model = tf.keras.Sequential(name="loan_approval_mlp")
    model.add(tf.keras.layers.Input(shape=(len(feature_cols),)))
    for units in hidden:
        model.add(tf.keras.layers.Dense(
            units, activation="relu", kernel_regularizer=reg
        ))
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
    model.compile(
        optimizer=make_optimizer(optimizer, lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

def train_model(model, epochs=500, batch_size=32, verbose=0):
    start = time.perf_counter()
    history = model.fit(
        X_train_s, y_train.to_numpy(),
        validation_data=(X_val_s, y_val.to_numpy()),
        epochs=epochs, batch_size=batch_size, verbose=verbose,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=35, restore_best_weights=True
            )
        ]
    )
    elapsed = time.perf_counter() - start
    return history, elapsed

def evaluate(model, x, y_true):
    probs = model.predict(x, verbose=0).ravel()
    pred = (probs >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probs),
    }

def plot_history(history, title, filename):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history.history["loss"], label="Training loss")
    ax.plot(history.history["val_loss"], label="Validation loss")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Binary cross-entropy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, filename), dpi=150)
    plt.close(fig)

def run_experiment(label, **kwargs):
    model = make_model(**kwargs)
    history, seconds = train_model(
        model, epochs=kwargs.pop("epochs", 500) if "epochs" in kwargs else 500,
        batch_size=kwargs.pop("batch_size", 32) if "batch_size" in kwargs else 32
    )
    val_metrics = evaluate(model, X_val_s, y_val)
    row = {"experiment": label, "epochs_run": len(history.history["loss"]),
           "training_seconds": seconds, **{f"val_{k}": v for k,v in val_metrics.items()}}
    return model, history, row

# ---------- Task 1: Baseline ----------
baseline = make_model(hidden=(32,16), optimizer="adam", lr=0.001)
baseline_history, baseline_seconds = train_model(baseline, epochs=500, batch_size=32)
plot_history(baseline_history, "Baseline MLP: training vs validation loss", "baseline_loss.png")
baseline_val = evaluate(baseline, X_val_s, y_val)
print("\nBASELINE validation:", baseline_val)

# ---------- Task 2: Optimiser comparison ----------
optim_rows = []
optim_models = {}
for opt, lr in [
    ("sgd", 0.01), ("momentum", 0.01), ("adagrad", 0.001),
    ("rmsprop", 0.001), ("adam", 0.001)
]:
    print("Optimiser experiment:", opt)
    m = make_model(hidden=(32,16), optimizer=opt, lr=lr)
    h, seconds = train_model(m, epochs=500, batch_size=32)
    vm = evaluate(m, X_val_s, y_val)
    optim_rows.append({
        "optimizer": opt, "learning_rate": lr,
        "epochs_run": len(h.history["loss"]), "training_seconds": seconds,
        "best_val_loss": float(np.min(h.history["val_loss"])),
        **{f"val_{k}":v for k,v in vm.items()}
    })
    optim_models[opt] = (m, h)
    plot_history(h, f"Optimizer: {opt}", f"optimizer_{opt}_loss.png")
optim_df = pd.DataFrame(optim_rows)
optim_df.to_csv(os.path.join(OUT, "optimizer_comparison.csv"), index=False)
print("\nOPTIMISER COMPARISON:\n", optim_df.to_string(index=False))

# ---------- Task 3: Hyperparameter tuning (8+ configurations) ----------
configs = [
    {"hidden": (8,), "optimizer":"adam", "lr":0.001, "batch_size":16},
    {"hidden": (16,), "optimizer":"adam", "lr":0.001, "batch_size":32},
    {"hidden": (32,16), "optimizer":"adam", "lr":0.001, "batch_size":32},
    {"hidden": (64,32), "optimizer":"adam", "lr":0.001, "batch_size":32},
    {"hidden": (32,16), "optimizer":"momentum", "lr":0.01, "batch_size":32},
    {"hidden": (32,16), "optimizer":"rmsprop", "lr":0.001, "batch_size":64},
    {"hidden": (32,16,8), "optimizer":"adam", "lr":0.001, "batch_size":64},
    {"hidden": (64,32,16), "optimizer":"adam", "lr":0.01, "batch_size":16},
]
tune_rows = []
tune_models = []
for i, cfg in enumerate(configs, 1):
    print(f"Tuning config {i}/{len(configs)}:", cfg)
    model = make_model(hidden=cfg["hidden"], optimizer=cfg["optimizer"], lr=cfg["lr"])
    hist, sec = train_model(model, epochs=500, batch_size=cfg["batch_size"])
    vm = evaluate(model, X_val_s, y_val)
    row = {
        "config_id": i, "hidden_layers": str(cfg["hidden"]),
        "optimizer": cfg["optimizer"], "learning_rate": cfg["lr"],
        "batch_size": cfg["batch_size"], "epochs_run": len(hist.history["loss"]),
        "training_seconds": sec, "best_val_loss": float(np.min(hist.history["val_loss"])),
        **{f"val_{k}":v for k,v in vm.items()}
    }
    tune_rows.append(row)
    tune_models.append((model, hist, cfg, row))
tune_df = pd.DataFrame(tune_rows)
tune_df.to_csv(os.path.join(OUT, "hyperparameter_tuning.csv"), index=False)
# Select using validation ROC-AUC, never test results.
best_tune_idx = int(tune_df["val_roc_auc"].idxmax())
best_tune_model, best_tune_history, best_tune_cfg, best_tune_row = tune_models[best_tune_idx]
print("\nTUNING RESULTS:\n", tune_df.to_string(index=False))
print("\nBest configuration by validation ROC-AUC:", best_tune_cfg)

# ---------- Task 4: Regularisation comparison ----------
reg_specs = [
    ("none", None, 0.0), ("l1_0.01", "l1", 0.01),
    ("l1_0.5", "l1", 0.5), ("l2_0.01", "l2", 0.01),
    ("l2_0.5", "l2", 0.5), ("elastic_0.5_alpha1", "elastic", 0.5)
]
reg_rows, reg_models = [], {}
for label, reg, strength in reg_specs:
    print("Regularisation experiment:", label)
    m = make_model(hidden=(32,16), optimizer="adam", lr=0.001,
                   regularizer=reg, reg_strength=strength)
    h, sec = train_model(m, epochs=500, batch_size=32)
    vm = evaluate(m, X_val_s, y_val)
    reg_rows.append({
        "model": label, "regularizer": str(reg), "strength": strength,
        "epochs_run": len(h.history["loss"]), "training_seconds": sec,
        "best_val_loss": float(np.min(h.history["val_loss"])),
        **{f"val_{k}":v for k,v in vm.items()}
    })
    reg_models[label] = (m,h)
reg_df = pd.DataFrame(reg_rows)
reg_df.to_csv(os.path.join(OUT, "regularisation_comparison.csv"), index=False)
print("\nREGULARISATION:\n", reg_df.to_string(index=False))

# ---------- Task 5: Dropout comparison ----------
drop_rows, drop_models = [], {}
for rate in [0.0, 0.2, 0.5]:
    print("Dropout experiment:", rate)
    m = make_model(hidden=(32,16), optimizer="adam", lr=0.001, dropout=rate)
    h, sec = train_model(m, epochs=500, batch_size=32)
    vm = evaluate(m, X_val_s, y_val)
    drop_rows.append({
        "dropout_rate": rate, "epochs_run": len(h.history["loss"]),
        "training_seconds": sec, "best_val_loss": float(np.min(h.history["val_loss"])),
        **{f"val_{k}":v for k,v in vm.items()}
    })
    drop_models[rate] = (m,h)
drop_df = pd.DataFrame(drop_rows)
drop_df.to_csv(os.path.join(OUT, "dropout_comparison.csv"), index=False)
print("\nDROPOUT:\n", drop_df.to_string(index=False))

# ---------- Task 6: Improved model selection and held-out test evaluation ----------
# Pick the candidate with the best validation ROC-AUC among tuning, regularisation,
# and dropout runs. Test set is used only once for final evaluation.

# Build the improved model combining selected settings,
# L2 regularization, and dropout.

best_model = make_model(
    hidden=(32, 16),
    optimizer="momentum",
    lr=0.01,
    regularizer="l2",
    reg_strength=0.01,
    dropout=0.2
)

# Train using training data and monitor validation data.
best_hist, final_training_seconds = train_model(
    best_model,
    epochs=500,
    batch_size=32,
    verbose=0
)

best_name = "combined_momentum_l2_0.01_dropout_0.2"

# Evaluate on validation data.
val_metrics = evaluate(best_model, X_val_s, y_val)

# Evaluate on held-out test data after model selection.
test_metrics = evaluate(best_model, X_test_s, y_test)

print("\nSELECTED FINAL MODEL:", best_name)
print("VALIDATION METRICS:", val_metrics)
print("HELD-OUT TEST METRICS:", test_metrics)
print("FINAL TRAINING TIME:", final_training_seconds)

test_pred = (
    best_model.predict(X_test_s, verbose=0).ravel() >= 0.5
).astype(int)

print("\nTEST CLASSIFICATION REPORT:\n",
      classification_report(
          y_test,
          test_pred,
          target_names=["Rejected", "Approved"],
          zero_division=0
      ))

cm = confusion_matrix(y_test, test_pred)

fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Rejected", "Approved"],
    yticklabels=["Rejected", "Approved"],
    ax=ax
)
ax.set_xlabel("Predicted label")
ax.set_ylabel("True label")
ax.set_title("Final model confusion matrix")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "confusion_matrix.png"), dpi=150)
plt.close(fig)

plot_history(
    best_hist,
    f"Final model ({best_name})",
    "final_loss.png"
)

best_model.save(os.path.join(OUT, "best_model.keras"))

with open(os.path.join(OUT, "final_metrics.json"), "w") as f:
    json.dump({
        "selected_model": best_name,
        "validation_selected_by": "Preselected hyperparameters; validation monitored",
        "validation_metrics": {
            k: float(v) for k, v in val_metrics.items()
        },
        "test_metrics": {
            k: float(v) for k, v in test_metrics.items()
        },
        "training_seconds": float(final_training_seconds),
        "hyperparameters": {
            "hidden_layers": [32, 16],
            "optimizer": "momentum",
            "learning_rate": 0.01,
            "batch_size": 32,
            "regularizer": "l2",
            "regularization_strength": 0.01,
            "dropout": 0.2
        },
        "seed": SEED,
        "tensorflow_version": tf.__version__
    }, f, indent=2)

print("\nSaved improved model and evaluation results in:", OUT)
print("Use app.py with the outputs folder to run the deployment interface.")


best_model.save(os.path.join(OUT, "best_model.keras"))
with open(os.path.join(OUT, "final_metrics.json"), "w") as f:
    json.dump({"selected_model": best_name, "validation_selected_by": "ROC-AUC",
               "test_metrics": {k: float(v) for k,v in test_metrics.items()},
               "seed": SEED, "tensorflow_version": tf.__version__}, f, indent=2)

print("\nSaved model, scaler, metrics, CSV tables and plots in:", OUT)
print("Use app.py with the outputs folder to run the deployment interface.")
