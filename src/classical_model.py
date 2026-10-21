from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 10.0,
    gamma: str = "scale",
    kernel: str = "rbf",
    probability: bool = True,
    verbose: bool = True,
) -> Pipeline:
    if verbose:
        print(f"[SVM] Training SVM (C={C}, kernel={kernel}) on {X_train.shape[0]} samples...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            probability=probability,
            class_weight="balanced",
            random_state=42,
        )),
    ])
    pipeline.fit(X_train, y_train)

    if verbose:
        train_acc = accuracy_score(y_train, pipeline.predict(X_train))
        print(f"[SVM] Training accuracy: {train_acc:.4f}")

    return pipeline


def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    max_depth: Optional[int] = None,
    verbose: bool = True,
) -> Pipeline:
    if verbose:
        print(f"[RF] Training Random Forest ({n_estimators} trees) on {X_train.shape[0]} samples...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)

    if verbose:
        train_acc = accuracy_score(y_train, pipeline.predict(X_train))
        print(f"[RF] Training accuracy: {train_acc:.4f}")

    return pipeline


def evaluate_classifier(
    model: Pipeline,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: List[str],
    model_name: str = "Classifier",
    save_dir: Optional[str] = None,
) -> Dict[str, Any]:
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{model_name} Results:")
    print(f"Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}\n")
    print(report)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(f"{model_name} - Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig_path = Path(save_dir) / f"{model_name.replace(' ', '_').lower()}_cm.png"
        plt.savefig(fig_path, dpi=150)

    plt.show()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "report": report,
        "cm": cm,
        "y_pred": y_pred,
    }


def plot_feature_importance(
    rf_pipeline: Pipeline,
    feature_names: Optional[List[str]] = None,
    top_n: int = 20,
) -> None:
    rf = rf_pipeline.named_steps.get("rf")
    if rf is None or not hasattr(rf, "feature_importances_"):
        return

    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importances))]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(top_n), importances[indices], color="mediumseagreen")
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha="right", fontsize=9)
    ax.set_title("Random Forest - Top Feature Importances")
    ax.set_ylabel("Importance")
    plt.tight_layout()
    plt.show()


def save_model(model: Pipeline, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str) -> Pipeline:
    return joblib.load(path)


def predict_single_image(
    model: Pipeline,
    image_path: str,
    class_names: List[str],
) -> Tuple[str, float]:
    from src.dataset import extract_features_single

    feat = extract_features_single(image_path).reshape(1, -1)
    pred_idx = model.predict(feat)[0]
    predicted_class = class_names[pred_idx]

    confidence = 0.0
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(feat)[0]
        confidence = float(proba[pred_idx])

    return predicted_class, confidence
