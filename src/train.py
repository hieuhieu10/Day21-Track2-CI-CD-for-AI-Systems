import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

EVAL_THRESHOLD = 0.70


def _configure_mlflow() -> None:
    if os.environ.get("CI"):
        return
    mlflow.set_tracking_uri(
        os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )


def train(
    params: dict,
    data_path: str = "data/train_phase1.csv",
    eval_path: str = "data/eval.csv",
) -> float:
    """
    Huan luyen mo hinh va ghi nhan ket qua vao MLflow.

    Tham so:
        params     : dict chua cac sieu tham so cho RandomForestClassifier.
        data_path  : duong dan den file du lieu huan luyen.
        eval_path  : duong dan den file du lieu danh gia.

    Tra ve:
        accuracy (float): do chinh xac tren tap danh gia.
    """
    df_train = pd.read_csv(data_path)
    df_eval = pd.read_csv(eval_path)

    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]
    X_eval = df_eval.drop(columns=["target"])
    y_eval = df_eval["target"]

    model = RandomForestClassifier(**params, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_eval)
    acc = accuracy_score(y_eval, preds)
    f1 = f1_score(y_eval, preds, average="weighted")

    print(f"Accuracy: {acc:.4f} | F1: {f1:.4f}")

    if not os.environ.get("CI"):
        _configure_mlflow()
        with mlflow.start_run():
            mlflow.log_params(params)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.sklearn.log_model(model, "model")

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/metrics.json", "w") as f:
        json.dump({"accuracy": acc, "f1_score": f1}, f)

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    return acc


if __name__ == "__main__":
    with open("params.yaml") as f:
        params = yaml.safe_load(f)
    train(params)
