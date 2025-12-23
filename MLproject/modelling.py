import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import os
import sys
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Ambil parameter dari sys.argv atau default
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 505
    max_depth    = int(sys.argv[2]) if len(sys.argv) > 2 else 37
    file_path    = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_pca.csv")

    data = pd.read_csv(file_path)

    X_train, X_test, y_train, y_test = train_test_split(
        data.drop("Credit_Score", axis=1),
        data["Credit_Score"],
        random_state=42,
        test_size=0.2,
        stratify=data["Credit_Score"]
    )

    input_example = X_train.iloc[0:5]

    with mlflow.start_run(run_name="CreditScoring-RF"):
        mlflow.sklearn.autolog()  # otomatis log banyak hal

        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)

        accuracy = model.score(X_test, y_test)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=input_example
        )

    print(f"Training selesai! Accuracy: {accuracy:.4f}")
