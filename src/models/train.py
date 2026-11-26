
import logging
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold,
    GridSearchCV
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from sklearn.svm import SVC

from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.features.feature_engineering import create_preprocessor


# ==========================================
# 1. LOGGING CONFIGURATION
# ==========================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================
# 2. LOAD DATA
# ==========================================

def load_data():

    logger.info("Loading cleaned dataset")

    data_path = Path("data/processed/bank_cleaned_data.csv")

    df = pd.read_csv(data_path)

    logger.info(f"Dataset shape: {df.shape}")

    X = df.drop(columns=["fraud"])
    y = df["fraud"]

    return X, y


# ==========================================
# 3. DEFINE MODELS
# ==========================================

def get_models():

    models = {

        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),

        "Gradient Boosting": GradientBoostingClassifier(
            random_state=42
        ),

        "SVM": SVC(
            class_weight="balanced",
            probability=True
        )
    }

    return models


# ==========================================
# 4. CROSS-VALIDATION MODEL COMPARISON
# ==========================================

def compare_models(X_train, y_train):

    logger.info("Starting model comparison")

    models = get_models()

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    results = {}

    for name, model in models.items():

        logger.info(f"Evaluating {name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor()),
                ("model", model)
            ]
        )

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=[
                "precision",
                "recall",
                "f1",
                "roc_auc"
            ],
            n_jobs=-1
        )

        results[name] = {
            "precision": scores["test_precision"].mean(),
            "recall": scores["test_recall"].mean(),
            "f1": scores["test_f1"].mean(),
            "roc_auc": scores["test_roc_auc"].mean()
        }

        with mlflow.start_run(
            run_name=name,
            nested=True
        ):

            mlflow.log_param("model", name)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_metrics(results[name])

    results_df = pd.DataFrame(results).T

    print("\nMODEL COMPARISON")
    print(results_df.sort_values("f1", ascending=False))

    best_model_name = results_df["f1"].idxmax()

    logger.info(f"Selected model: {best_model_name}")

    return models[best_model_name], best_model_name


# ==========================================
# 5. HYPERPARAMETER TUNING
# ==========================================

def tune_best_model(model, model_name, X_train, y_train):

    logger.info(f"Starting GridSearchCV for {model_name}")

    pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", model)
        ]
    )

    param_grids = {

        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1, 10]
        },

        "Random Forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20]
        },

        "Gradient Boosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.01, 0.1],
            "model__max_depth": [3, 5]
        },

        "SVM": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"]
        }
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[model_name],
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    logger.info(f"Best parameters: {grid_search.best_params_}")
    logger.info(f"Best CV F1: {grid_search.best_score_:.4f}")

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_f1", grid_search.best_score_)

    return grid_search.best_estimator_


# ==========================================
# 6. MODEL EVALUATION
# ==========================================

def evaluate_model(model, X_test, y_test):

    logger.info("Evaluating final model on test set")

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(X_test)[:, 1]

    metrics = {

        "test_precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "test_recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "test_f1": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),

        "test_roc_auc": roc_auc_score(
            y_test,
            probabilities
        )
    }

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    print("\nTEST METRICS")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    mlflow.log_metrics(metrics)

    return metrics


# ==========================================
# 7. SAVE TRAINED MODEL
# ==========================================

def save_model(model, model_name):

    # Root models directory, NOT src/models
    model_dir = Path("models")

    model_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    filename = model_name.lower().replace(" ", "_") + ".joblib"

    model_path = model_dir / filename

    joblib.dump(model, model_path)

    logger.info(f"Model saved successfully: {model_path}")

    return model_path


# ==========================================
# 8. MAIN PIPELINE
# ==========================================

def main():

    logger.info("Starting BankSim Fraud Detection pipeline")

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    mlflow.set_experiment("BankSim_Fraud_Detection")

    with mlflow.start_run(run_name="model_training"):

        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("cv_folds", 5)
        mlflow.log_param("selection_metric", "f1")

        # Step 1: Compare algorithms
        best_model, best_model_name = compare_models(
            X_train,
            y_train
        )

        mlflow.log_param("selected_model", best_model_name)

        # Step 2: Tune selected algorithm
        best_model = tune_best_model(
            best_model,
            best_model_name,
            X_train,
            y_train
        )

        # Step 3: Evaluate on unseen test data
        evaluate_model(
            best_model,
            X_test,
            y_test
        )

        # Step 4: Save model using Joblib
        model_path = save_model(
            best_model,
            best_model_name
        )

        # Step 5: Log model in MLflow
        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model"
        )

        mlflow.log_param("saved_model_path", str(model_path))

    logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    main()
