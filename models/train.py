import pandas as pd
from src.features.feature_engineering import create_preprocessor
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
from sklearn.model_selection import (
    train_test_split,
    cross_validate,
    StratifiedKFold,
    GridSearchCV
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import logging

logger=logging.logger(__name__)

def compare_models(x_train,y_train):
    models={
        'logistic regression' : LogisticRegression(max_iter=1000,class_weight='balanced'),
        'random forest' : RandomForestClassifier(class_weight='balanced',random_state=42)
        ,
        'Gradient boosting': GradientBoostingClassifier(random_state=42),
        'SVM' :SVC(class_weight='balanced')
    }

    cv=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    results={}
    for name , model in models.items() :
        Pipeline=Pipeline(
            steps=[
                ('preprocessor',create_preprocessor()),
                ('model',model)
            ]
        )
        scores=cross_validate(
            Pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring=['precision','recall','f1','roc_auc']
        )
        results[name]={
            "precision": scores["test_precision"].mean(),
            "recall": scores["test_recall"].mean(),
            "f1": scores["test_f1"].mean(),
            "roc_auc": scores["test_roc_auc"].mean()

        }
    results_df=pd.DataFrame(results).T
    print('model comparison')
    print(results_df.sort_values('f1',ascending=False))
    best_model_name=results_df['f1'].idxmax()
    logger.info('best model',best_model_name)
    return models[best_model_name],best_model_name

def tune_best_model(model,model_name,x_train,y_train):
    pipeline=Pipeline(steps=[('preprocessor',create_preprocessor()),
                             ('model',model)])
    param_grids = {
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1, 10]
        },
        "Random Forest": {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 10, 20]
        },
           "model__learning_rate": [0.01, 0.1],
          "Gradient Boosting": {
            "model__n_estimators": [100, 200],
           "model__max_depth": [3, 5]
        },
        "SVM": {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"]
        }
    }
    cv=StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    grid_search=GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[model_name],
        cv=cv,
        scoring='f1'
        ,n__jobs=-1,
        verbose=1
    )
    grid_search.fit(x_train,y_train)
    return grid_search.best_estimator_
def evaluate_model(model,x_test,y_test):
    predictions=model.predict(x_test)
    print('final test evaluation :')
    print(classification_report(y_test,predictions))

def save_model(model, model_name):

    model_dir = Path("models")
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_name.lower().replace(' ', '_')}.joblib"

    joblib.dump(model, model_path)

    logger.info(f"Model saved successfully: {model_path}")

    return model_path

if __name__ == "__main__":
    df = pd.read_csv("data/processed/bank_cleaned_data.csv")

    X = df.drop(columns=["fraud"])
    y = df["fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    best_model, best_model_name = compare_models(
        X_train,
        y_train
    )
    best_model = tune_best_model(
        best_model,
        best_model_name,
        X_train,
        y_train
    )
    evaluate_model(
        best_model,
        X_test,
        y_test
    )  
    save_model(best_model, best_model_name)