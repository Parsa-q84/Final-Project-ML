import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

from mlflow_utils import setup_mlflow_experiment


def train_and_evaluate_version(version="v3"):
    setup_mlflow_experiment()

    data_path = f"E:/ML_Project/data/{version}/{version}.csv"

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"فایل داده برای نسخه {version} یافت نشد. ابتدا بخش پیش‌پردازش را اجرا کنید.")

    df = pd.read_csv(data_path)
    target_col = 'Churn Value'

    X = df.drop(columns=[target_col])
    y = df[target_col]

    # تقسیم داده‌ها به آموزش و تست (حفظ نسبت کلاس‌ها با stratify)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # تعریف هایپرپارامترهای مدل
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "random_state": 42,
        "eval_metric": "logloss"
    }

    # شروع یک Run در MLflow با نام مشخص برای هر نسخه
    run_name = f"XGBoost_5Fold_CV_{version.upper()}"
    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("dataset_version", version)
        mlflow.log_params(params)

        # تعریف Stratified K-Fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        cv_accuracies = []
        cv_f1s = []

        print(f"\n--- شروع فرآیند Cross-Validation روی دیتاست نسخه {version.upper()} ---")

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

            fold_model = XGBClassifier(**params)
            fold_model.fit(X_tr, y_tr)

            preds = fold_model.predict(X_val)

            acc = accuracy_score(y_val, preds)
            f1 = f1_score(y_val, preds)

            cv_accuracies.append(acc)
            cv_f1s.append(f1)

            # ثبت معیارهای هر فولد
            mlflow.log_metric(f"fold_{fold}_accuracy", acc)
            mlflow.log_metric(f"fold_{fold}_f1", f1)

        # محاسبه و ثبت میانگین CV
        mean_accuracy = np.mean(cv_accuracies)
        mean_f1 = np.mean(cv_f1s)

        mlflow.log_metric("cv_mean_accuracy", mean_accuracy)
        mlflow.log_metric("cv_mean_f1_score", mean_f1)

        # آموزش مدل نهایی روی کل داده‌های Train این نسخه
        final_model = XGBClassifier(**params)
        final_model.fit(X_train, y_train)

        # ارزیابی روی داده‌های تست همین نسخه
        test_preds = final_model.predict(X_test)
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds)

        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_score", test_f1)

        mlflow.xgboost.log_model(final_model, name="model")

        print(f"پایان آموزش نسخه {version.upper()}. Test Accuracy: {test_acc:.4f} | Test F1: {test_f1:.4f}")


if __name__ == "__main__":
    train_and_evaluate_version("v2")
    train_and_evaluate_version("v3")