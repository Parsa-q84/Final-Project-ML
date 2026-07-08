import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def evaluate_model_by_version(version="v3"):
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    data_path = f"E:/ML_Project/data/{version}/{version}.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"داده‌های نسخه {version} یافت نشد.")

    df = pd.read_csv(data_path)
    target_col = 'Churn Value'

    X = df.drop(columns=[target_col])
    y = df[target_col]

    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    experiment = mlflow.get_experiment_by_name("Telco_Churn_Project")
    if experiment is None:
        raise ValueError("آزمایش یافت نشد.")

    filter_string = f"params.dataset_version = '{version}'"
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=filter_string,
        order_by=["start_time DESC"]
    )

    if runs.empty:
        print(f"هیچ رانی برای نسخه {version} پیدا نشد. ابتدا کد آموزش را برای این نسخه اجرا کنید.")
        return

    latest_run_id = runs.iloc[0]["run_id"]
    print(f"\n[ارزیابی] در حال بارگذاری مدل نسخه {version.upper()} از ران: {latest_run_id}")

    model_uri = f"runs:/{latest_run_id}/model"
    model = mlflow.xgboost.load_model(model_uri)

    # باز کردن ران جهت ثبت آرتیفکت ارزیابی نهایی
    with mlflow.start_run(run_id=latest_run_id):
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred)

        print(f"--- گزارش طبقه‌بندی نسخه {version.upper()} ---")
        print(report)

        # رسم ماتریس آشفتگی
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges' if version == 'v2' else 'Blues',
                    xticklabels=['Stayed (0)', 'Churned (1)'],
                    yticklabels=['Stayed (0)', 'Churned (1)'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title(f'Confusion Matrix - Churn {version.upper()}')

        plot_path = f"confusion_matrix_{version}.png"
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        mlflow.log_artifact(plot_path)

        if os.path.exists(plot_path):
            os.remove(plot_path)

        print(f"نمودار ارزیابی نسخه {version.upper()} با موفقیت در MLflow ثبت شد.")


if __name__ == "__main__":
    evaluate_model_by_version("v2")
    evaluate_model_by_version("v3")