import mlflow


def setup_mlflow_experiment(experiment_name="Telco_Churn_Project"):
    """
    تنظیم و فعال‌سازی یک آزمایش (Experiment) در MLflow
    """
    # تنظیم آدرس سرور MLflow (به طور پیش‌فرض روی لوکال‌هاست)
    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    # بررسی وجود آزمایش و ساخت آن در صورت عدم وجود
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"آزمایش جدید با نام '{experiment_name}' و شناسه {experiment_id} ایجاد شد.")
    else:
        experiment_id = experiment.experiment_id
        print(f"از آزمایش موجود با نام '{experiment_name}' استفاده می‌شود.")

    mlflow.set_experiment(experiment_name)
    return experiment_id