import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# وارد کردن توابع مراحل قبلی به عنوان گام‌های پایپ‌لاین
from data_loader import load_and_save_raw_data
from preprocessing import clean_data
from features import feature_engineering
from train import train_and_evaluate_version
from evaluate import evaluate_model_by_version


def run_entire_pipeline(source_file_path):
    print("=" * 60)
    print("🚀 شروع فرآیند خودکار خط لوله MLOps (Run Pipeline) 🚀")
    print("=" * 60)

    try:
        print("\n[گام ۱/۵] در حال بارگذاری و نسخه‌بندی داده‌های خام (v1)...")
        load_and_save_raw_data(source_file_path)

        print("\n[گام ۲/۵] در حال پاک‌سازی و آماده‌سازی نسخه ۲ داده‌ها (v2)...")
        clean_data(input_path="E:/ML_Project/data/v1/v1.csv", target_dir="E:/ML_Project/data/v2")

        print("\n[گام ۳/۵] در حال اعمال مهندسی ویژگی‌ها و تولید نسخه ۳ (v3)...")
        feature_engineering(input_path="E:/ML_Project/data/v2/v2.csv", target_dir="E:/ML_Project/data/v3")

        print("\n[گام ۴/۵] شروع فرآیند آموزش مدل با 5-Fold CV برای نسخه‌های دیتاست...")
        print("-> آموزش روی نسخه ۲ (پاک‌سازی شده)...")
        train_and_evaluate_version("v2")
        print("-> آموزش روی نسخه ۳ (مهندسی ویژگی‌ها)...")
        train_and_evaluate_version("v3")

        print("\n[گام ۵/۵] شروع ارزیابی نهایی مدل‌ها و ثبت نمودارها در MLflow...")
        print("-> ارزیابی مدل نسخه ۲...")
        evaluate_model_by_version("v2")
        print("-> ارزیابی مدل نسخه ۳...")
        evaluate_model_by_version("v3")

        print("\n" + "=" * 60)
        print("✅ خط لوله با موفقیت کامل اجرا شد و تمام نتایج در MLflow ثبت شدند.")
        print("=" * 60)

    except Exception as e:
        print("\n❌ خطا در اجرای خط لوله رخ داده است:")
        print(str(e))
        sys.exit(1)


if __name__ == "__main__":
    main_dataset_path = "Telco_customer_churn.xlsx"

    run_entire_pipeline(main_dataset_path)