import os
import pandas as pd


def load_and_save_raw_data(source_path, target_dir="E:/ML_Project/data/v1"):
    # ایجاد پوشه مقصد در صورت عدم وجود
    os.makedirs(target_dir, exist_ok=True)

    # بارگذاری داده‌ها
    if source_path.endswith('.xlsx') or source_path.endswith('.xls'):
        df = pd.read_excel(source_path)
    else:
        df = pd.read_csv(source_path)

    # ذخیره در پوشه v1
    target_path = os.path.join(target_dir, "v1.csv")
    df.to_csv(target_path, index=False)
    print(f"نسخه خام داده‌ها (v1) با موفقیت در {target_path} ذخیره شد.")
    return df


if __name__ == "__main__":
    source_file = "Telco_customer_churn.xlsx"
    load_and_save_raw_data(source_file)