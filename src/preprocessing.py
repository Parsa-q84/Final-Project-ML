import os
import pandas as pd
import numpy as np


def clean_data(input_path="E:/ML_Project/data/v1/v1.csv", target_dir="E:/ML_Project/data/v2"):
    os.makedirs(target_dir, exist_ok=True)
    df = pd.read_csv(input_path)

    # ۱. حذف ستون‌های شناسه، جغرافیایی و متنی غیرضروری برای مدل‌سازی
    cols_to_drop = [
        'CustomerID', 'Count', 'Country', 'State', 'City', 'Zip Code',
        'Lat Long', 'Latitude', 'Longitude', 'Churn Label', 'Churn Reason'
    ]

    existing_drops = [col for col in cols_to_drop if col in df.columns]
    df = df.drop(columns=existing_drops)

    # ۲. رفع مقادیر گم‌شده در Total Charges
    if 'Total Charges' in df.columns:
        # جایگزینی فضاهای خالی احتمالی با NaN
        df['Total Charges'] = df['Total Charges'].replace(r'^\s*$', np.nan, regex=True)
        df['Total Charges'] = pd.to_numeric(df['Total Charges'])
        # پر کردن مقادیر خالی با میانه ستون
        df['Total Charges'] = df['Total Charges'].fillna(df['Total Charges'].median())


    categorical_cols = df.select_dtypes(include=['object', 'string']).columns.tolist()

    # اعمال One-Hot Encoding روی ستون‌های کاتگوریکال یافته شده
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # تبدیل ستون‌های True/False حاصل از get_dummies به 0 و 1 عددی
    for col in df.columns:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)

    # ذخیره نسخه ۲
    target_path = os.path.join(target_dir, "v2.csv")
    df.to_csv(target_path, index=False)
    print(f"نسخه پاک‌سازی شده داده‌ها (v2) با موفقیت در {target_path} ذخیره شد.")
    return df


if __name__ == "__main__":
    clean_data()