import os
import pandas as pd
from sklearn.preprocessing import StandardScaler


def feature_engineering(input_path="E:/ML_Project/data/v2/v2.csv", target_dir="E:/ML_Project/data/v3"):
    os.makedirs(target_dir, exist_ok=True)
    df = pd.read_csv(input_path)

    # ۱. ساخت ویژگی جدید: میانگین هزینه واقعی بر اساس ماه‌های ماندگاری (tenure)
    if 'tenure' in df.columns and 'TotalCharges' in df.columns:
        # افزودن ۱ برای جلوگیری از خطای تقسیم بر صفر برای مشتریان جدید
        df['AverageCostPerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)

    # ۲. نرمال‌سازی ویژگی‌های عددی اصلی با StandardScaler
    scaler = StandardScaler()
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AverageCostPerMonth']

    # پیدا کردن ستون‌هایی که در دیتاست وجود دارند
    cols_to_scale = [col for col in numeric_features if col in df.columns]

    if cols_to_scale:
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])

    # ذخیره نسخه ۳
    target_path = os.path.join(target_dir, "v3.csv")
    df.to_csv(target_path, index=False)
    print(f"نسخه مهندسی ویژگی‌ها (v3) با موفقیت در {target_path} ذخیره شد.")
    return df


if __name__ == "__main__":
    feature_engineering()