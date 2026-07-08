FROM python:3.10-slim
WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pandas numpy scikit-learn xgboost lightgbm catboost mlflow openpyxl skops

EXPOSE 5001

CMD ["mlflow", "models", "serve", "-m", "http://127.0.0.1:5000/#/experiments/0", "-p", "5001", "--host", "0.0.0.0", "--no-conda"]