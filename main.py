import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns

# 日志配置
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 读取数据
df = pd.read_csv("data_cleaned.csv", parse_dates=["time"])
df = df.set_index("time")

# 删除未来无法得知的字段
df = df.drop(columns=["price day ahead"])

# 添加时间特征
df["hour"] = df.index.hour
df["dayofweek"] = df.index.dayofweek
df["month"] = df.index.month
df["is_weekend"] = df["dayofweek"] >= 5

# 创建滞后特征（如前1~24小时的价格与负荷）
for lag in range(1, 25):
    df[f"load_lag_{lag}"] = df["total load actual"].shift(lag)
    df[f"price_lag_{lag}"] = df["price actual"].shift(lag)

# 删除缺失行
df = df.dropna()

# 预测目标
target_cols = ["total load actual", "price actual"]

# 构建训练集特征
feature_cols = [col for col in df.columns if col not in target_cols]

# 划分训练集与测试集（用2018年前的数据训练，2018年测试）
train_df = df[df.index < "2018-01-01"]
test_df = df[df.index >= "2018-01-01"]

X_train, y_train = train_df[feature_cols], train_df[target_cols]
X_test, y_test = test_df[feature_cols], test_df[target_cols]

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 模型训练（可替换为更复杂模型如LSTM/XGBoost）
load_model = GradientBoostingRegressor()
price_model = GradientBoostingRegressor()

logging.info("Training load model...")
load_model.fit(X_train_scaled, y_train["total load actual"])
logging.info("Training price model...")
price_model.fit(X_train_scaled, y_train["price actual"])

# 预测与评估
load_pred = load_model.predict(X_test_scaled)
price_pred = price_model.predict(X_test_scaled)

load_mae = mean_absolute_error(y_test["total load actual"], load_pred)
price_mae = mean_absolute_error(y_test["price actual"], price_pred)

logging.info(f"MAE for load prediction: {load_mae:.2f}")
logging.info(f"MAE for price prediction: {price_mae:.2f}")

# 保存未来某天的预测
future_day = "2018-12-15"
future_data = df[future_day:future_day + " 23:00:00"]
future_features = scaler.transform(future_data[feature_cols])
future_load_pred = load_model.predict(future_features)
future_price_pred = price_model.predict(future_features)

pd.DataFrame({
    "time": future_data.index,
    "predicted_load": future_load_pred,
    "predicted_price": future_price_pred
}).to_csv("predicted_day.csv", index=False)

logging.info("Prediction saved to predicted_day.csv.")

# 设置样式
sns.set(style="whitegrid")

# 选取某日真实值与预测值进行对比（以2018-12-15为例）
actual_load = y_test.loc[future_data.index, "total load actual"]
actual_price = y_test.loc[future_data.index, "price actual"]

# 创建图表
plt.figure(figsize=(14, 6))

# 子图1：负荷预测对比
plt.subplot(1, 2, 1)
plt.plot(future_data.index.hour, actual_load.values, label="Actual Load", marker='o')
plt.plot(future_data.index.hour, future_load_pred, label="Predicted Load", marker='x')
plt.title("Load Prediction vs Actual (2018-12-15)")
plt.xlabel("Hour")
plt.ylabel("Load (MW)")
plt.legend()

# 子图2：电价预测对比
plt.subplot(1, 2, 2)
plt.plot(future_data.index.hour, actual_price.values, label="Actual Price", marker='o')
plt.plot(future_data.index.hour, future_price_pred, label="Predicted Price", marker='x')
plt.title("Price Prediction vs Actual (2018-12-15)")
plt.xlabel("Hour")
plt.ylabel("Price (€/MWh)")
plt.legend()

plt.tight_layout()
plt.show()