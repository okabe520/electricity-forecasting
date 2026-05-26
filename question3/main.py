import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'SimHei'  # 设置中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负
# 1. 读取数据
df = pd.read_csv('df_final_cleaned.csv', parse_dates=['time'])
df.sort_values('time', inplace=True)
df.set_index('time', inplace=True)

# 2. 填补缺失值（如有）
df.fillna(method='ffill', inplace=True)

# 3. 选择特征和目标（多特征）
targets = ['total load actual', 'price actual','generation biomass']

features = df.columns.tolist()

# 4. 数据标准化（归一化）
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df[features])
df_scaled = pd.DataFrame(data_scaled, columns=features, index=df.index)

# 5. 准备滑动窗口数据
def create_sequences(data, target_cols, input_steps=24, output_steps=24):
    X, y = [], []
    for i in range(len(data) - input_steps - output_steps + 1):
        X.append(data.iloc[i:i+input_steps].values)
        y.append(data.iloc[i+input_steps:i+input_steps+output_steps][target_cols].values)
    return np.array(X), np.array(y)

# 6. 构建训练/测试集
last_day = df_scaled.index[-1].normalize()
test_start = df_scaled.index.get_loc(last_day)

input_steps = 24  # 使用过去24小时
output_steps = 24 # 预测未来24小时

X, y = create_sequences(df_scaled, targets, input_steps, output_steps)

# 按最后一天划分
split_idx = test_start - input_steps - output_steps + 1
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:split_idx+1], y[split_idx:split_idx+1]

# 7. 构建LSTM模型
model = Sequential()
model.add(LSTM(128, activation='relu', input_shape=(input_steps, len(features))))
model.add(Dense(output_steps * len(targets)))
model.compile(optimizer='adam', loss='mse')

# 8. 模型训练
model.fit(X_train, y_train.reshape(y_train.shape[0], -1), epochs=15, batch_size=64)

# 9. 预测
y_pred = model.predict(X_test)
y_pred = y_pred.reshape(output_steps, len(targets))

# 10. 反归一化
def inverse_transform(scaler, data, target_cols):
    dummy = np.zeros((data.shape[0], len(scaler.feature_names_in_)))
    for i, col in enumerate(target_cols):
        col_idx = list(scaler.feature_names_in_).index(col)
        dummy[:, col_idx] = data[:, i]
    return scaler.inverse_transform(dummy)[:, [list(scaler.feature_names_in_).index(col) for col in target_cols]]

y_test_inv = inverse_transform(scaler, y_test[0], targets)
y_pred_inv = inverse_transform(scaler, y_pred, targets)

# 11. 评估并可视化
for i, col in enumerate(targets):
    plt.figure(figsize=(10, 4))
    plt.plot(y_test_inv[:, i], label='actual',marker='o')
    plt.plot(y_pred_inv[:, i], label='my_forecast',marker='o')
    if col == 'price actual':
        plt.plot(df.loc[last_day: last_day + pd.Timedelta(hours=23), 'price day ahead'].values[:24], label='forecast',marker='o')
        plt.title(f' 2018-12-30 23:00后24小时 price')
    elif col == 'total load actual':
        plt.title(f' 2018-12-30 23:00后24小时 load')
    elif col == 'generation biomass':
        plt.title(f' 2018-12-30 23:00后24小时 generation biomass')

    # elif col == 'total load actual':
    #     plt.plot(df.loc[last_day: last_day + pd.Timedelta(hours=23), 'total load forecast'].values[:24], label='forecast')
    # elif col == 'generation wind offshore':
    #     plt.plot(df.loc[last_day: last_day + pd.Timedelta(hours=23), 'forecast wind offshore eday ahead'].values[:24], label='forecast')
    # elif col == 'generation wind offshore':
    #     plt.plot(df.loc[last_day: last_day + pd.Timedelta(hours=23), 'forecast wind offshore eday ahead'].values[:24], label='forecast')

    plt.legend()
    plt.grid(True)
    plt.show()
