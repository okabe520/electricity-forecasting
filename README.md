# Spanish Electricity Forecasting

西班牙电网负荷与电价预测，基于 2014-2018 年 Red Eléctrica 数据。

## 方法

| 模型 | 文件 | 说明 |
|------|------|------|
| GradientBoosting | `main.py` | calendar 特征 + 24h 历史负荷/实际电价滞后特征；滚动 one-hour-ahead 预测 |
| LSTM | `question3/main.py` | 24h 滑动窗口多步多输出（负荷/电价/生物质） |

## 数据

`data_cleaned.csv`：2014-2018 小时级数据
- 14 种发电方式出力（核/光/风/气/煤等）
- 总负荷 + 日前电价 + 实际电价
- 5 城气象（巴塞罗那/毕尔巴鄂/马德里/塞维利亚/瓦伦西亚）

## 输出

- `predicted_day.csv` — 2018-12-15 的 24 个滚动 one-hour-ahead GB 预测；MAE 在完整 held-out 2018 period 上计算
- `question3/` — LSTM 预测对比图 (load/price/biomass)

## 技术栈

Python + scikit-learn + TensorFlow/Keras + pandas

## 运行

```bash
pip install pandas scikit-learn tensorflow openpyxl
python main.py
cd question3 && python main.py
```

## 项目结构

```
├── main.py               # GradientBoosting 预测
├── data_cleaned.csv       # 原始数据
├── predicted_day.csv      # 预测结果
├── question3/
│   └── main.py            # LSTM 多步预测
└── README.md
```
