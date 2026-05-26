# Spanish Electricity Forecasting

西班牙电网负荷与电价预测，基于 2014-2018 年 Red Eléctrica 数据。

## 方法

| 模型 | 文件 | 说明 |
|------|------|------|
| GradientBoosting | `main.py` | 24h 滞后特征 + 时间描述符，预测 2018-12-15 |
| LSTM | `question3/main.py` | 24h 滑动窗口多步多输出（负荷/电价/生物质） |

## 数据

`data_cleaned.csv`：2014-2018 小时级数据
- 14 种发电方式出力（核/光/风/气/煤等）
- 总负荷 + 日前电价 + 实际电价
- 5 城气象（巴塞罗那/毕尔巴鄂/马德里/塞维利亚/瓦伦西亚）

## 输出

- `predicted_day.csv` — GB 模型 24h 预测 + MAE 评估
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
