# Spanish Electricity Forecasting

西班牙电网负荷与电价预测，基于公开发布的 2015-2018 年西班牙小时级电力与气象数据集。

## 方法

| 模型 | 文件 | 说明 |
|------|------|------|
| GradientBoosting | `main.py` | calendar 特征 + 24h 历史负荷/实际电价滞后特征；滚动 one-hour-ahead 预测 |
| LSTM | `question3/main.py` | 24h 滑动窗口多步多输出（负荷/电价/生物质） |

## 数据

`data_cleaned.csv`：2015-2018 小时级数据
- 14 种发电方式出力（核/光/风/气/煤等）
- 总负荷 + 日前电价 + 实际电价
- 5 城气象（巴塞罗那/毕尔巴鄂/马德里/塞维利亚/瓦伦西亚）

## Data source

The processed CSV files in this repository follow the schema of Kaggle's [Hourly energy demand generation and weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) dataset for Spain. They are a merged public dataset, not files collected directly from a REE API by this repository.

According to the Kaggle data card, hourly load and generation data originate from ENTSO-E, settlement-price data from the Spanish TSO Red Eléctrica de España (REE), and weather observations from the OpenWeather API for five Spanish cities. Kaggle lists the dataset release as CC0: Public Domain.

## 输出

- `predicted_day.csv` — 2018-12-15 的 24 个滚动 one-hour-ahead GB 预测；MAE 在完整 held-out 2018 period 上计算
- `question3/` — LSTM 预测对比图 (load/price/biomass)

## Gradient Boosting held-out evaluation

The Gradient Boosting model is compared with daily persistence (`prediction(t) = actual(t-24h)`) on the same 8,759 held-out 2018 timestamps.

| Target | Persistence (t-24) MAE | Gradient Boosting MAE | Improvement |
| --- | ---: | ---: | ---: |
| Load | 2520.97 MW | 464.76 MW | 81.56% |
| Price | 5.20 €/MWh | 1.73 €/MWh | 66.67% |

## 技术栈

Python + scikit-learn + TensorFlow/Keras + pandas

## 运行

```bash
pip install -r requirements.txt
python main.py
cd question3 && python main.py
```

## 项目结构

```
├── main.py               # GradientBoosting 预测
├── requirements.txt       # Python dependencies
├── data_cleaned.csv       # 原始数据
├── predicted_day.csv      # 预测结果
├── question3/
│   └── main.py            # LSTM 多步预测
└── README.md
```
