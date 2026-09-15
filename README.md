# Spanish Electricity Load and Price Forecasting

A Python project for hourly electricity-load and electricity-price forecasting using a public Spanish energy and weather dataset covering 2015–2018. The validated path is a Gradient Boosting model for rolling one-hour-ahead forecasts; the repository also includes a separate experimental LSTM implementation.

## Validated Gradient Boosting Forecast

`main.py` forecasts load and actual electricity price at time `t` using only information available by the forecast origin:

- calendar features: hour, day of week, month, and weekend indicator;
- the preceding 24 hours of observed load; and
- the preceding 24 hours of observed actual price.

The feature set intentionally excludes contemporaneous generation, observed weather, and other target-time system variables. The task is **rolling one-hour-ahead forecasting**, not fixed-origin day-ahead forecasting.

## Held-out Evaluation

Training uses observations before 2018-01-01; evaluation uses the chronological 2018 hold-out period. The scaler is fit on the training partition only.

Gradient Boosting is evaluated against daily persistence, where `prediction(t) = actual(t - 24h)`, on the same **8,759** held-out timestamps.

| Target | Persistence (t-24) MAE | Gradient Boosting MAE | Improvement |
| --- | ---: | ---: | ---: |
| Load | 2520.97 MW | 464.76 MW | 81.56% |
| Price | 5.20 €/MWh | 1.73 €/MWh | 66.67% |

`predicted_day.csv` contains the 24 rolling one-hour-ahead Gradient Boosting predictions for 2018-12-15. The MAE values above are calculated over the full held-out 2018 period.

## Experimental LSTM

`question3/main.py` contains a 24-hour-input, 24-hour multi-output forecasting experiment for load, price, and biomass generation. It is retained as an **experimental implementation**; it is not presented here as a validated held-out forecasting result.

## Dataset

The processed CSV files follow the schema of Kaggle's [Hourly energy demand generation and weather](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather) dataset for Spain. This repository uses a processed, merged public dataset and does not collect data directly from a REE API.

According to the Kaggle data card, the merged dataset includes:

- ENTSO-E load and generation data;
- settlement-price data from Red Eléctrica de España (REE); and
- OpenWeather API observations for five Spanish cities.

Kaggle lists that dataset release as CC0: Public Domain.

## Tech Stack

- Python
- pandas and NumPy
- scikit-learn
- TensorFlow/Keras
- matplotlib and seaborn

## Installation and Usage

```bash
pip install -r requirements.txt
python main.py
```

To run the experimental LSTM script:

```bash
cd question3
python main.py
```

## Repository Structure

```text
├── main.py               # Validated Gradient Boosting forecasting pipeline
├── requirements.txt       # Python dependencies
├── data_cleaned.csv       # Processed hourly Spanish energy and weather data
├── predicted_day.csv      # Gradient Boosting predictions for 2018-12-15
├── question3/
│   └── main.py            # Experimental LSTM forecasting implementation
└── README.md
```
