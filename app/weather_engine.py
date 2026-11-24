
import requests
import pandas as pd
import numpy as np

from datetime import date, timedelta

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor
)


# ==========================================
# 1. FEATURE CONFIGURATION
# ==========================================

BASE_FEATURES = [
    "mean_temperature",
    "max_temperature",
    "min_temperature",
    "humidity",
    "rainfall",
    "wind_speed",
    "pressure",
    "cloud_cover",
    "mean_temperature_lag_1",
    "mean_temperature_lag_2",
    "mean_temperature_lag_3",
    "humidity_lag_1",
    "rainfall_lag_1",
    "wind_speed_lag_1",
    "temperature_rolling_3",
    "temperature_rolling_7",
    "humidity_rolling_3"
]

FEATURE_COLUMNS = BASE_FEATURES + [
    "horizon",
    "forecast_month",
    "forecast_day_of_year",
    "forecast_day_of_week"
]

TARGETS = [
    "mean_temperature",
    "humidity",
    "rainfall",
    "wind_speed"
]


# ==========================================
# 2. FETCH HISTORICAL WEATHER DATA
# ==========================================

def get_historical_weather(latitude, longitude):

    end_date = date.today() - timedelta(days=6)
    start_date = end_date - timedelta(days=5 * 365)

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max",
            "pressure_msl_mean",
            "cloud_cover_mean"
        ],
        "timezone": "auto"
    }

    response = requests.get(
        url,
        params=params,
        timeout=60
    )

    response.raise_for_status()

    daily = response.json()["daily"]

    df = pd.DataFrame({
        "date": daily["time"],
        "max_temperature": daily["temperature_2m_max"],
        "min_temperature": daily["temperature_2m_min"],
        "mean_temperature": daily["temperature_2m_mean"],
        "humidity": daily["relative_humidity_2m_mean"],
        "rainfall": daily["precipitation_sum"],
        "wind_speed": daily["wind_speed_10m_max"],
        "pressure": daily["pressure_msl_mean"],
        "cloud_cover": daily["cloud_cover_mean"]
    })

    return df


# ==========================================
# 2B. FETCH OFFICIAL 7-DAY WEATHER FORECAST
# ==========================================

def get_official_weather_forecast(latitude, longitude):
    """Fetch Open-Meteo's separate seven-day forecast for a location."""

    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "daily": [
            "temperature_2m_max",
            "temperature_2m_min",
            "temperature_2m_mean",
            "relative_humidity_2m_mean",
            "precipitation_sum",
            "wind_speed_10m_max"
        ],
        "forecast_days": 7,
        "timezone": "auto"
    }

    response = requests.get(
        url,
        params=params,
        timeout=60
    )
    response.raise_for_status()

    daily = response.json()["daily"]

    forecast_df = pd.DataFrame({
        "date": daily["time"],
        "max_temperature": daily["temperature_2m_max"],
        "min_temperature": daily["temperature_2m_min"],
        "mean_temperature": daily["temperature_2m_mean"],
        "humidity": daily["relative_humidity_2m_mean"],
        "rainfall": daily["precipitation_sum"],
        "wind_speed": daily["wind_speed_10m_max"]
    })

    forecast_df["date"] = pd.to_datetime(forecast_df["date"])

    return forecast_df


# ==========================================
# 3. DATA PREPROCESSING
# ==========================================

def prepare_weather_data(df):

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values("date")

    df = df.drop_duplicates(
        subset=["date"]
    )

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns

    df[numeric_columns] = (
        df[numeric_columns]
        .interpolate(limit_direction="both")
    )

    df[numeric_columns] = df[numeric_columns].fillna(
        df[numeric_columns].median()
    )

    # Calendar features
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear

    # Lag features
    df["mean_temperature_lag_1"] = (
        df["mean_temperature"].shift(1)
    )

    df["mean_temperature_lag_2"] = (
        df["mean_temperature"].shift(2)
    )

    df["mean_temperature_lag_3"] = (
        df["mean_temperature"].shift(3)
    )

    df["humidity_lag_1"] = (
        df["humidity"].shift(1)
    )

    df["rainfall_lag_1"] = (
        df["rainfall"].shift(1)
    )

    df["wind_speed_lag_1"] = (
        df["wind_speed"].shift(1)
    )

    # Rolling features
    df["temperature_rolling_3"] = (
        df["mean_temperature"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    df["temperature_rolling_7"] = (
        df["mean_temperature"]
        .shift(1)
        .rolling(7)
        .mean()
    )

    df["humidity_rolling_3"] = (
        df["humidity"]
        .shift(1)
        .rolling(3)
        .mean()
    )

    df = df.dropna().reset_index(drop=True)

    return df


# ==========================================
# 4. CREATE SUPERVISED TRAINING DATA
# ==========================================

def create_training_data(df):

    records = []

    for i in range(len(df) - 7):

        current = df.iloc[i]

        for horizon in range(1, 8):

            future = df.iloc[i + horizon]

            row = {
                feature: current[feature]
                for feature in BASE_FEATURES
            }

            row["horizon"] = horizon

            row["forecast_month"] = future["month"]

            row["forecast_day_of_year"] = (
                future["day_of_year"]
            )

            row["forecast_day_of_week"] = (
                future["day_of_week"]
            )

            for target in TARGETS:

                row[target + "_target"] = future[target]

            records.append(row)

    return pd.DataFrame(records)


# ==========================================
# 5. TRAIN LOCATION-SPECIFIC ML MODELS
# ==========================================

def train_location_models(training_data):

    models = {}

    for target in TARGETS:

        # Extra Trees for temperature
        if target == "mean_temperature":

            model = ExtraTreesRegressor(
                n_estimators=300,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

        # Random Forest for other variables
        else:

            model = RandomForestRegressor(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

        model.fit(
            training_data[FEATURE_COLUMNS],
            training_data[target + "_target"]
        )

        models[target] = model

    return models


# ==========================================
# 6. GENERATE 7-DAY WEATHER PREDICTIONS
# ==========================================

def predict_location_weather(df, models):

    latest = df.iloc[-1]

    temperature_history = (
        df["mean_temperature"].tail(7).tolist()
    )

    humidity_history = (
        df["humidity"].tail(3).tolist()
    )

    state = latest.copy()

    predictions = []

    for horizon in range(1, 8):

        target_date = latest["date"] + pd.Timedelta(
            days=horizon
        )

        features = {
            feature: state[feature]
            for feature in BASE_FEATURES
        }

        # Update lag features
        features["mean_temperature_lag_1"] = (
            temperature_history[-1]
        )

        features["mean_temperature_lag_2"] = (
            temperature_history[-2]
        )

        features["mean_temperature_lag_3"] = (
            temperature_history[-3]
        )

        features["humidity_lag_1"] = (
            humidity_history[-1]
        )

        features["rainfall_lag_1"] = state["rainfall"]

        features["wind_speed_lag_1"] = state["wind_speed"]

        # Update rolling features
        features["temperature_rolling_3"] = np.mean(
            temperature_history[-3:]
        )

        features["temperature_rolling_7"] = np.mean(
            temperature_history[-7:]
        )

        features["humidity_rolling_3"] = np.mean(
            humidity_history[-3:]
        )

        # Calendar features
        features["horizon"] = horizon

        features["forecast_month"] = target_date.month

        features["forecast_day_of_year"] = (
            target_date.dayofyear
        )

        features["forecast_day_of_week"] = (
            target_date.dayofweek
        )

        input_df = pd.DataFrame(
            [features],
            columns=FEATURE_COLUMNS
        )

        result = {
            "date": target_date,
            "horizon": horizon
        }

        # Predict all weather variables
        for target in TARGETS:

            value = float(
                models[target].predict(input_df)[0]
            )

            if target == "humidity":

                value = np.clip(value, 0, 100)

            elif target in ["rainfall", "wind_speed"]:

                value = max(0, value)

            elif target == "mean_temperature":

                value = np.clip(value, -80, 60)

            result[target] = value

        predictions.append(result)

        # Update recursive prediction history
        temperature_history.append(
            result["mean_temperature"]
        )

        humidity_history.append(
            result["humidity"]
        )

        state["mean_temperature"] = (
            result["mean_temperature"]
        )

        state["humidity"] = result["humidity"]

        state["rainfall"] = result["rainfall"]

        state["wind_speed"] = result["wind_speed"]

    return pd.DataFrame(predictions)


# ==========================================
# 7. COMPLETE LOCATION FORECAST PIPELINE
# ==========================================

def build_location_forecast(latitude, longitude):

    historical_df = get_historical_weather(
        latitude,
        longitude
    )

    prepared_df = prepare_weather_data(
        historical_df
    )

    if len(prepared_df) < 365:

        raise ValueError(
            "Not enough historical weather data to train the model."
        )

    training_data = create_training_data(
        prepared_df
    )

    models = train_location_models(
        training_data
    )

    forecast_df = predict_location_weather(
        prepared_df,
        models
    )

    return forecast_df
