
import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

from weather_engine import build_location_forecast


# ---------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------

st.set_page_config(
    page_title="Weather Prediction by Area",
    page_icon="🌦️",
    layout="wide"
)

st.title("🌦️ Weather Prediction by Area")

st.write(
    "Explore weather forecasts for cities worldwide "
    "using historical weather data and machine learning."
)


# ---------------------------------------
# 2. LOCATION SEARCH
# ---------------------------------------

st.sidebar.header("Location Search")

city = st.sidebar.text_input(
    "Enter city name",
    value="Hyderabad"
)


@st.cache_data(ttl=3600)
def get_coordinates(city_name):

    url = "https://geocoding-api.open-meteo.com/v1/search"

    params = {
        "name": city_name,
        "count": 5,
        "language": "en",
        "format": "json"
    }

    response = requests.get(
        url,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    if not data.get("results"):
        return None

    return data["results"][0]


# ---------------------------------------
# 3. PREDICT WEATHER BUTTON
# ---------------------------------------

if st.sidebar.button("Predict Weather"):

    if not city.strip():

        st.error("Please enter a city name.")

    else:

        try:

            # Find location

            with st.spinner("Finding your location..."):

                location = get_coordinates(city)

                if location is None:

                    st.error(
                        "City not found. Please try another city."
                    )

                    st.stop()

            latitude = location["latitude"]
            longitude = location["longitude"]

            st.subheader(
                f"📍 {location['name']}, "
                f"{location.get('country', '')}"
            )

            st.caption(
                f"Coordinates: {latitude:.4f}, {longitude:.4f}"
            )


            # ---------------------------------------
            # 4. LOCATION-SPECIFIC ML MODEL
            # ---------------------------------------

            with st.spinner(
                "Downloading historical weather and training "
                "location-specific ML models. This may take a few minutes..."
            ):

                ml_forecast = build_location_forecast(
                    latitude,
                    longitude
                )

            st.success(
                "Location-specific ML forecast generated!"
            )


            # ---------------------------------------
            # 5. SEVEN-DAY FORECAST TABLE
            # ---------------------------------------

            st.subheader(
                "🤖 Experimental ML Forecast — 7 Days"
            )

            display_df = ml_forecast.copy()

            display_df["date"] = pd.to_datetime(
                display_df["date"]
            ).dt.strftime("%Y-%m-%d")

            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True
            )


            # ---------------------------------------
            # 6. DOWNLOAD FORECAST AS CSV
            # ---------------------------------------

            csv_data = ml_forecast.to_csv(
                index=False
            )

            st.download_button(
                label="📥 Download 7-Day Forecast",
                data=csv_data,
                file_name=f"{city}_weather_forecast.csv",
                mime="text/csv"
            )


            # ---------------------------------------
            # 7. TEMPERATURE PREDICTION CHART
            # ---------------------------------------

            st.subheader("🌡️ Temperature Prediction")

            fig, ax = plt.subplots(figsize=(10, 4))

            ax.plot(
                pd.to_datetime(ml_forecast["date"]),
                ml_forecast["mean_temperature"],
                marker="o"
            )

            ax.set_xlabel("Date")
            ax.set_ylabel("Temperature (°C)")
            ax.set_title("Predicted Mean Temperature")

            ax.grid(True)

            st.pyplot(fig)

            plt.close(fig)


            # ---------------------------------------
            # 8. RAINFALL PREDICTION CHART
            # ---------------------------------------

            st.subheader("💧 Rainfall Prediction")

            fig, ax = plt.subplots(figsize=(10, 4))

            ax.bar(
                pd.to_datetime(ml_forecast["date"]),
                ml_forecast["rainfall"]
            )

            ax.set_xlabel("Date")
            ax.set_ylabel("Rainfall (mm)")
            ax.set_title("Predicted Daily Rainfall")

            ax.grid(axis="y")

            st.pyplot(fig)

            plt.close(fig)


            # ---------------------------------------
            # 9. WEATHER SUMMARY
            # ---------------------------------------

            st.subheader("📊 Weather Summary")

            col1, col2, col3, col4 = st.columns(4)

            col1.metric(
                "Average Temperature",
                f"{ml_forecast['mean_temperature'].mean():.1f} °C"
            )

            col2.metric(
                "Average Humidity",
                f"{ml_forecast['humidity'].mean():.1f}%"
            )

            col3.metric(
                "Total Predicted Rainfall",
                f"{ml_forecast['rainfall'].sum():.1f} mm"
            )

            col4.metric(
                "Maximum Wind Speed",
                f"{ml_forecast['wind_speed'].max():.1f} km/h"
            )


            # ---------------------------------------
            # 10. PROJECT INFORMATION
            # ---------------------------------------

            st.info(
                "Research project note: These are experimental "
                "machine-learning predictions trained on historical "
                "weather data for the selected location. They are not "
                "official meteorological forecasts."
            )


        except Exception as error:

            st.error(
                f"Unable to generate the forecast: {error}"
            )

            st.exception(error)


# ---------------------------------------
# 11. SIDEBAR FOOTER
# ---------------------------------------

st.sidebar.markdown("---")

st.sidebar.caption(
    "Weather Data: Open-Meteo | ML Algorithm: Random Forest"
)
