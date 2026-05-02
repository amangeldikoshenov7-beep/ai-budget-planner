import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="AI Budget Planner", page_icon="💰", layout="wide")

st.title("💰 AI Budget Planner")
st.write("Прогнозирование расходов компании по категориям")

model = joblib.load("budget_forecast_model.pkl")

uploaded_file = st.file_uploader("Загрузите CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df["date"] = pd.to_datetime(df["date"])

    st.subheader("Данные")
    st.dataframe(df.head())

    category_df = (
        df.groupby(["date", "category"])["amount_kzt"]
        .sum()
        .reset_index()
        .sort_values(["category", "date"])
    )

    category_df["year"] = category_df["date"].dt.year
    category_df["month"] = category_df["date"].dt.month

    category_df["month_sin"] = np.sin(2 * np.pi * category_df["month"] / 12)
    category_df["month_cos"] = np.cos(2 * np.pi * category_df["month"] / 12)

    category_df["lag_1"] = category_df.groupby("category")["amount_kzt"].shift(1)
    category_df["lag_2"] = category_df.groupby("category")["amount_kzt"].shift(2)
    category_df["lag_3"] = category_df.groupby("category")["amount_kzt"].shift(3)

    category_df["rolling_mean_3"] = (
        category_df.groupby("category")["amount_kzt"]
        .transform(lambda x: x.rolling(3).mean())
    )

    category_df = category_df.dropna()

    st.subheader("Динамика общих расходов")
    monthly_total = df.groupby("date")["amount_kzt"].sum()
    st.line_chart(monthly_total)

    features = [
        "category",
        "year",
        "month",
        "month_sin",
        "month_cos",
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_3"
    ]

    last_data = category_df.groupby("category").last().reset_index()

    last_date = category_df["date"].max()
    next_date = last_date + pd.DateOffset(months=1)

    last_data["date"] = next_date
    last_data["year"] = next_date.year
    last_data["month"] = next_date.month
    last_data["month_sin"] = np.sin(2 * np.pi * last_data["month"] / 12)
    last_data["month_cos"] = np.cos(2 * np.pi * last_data["month"] / 12)

    future_X = last_data[features]
    preds = model.predict(future_X)

    last_data["predicted_amount_kzt"] = preds

    forecast = last_data[["category", "predicted_amount_kzt"]].sort_values(
        "predicted_amount_kzt", ascending=False
    )

    st.subheader(f"Прогноз расходов по категориям на {next_date.strftime('%Y-%m')}")
    st.dataframe(forecast)

    st.subheader("График прогноза")
    st.bar_chart(forecast.set_index("category")["predicted_amount_kzt"])

    top_category = forecast.iloc[0]["category"]
    top_amount = forecast.iloc[0]["predicted_amount_kzt"]

    st.subheader("AI-анализ")
    st.write(
        f"Наибольший прогнозируемый расход ожидается в категории **{top_category}** "
        f"— примерно **{top_amount:,.0f} ₸**. "
        "Рекомендуется дополнительно проверить эту категорию при планировании бюджета."
    )

else:
    st.info("Загрузите CSV-файл, чтобы получить прогноз.")
