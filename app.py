import streamlit as st
import pandas as pd
import joblib

st.title("💰 AI Budget Planner")

# 👉 грузим модель
model = joblib.load("budget_forecast_model.pkl")

uploaded_file = st.file_uploader("Загрузите CSV", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)
    df['date'] = pd.to_datetime(df['date'])

    st.write("### Данные")
    st.write(df.head())

    category_df = df.groupby(['date','category'])['amount_kzt'].sum().reset_index()
    category_df = category_df.sort_values(['category','date'])

    category_df['month'] = category_df['date'].dt.month
    category_df['year'] = category_df['date'].dt.year

    category_df['lag_1'] = category_df.groupby('category')['amount_kzt'].shift(1)
    category_df['lag_2'] = category_df.groupby('category')['amount_kzt'].shift(2)
    category_df['lag_3'] = category_df.groupby('category')['amount_kzt'].shift(3)

    category_df = category_df.dropna()

    # 👉 важно: фичи должны совпадать с обучением
    X = category_df[['month','year','lag_1','lag_2','lag_3']]

    # прогноз
    last_data = category_df.groupby('category').last().reset_index()
    last_data['month'] = last_data['month'] + 1

    future_X = last_data[['month','year','lag_1','lag_2','lag_3']]
    preds = model.predict(future_X)

    last_data['prediction'] = preds

    st.write("### Прогноз")
    st.write(last_data[['category','prediction']])

    st.line_chart(df.groupby('date')['amount_kzt'].sum())
