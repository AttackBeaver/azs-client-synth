import streamlit as st
import pandas as pd
from generator import generate_clients
from mapper import load_portraits, map_clients_to_portraits
import os
import plotly.express as px

st.set_page_config(page_title="AZS TwinLab", layout="wide")
st.title("AZS TwinLab — синтез и маппинг портретов")

st.sidebar.header("Синтетика")
if st.sidebar.button("Сгенерировать synthetic.csv (1000)"):
    df = generate_clients(1000)
    df.to_csv("data/synthetic.csv", index=False)
    st.sidebar.success("data/synthetic.csv создан")

uploaded = st.file_uploader("Загрузите CSV с обезличенными пользователями", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    if os.path.exists("data/synthetic.csv"):
        df = pd.read_csv("data/synthetic.csv")
        st.info("Загружен synthetic.csv из проекта (авто)")
    else:
        st.warning("Нет данных. Нажмите кнопку генерации или загрузите CSV.")
        st.stop()

st.subheader("Превью данных")
st.dataframe(df.head())

portraits = load_portraits("portraits.json")
if st.button("Сопоставить с портретами"):
    mapped = map_clients_to_portraits(df, portraits)
    # attach
    df2 = df.copy()
    df2["assigned_portrait"] = mapped["portrait_name"].values
    st.subheader("Распределение по портретам")
    counts = df2["assigned_portrait"].value_counts().reset_index()
    counts.columns = ["portrait", "count"]
    fig = px.bar(counts, x="portrait", y="count", title="Распределение клиентов по портретам")
    st.plotly_chart(fig)
    st.subheader("Таблица с назначениями (первые 200)")
    st.dataframe(df2.head(200))
    csv = df2.to_csv(index=False).encode('utf-8')
    st.download_button("Скачать mapped CSV", csv, "mapped.csv", "text/csv")
