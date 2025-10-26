import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
from generator import generate_clients
from mapper import map_clients_to_portraits
from visualization import plot_portrait_distribution, plot_heatmap_features, plot_metric
from simulator_advanced import simulate_feature_response
import predictor

st.set_page_config(
    page_title="АЗС TwinLab",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="⛽"
)

# === заголовок ===
st.title("⛽АЗС TwinLab⛽")
st.subheader("Команда-разработчик: «404: Имя не найдено»")
with st.expander("Проект подготовлен в рамках хакатона «Моя профессия – IT 2025»", expanded=False):
    st.markdown("""
Приложение демонстрирует, как на основе данных о клиентах АЗС можно:
- сегментировать пользователей,
- построить клиентские портреты,
- анализировать поведение по категориям,
- поддерживать продуктовые решения (например, выбор целевой аудитории, прогноз отклика и проведение A/B-тестов).
""")

# === бар слева ===
st.sidebar.image("docs/f404.png", width="content")

st.sidebar.header("⚙️Настройки генерации")
num_clients = st.sidebar.slider(
    "Количество клиентов", min_value=1000, max_value=10000, value=5000, step=100)
with st.sidebar:
    st.markdown(""" --- """)

clients_df = st.session_state.get("clients_df", None)
portraits_rules = json.load(
    open("src/behavior_rules.json", "r", encoding="utf-8"))
feature_hypotheses = json.load(
    open("src/feature_hypotheses.json", "r", encoding="utf-8"))

st.sidebar.header("⚙️Настройки симуляции")
selected_feature = st.sidebar.selectbox(
    "Выберите фичу для симуляции",
    [f["feature_name"] for f in feature_hypotheses]
)

with st.sidebar:
    st.markdown(""" --- """)

st.sidebar.header("⚙️Настройка прогноза")
feature_choice = st.sidebar.selectbox(
    "Выберите фичу для моделирования:",
    [f["feature_name"] for f in feature_hypotheses]
)
train_model = st.sidebar.checkbox(
    "Обучить модель на симуляциях (если есть данные)", value=True)


DATA_PATH = "data/synthetic.csv"
MAPPED_PATH = "data/synthetic_mapped.csv"

# === выбираем откуда брать данные ===
st.subheader("Источник данных")

uploaded = st.file_uploader(
    "Загрузите CSV с обезличенными пользователями (или перетащите файл)", type=["csv"])

col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Загрузить data/synthetic.csv (если есть)"):
        if os.path.exists(DATA_PATH):
            try:
                df = pd.read_csv(DATA_PATH)
                st.session_state["clients_df"] = df
                st.success(f"Загружен {DATA_PATH} ({len(df)} строк)")
            except Exception as e:
                st.error(f"Ошибка чтения {DATA_PATH}: {e}")
        else:
            st.warning(f"{DATA_PATH} не найден")

with col2:
    if st.button("Сгенерировать данные"):
        df = generate_clients(num_clients)
        # сохраняем автоматически в data/
        os.makedirs("data", exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
        st.session_state["clients_df"] = df
        st.success(
            f"Данные сгенерированы и сохранены в {DATA_PATH} ({len(df)} строк)")

# если загрузил файл через drag&drop
if uploaded is not None:
    try:
        df_uploaded = pd.read_csv(uploaded)
        st.session_state["clients_df"] = df_uploaded
        st.success(
            f"Загружен файл: {uploaded.name} ({len(df_uploaded)} строк)")
    except Exception as e:
        st.error(f"Не удалось прочитать загруженный файл: {e}")

st.subheader("Превью данных")
if "clients_df" in st.session_state:
    df = st.session_state["clients_df"]
    st.dataframe(df.head(10))
else:
    st.info("Нет данных. Загрузите CSV или сгенерируйте новый набор.")

# === описание портретов клиентов ===
st.subheader("Сводка по портретам")
with st.expander("Наши клиенты", expanded=False):
    with open("src/portraits.json", "r", encoding="utf-8") as f:
        portraits_info = json.load(f)

    for portrait in portraits_info:
        with st.expander(f"📌 {portrait['portrait_name']}", expanded=False):
            st.write(f"**Описание:** {portrait['description']}")
            st.write(f"**Бизнес-ценность:** {portrait['business_value']}")
            st.write("**Рекомендации по взаимодействию:**")
            for rec in portrait['recommendations']:
                st.write(f"• {rec}")

# === маппинг ===
st.subheader("Маппинг клиентов на портреты")
if "clients_df" in st.session_state:
    if st.button("Сопоставить с портретами"):
        with st.spinner("Анализ и кластеризация клиентов..."):
            try:
                with open("src/portraits.json", "r", encoding="utf-8") as f:
                    portraits = json.load(f)
            except Exception as e:
                st.error(f"Ошибка загрузки portraits.json: {e}")
                st.stop()

            try:
                mapped_df = map_clients_to_portraits(
                    st.session_state["clients_df"], portraits)
            except Exception as e:
                st.error(f"Ошибка маппинга: {e}")
                st.stop()

            st.session_state["mapped_df"] = mapped_df
            # сохраняем результат
            os.makedirs("data", exist_ok=True)
            mapped_df.to_csv(MAPPED_PATH, index=False)

        st.success("✅ Маппинг завершен!")
        st.dataframe(mapped_df.head(10))

        st.markdown("### Распределение по портретам")
        counts = mapped_df["portrait_name"].value_counts()
        st.bar_chart(counts)
        st.markdown(f"Результат также сохранён в `{MAPPED_PATH}`")
else:
    st.info("Сначала загрузите или сгенерируйте данные.")

# === визуализация портретов ===
feature_names = {
    "visits_per_month": "Визиты в месяц",
    "avg_liters_per_visit": "Средний литраж",
    "avg_spend_per_visit": "Средний чек"
}

if "mapped_df" in st.session_state:
    df_mapped = st.session_state["mapped_df"]

    st.subheader("Визуализация портретов")
    st.plotly_chart(plot_portrait_distribution(df_mapped))

    st.plotly_chart(plot_heatmap_features(
        df_mapped, list(feature_names.keys()), feature_names))

    st.subheader("Метрики по портретам")
    for metric, name in feature_names.items():
        st.plotly_chart(plot_metric(df_mapped, metric, name))

# === симуляцмя реакции портретов ===
st.subheader("Симуляция реакции клиентов")
if clients_df is not None and st.button("Запустить симуляцию"):
    with st.spinner("Симуляция отклика клиентов..."):
        sim_df, metrics_df = simulate_feature_response(
            clients_df, portraits_rules, feature_hypotheses, selected_feature)
        st.session_state["sim_df"] = sim_df

    st.success("✅ Симуляция завершена!")

    st.subheader("Метрики отклика по портретам")
    st.dataframe(metrics_df)

    st.subheader("Распределение откликов по портретам")
    st.bar_chart(metrics_df.set_index("portrait_name")["response_rate"])

    st.subheader("Первые 200 клиентов с реакцией")
    st.dataframe(sim_df.head(200))
else:
    st.info("Сначала выполните маппинг клиентов.")

# === прогноз поведения ===
st.subheader("Прогнозирование поведения клиентов")

# загрузка данных

try:
    mapped_df = pd.read_csv("data/synthetic_mapped.csv")
    st.success("✅ Загрузка клиентов после маппинга: synthetic_mapped.csv")

    sim_df_path = "data/simulated_reactions_advanced.csv"
    sim_df = pd.read_csv(sim_df_path) if os.path.exists(sim_df_path) else None
    if sim_df is not None:
        st.success(
            "✅ Загружены данные симуляции: simulated_reactions_advanced.csv")

    with open("src/behavior_rules.json", "r", encoding="utf-8") as f:
        portraits_rules = json.load(f)

    with open("src/feature_hypotheses.json", "r", encoding="utf-8") as f:
        feature_hypotheses = json.load(f)

except Exception as e:
    st.error(f"Ошибка при загрузке данных: {e}")
    st.stop()

# запуск прогноза
if st.button("Запустить прогноз"):
    with st.spinner("Модуль прогнозирования выполняется..."):
        try:
            clients_forecast, portraits_forecast = predictor.run_behavior_forecast(
                mapped_df=mapped_df,
                sim_df=sim_df,
                portraits_rules=portraits_rules,
                feature_hypotheses=feature_hypotheses,
                feature_name=feature_choice,
                train_model=train_model,
                save_to="data"
            )

            st.session_state["forecast_clients"] = clients_forecast
            st.session_state["forecast_portraits"] = portraits_forecast

            st.success("✅ Прогноз успешно выполнен!")
        except Exception as e:
            st.error(f"Ошибка при выполнении прогноза: {e}")
            st.stop()

# резы и сохранение
if "forecast_clients" in st.session_state and "forecast_portraits" in st.session_state:
    clients_forecast = st.session_state["forecast_clients"]
    portraits_forecast = st.session_state["forecast_portraits"]

    st.markdown("### Прогноз по клиентам (первые строки)")
    st.dataframe(clients_forecast.head())

    st.markdown("### Сводный прогноз по портретам")
    st.dataframe(portraits_forecast)

    st.markdown("#### Изменение выручки по портретам")
    st.bar_chart(
        portraits_forecast.set_index("portrait_name")[
            ["baseline_revenue", "predicted_revenue"]]
    )

    st.markdown("#### Изменение количества визитов")
    st.bar_chart(
        portraits_forecast.set_index("portrait_name")[
            ["baseline_visits", "predicted_visits"]]
    )

    st.info("Результаты сохранены в папке `data/`")
else:
    st.info("Чтобы увидеть результаты, выполните прогнозирование.")

# текстовое описание
if "forecast_clients" in st.session_state and "forecast_portraits" in st.session_state:
    summary_text = predictor.generate_forecast_summary(
        st.session_state["forecast_clients"],
        st.session_state["forecast_portraits"],
        feature_name=feature_choice
    )
    st.markdown(summary_text)

# === подвал ===
st.markdown("""
---
© 2025. Проект команды **«404: Имя не найдено»**  
Хакатон *«Моя профессия — IT 2025»*
""")
