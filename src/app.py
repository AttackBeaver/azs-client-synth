import streamlit as st
import pandas as pd
import json
import os
from generator import generate_clients
from mapper import map_clients_to_portraits

st.set_page_config(
    page_title="АЗС TwinLab",
    #layout="wide",
    #initial_sidebar_state="expanded",
    page_icon="⛽"
)
st.markdown(
    "<h1 style='text-align: center; color: #ffffff;'>⛽АЗС TwinLab⛽</h1>", 
    unsafe_allow_html=True
)
#st.title("⛽АЗС TwinLab⛽", )
st.markdown("""
**Проект подготовлен в рамках хакатона «Моя профессия – IT 2025».**  
**Команда-разработчик:** *«404: Имя не найдено»*  

Приложение демонстрирует, как на основе данных о клиентах АЗС можно:
- сегментировать пользователей,
- построить клиентские портреты,
- анализировать поведение по категориям,
- поддерживать продуктовые решения (например, выбор целевой аудитории, прогноз отклика и проведение A/B-тестов).
---
""")

st.sidebar.header("⚙️Настройки генерации")
num_clients = st.sidebar.slider("Количество клиентов", min_value=20, max_value=2000, value=1000, step=10)
with st.sidebar:
    st.markdown("""
    ---
    """)
    st.image("docs/f404.png", width="content")
    
DATA_PATH = "data/synthetic.csv"
MAPPED_PATH = "data/synthetic_mapped.csv"

st.subheader("Источник данных")

uploaded = st.file_uploader("Загрузите CSV с обезличенными пользователями (или перетащите файл)", type=["csv"])

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
        st.success(f"Данные сгенерированы и сохранены в {DATA_PATH} ({len(df)} строк)")

# Если загрузил файл через drag & drop
if uploaded is not None:
    try:
        df_uploaded = pd.read_csv(uploaded)
        st.session_state["clients_df"] = df_uploaded
        st.success(f"Загружен файл: {uploaded.name} ({len(df_uploaded)} строк)")
    except Exception as e:
        st.error(f"Не удалось прочитать загруженный файл: {e}")

st.subheader("Превью данных")
if "clients_df" in st.session_state:
    df = st.session_state["clients_df"]
    st.dataframe(df.head(10))
else:
    st.info("Нет данных. Загрузите CSV или сгенерируйте новый набор.")

# === Маппинг портретов ===
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
                mapped_df = map_clients_to_portraits(st.session_state["clients_df"], portraits)
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

# === Дополнительные возможности ===
st.markdown("---")
st.subheader("В разработке")
with st.expander("Просмотр портретов и аналитики"):
    st.write("📌 В будущем здесь будет визуализация профилей портретов, тепловые карты и метрики сегментов.")
with st.expander("A/B тестирование"):
    st.write("📊 Планируется модуль симуляции реакции целевых групп на фичи и сервисы.")
with st.expander("Прогнозирование поведения"):
    st.write("🔮 Будет добавлен блок машинного обучения для прогноза визитов и объёмов покупок.")

# === Подвал ===
st.markdown("""
---
© 2025. Проект команды **«404: Имя не найдено»**  
Хакатон *«Моя профессия — IT 2025»*
""")
