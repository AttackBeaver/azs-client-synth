import os
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# === инструменты для фич ===


def infer_feature_info(feature_hypotheses, feature_name):
    """
    Найти запись гипотезы по имени фичи. Вернуть dict с ключами:
    'feature_name', 'description', 'target_metric', 'applicable_to', 'expected_lift_visits', 'expected_lift_spend'
    Если expected lifts отсутствуют, будут отсутствовать ключи — их генерирует compute_default_lifts().
    """
    for f in feature_hypotheses:
        if f.get("feature_name") == feature_name:
            return f
    raise KeyError(
        f"Feature '{feature_name}' не найдена в feature_hypotheses.")


def compute_default_lifts(feature_info):
    """
    Если у гипотезы нет численных expected_lift, выдаём разумные дефолты по типу фичи.
    Возвращает dict: {'lift_visits': float, 'lift_spend': float}
    */
    """
    name = feature_info.get("feature_name", "").lower()
    # базовые правила — эмпирические
    if "скидк" in name or "cashback" in name or "кэшбэк" in name:
        return {"lift_visits": 0.08, "lift_spend": 0.05}
    if "рекоменд" in name or "recommend" in name:
        return {"lift_visits": 0.02, "lift_spend": 0.12}
    if "уведомл" in name or "notification" in name:
        return {"lift_visits": 0.05, "lift_spend": 0.03}
    if "пакет" in name or "корпоратив" in name:
        # меньше визитов, больше объёма
        return {"lift_visits": -0.05, "lift_spend": 0.12}
    if "приоритет" in name or "скорость" in name:
        return {"lift_visits": 0.06, "lift_spend": 0.04}
    if "эколог" in name or "eco" in name:
        return {"lift_visits": 0.02, "lift_spend": 0.03}
    # общий дефолт
    return {"lift_visits": 0.05, "lift_spend": 0.05}


def estimate_response_prob(row, portraits_rules, target_metric):
    """
    Оценка вероятности отклика клиента, если нет реальных откликов.
    Берём правило для портрета и масштабируем по активности клиента.
    """
    portrait = row.get("portrait_name")
    rules = portraits_rules.get(portrait, {})
    base = rules.get(target_metric, None)
    if base is None:
        # минимальная вероятность по умолчанию
        base = 0.03
    # усиление вероятности для активных клиентов
    visits_factor = min(row.get("visits_per_month", 0) / 12.0, 1.0)
    spend_factor = min(row.get("avg_spend_per_visit", 0) / 10000.0, 1.0)
    prob = base + 0.4 * visits_factor + 0.2 * spend_factor
    prob = min(prob, 0.99)
    return prob

# === основная функция ===


def run_behavior_forecast(
    mapped_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    portraits_rules: dict,
    feature_hypotheses: list,
    feature_name: str,
    train_model: bool = True,
    save_to: str = None
):
    """
    Построить прогноз изменения визитов и среднего чека для каждого клиента при запуске feature_name.

    Входы:
    - mapped_df: DataFrame с колонками client_id, portrait_name, visits_per_month, avg_spend_per_visit, ...
    - sim_df: DataFrame с симуляциями (может быть None). Ожидается колонка f"response_to_{feature_name}" если симуляция проводилась.
    - portraits_rules: dict из behavior_rules.json
    - feature_hypotheses: список гипотез (feature_hypotheses.json)
    - feature_name: имя фичи для симуляции
    - train_model: если True и в sim_df есть наблюдаемые цели — обучаем регрессоры
    - save_to: путь (папка) для сохранения результатов CSV (опционально)

    Возвращает tuple (client_forecast_df, portrait_agg_df)
    """
    # копируем исходный DF
    df = mapped_df.copy().reset_index(drop=True)
    if "client_id" not in df.columns:
        df["client_id"] = df.index.astype(str)

    # получаем инфо по фиче
    feature_info = infer_feature_info(feature_hypotheses, feature_name)
    target_metric = feature_info.get("target_metric")
    applicable = feature_info.get("applicable_to", [])

    # определяем lifts
    lifts = {}
    lifts["lift_visits"] = feature_info.get("expected_lift_visits")
    lifts["lift_spend"] = feature_info.get("expected_lift_spend")
    # если нет — вычисляем дефолтные
    defaults = compute_default_lifts(feature_info)
    if lifts["lift_visits"] is None:
        lifts["lift_visits"] = defaults["lift_visits"]
    if lifts["lift_spend"] is None:
        lifts["lift_spend"] = defaults["lift_spend"]

    # попытка извлечь отклик из sim_df
    response_col = f"response_to_{feature_name}"
    has_response_column = sim_df is not None and response_col in sim_df.columns

    # сливаем sim_df (если есть) по client_id
    if sim_df is not None and "client_id" in sim_df.columns:
        df = df.merge(sim_df[["client_id", response_col]] if has_response_column else sim_df[[
                      "client_id"]], on="client_id", how="left")

    # если нет реального отклика, оцениваем вероятность отклика
    if not has_response_column:
        df[f"{response_col}_prob"] = df.apply(
            lambda r: estimate_response_prob(r, portraits_rules, target_metric), axis=1)
        # имитация бинарного отклика при необходимости (например для обучения) — не делаем без данных
    else:
        # есть бинарный отклик 0/1 — можем вычислить empirical lift при отклике
        df[f"{response_col}_prob"] = df[response_col]  # 0/1

    # подготовка признаков для возможного обучения / предсказания
    feat_num = ["visits_per_month",
                "avg_liters_per_visit", "avg_spend_per_visit"]
    for c in feat_num:
        if c not in df.columns:
            df[c] = 0.0

    # One-Hot кодирование портретов (для модели)
    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    portraits_arr = ohe.fit_transform(df[["portrait_name"]])
    portraits_cols = [f"portrait__{v}" for v in ohe.categories_[0]]
    portraits_df = pd.DataFrame(
        portraits_arr, columns=portraits_cols, index=df.index)

    # базовые признаки для модели
    X = pd.concat([df[feat_num].reset_index(drop=True),
                  portraits_df.reset_index(drop=True)], axis=1)

    # если в sim_df есть колонки post-метрик (например visits_post, spend_post), используем их для обучения
    y_visits = None
    y_spend = None
    if sim_df is not None:
        # возможные названия колонок — гибкость
        possible_visits_post = [
            f"visits_after_{feature_name}", f"visits_post_{feature_name}", "visits_after", "visits_post"]
        possible_spend_post = [
            f"spend_after_{feature_name}", f"spend_post_{feature_name}", "spend_after", "spend_post", "avg_spend_post"]
        for col in possible_visits_post:
            if col in sim_df.columns:
                df = df.merge(sim_df[["client_id", col]],
                              on="client_id", how="left")
                y_visits = df[col].values
                break
        for col in possible_spend_post:
            if col in sim_df.columns:
                df = df.merge(sim_df[["client_id", col]],
                              on="client_id", how="left")
                y_spend = df[col].values
                break

    # если есть наблюдаемые post-значения — обучаем регрессоры
    model_visits = None
    model_spend = None
    can_train = train_model and (y_visits is not None or y_spend is not None)

    if can_train:
        if y_visits is not None:
            # целевая переменная — относительное изменение (post / pre)
            baseline = df["visits_per_month"].values.astype(float) + 1e-6
            y_rel_visits = y_visits / baseline
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_rel_visits, test_size=0.2, random_state=42)
            model_visits = RandomForestRegressor(
                n_estimators=100, random_state=42)
            model_visits.fit(X_train, y_train)
            pred = model_visits.predict(X_val)
            print("Visits model MAE:", mean_absolute_error(
                y_val, pred), "R2:", r2_score(y_val, pred))
        if y_spend is not None:
            baseline_spend = df["avg_spend_per_visit"].values.astype(
                float) + 1e-6
            y_rel_spend = y_spend / baseline_spend
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_rel_spend, test_size=0.2, random_state=42)
            model_spend = RandomForestRegressor(
                n_estimators=100, random_state=42)
            model_spend.fit(X_train, y_train)
            pred = model_spend.predict(X_val)
            print("Spend model MAE:", mean_absolute_error(
                y_val, pred), "R2:", r2_score(y_val, pred))

    # предсказание эффекта: комбинируем вероятности отклика и lift-ы
    # если обученные модели есть — используем их для предсказания относительного изменения
    if model_visits is not None:
        rel_visits_pred = model_visits.predict(X)
    else:
        # baseline multiplier = 1 + prob * lift_visits
        rel_visits_pred = 1.0 + \
            df[f"{response_col}_prob"].values * lifts["lift_visits"]

    if model_spend is not None:
        rel_spend_pred = model_spend.predict(X)
    else:
        rel_spend_pred = 1.0 + \
            df[f"{response_col}_prob"].values * lifts["lift_spend"]

    # некоторое ограничение разумности предиктов
    rel_visits_pred = np.clip(rel_visits_pred, 0.5, 5.0)
    rel_spend_pred = np.clip(rel_spend_pred, 0.7, 5.0)

    # формируем прогнозы на клиента
    df["baseline_visits"] = df["visits_per_month"].astype(float)
    df["baseline_spend"] = df["avg_spend_per_visit"].astype(float)
    df["pred_rel_visits"] = rel_visits_pred
    df["pred_rel_spend"] = rel_spend_pred
    df["predicted_visits"] = df["baseline_visits"] * df["pred_rel_visits"]
    df["predicted_spend"] = df["baseline_spend"] * df["pred_rel_spend"]
    df["delta_visits"] = df["predicted_visits"] - df["baseline_visits"]
    df["delta_spend"] = df["predicted_spend"] - df["baseline_spend"]
    # выручка (в месяц) до/после
    df["baseline_revenue"] = df["baseline_visits"] * df["baseline_spend"]
    df["predicted_revenue"] = df["predicted_visits"] * df["predicted_spend"]
    df["revenue_change"] = df["predicted_revenue"] - df["baseline_revenue"]

    # агрегаты по портретам
    agg = df.groupby("portrait_name").agg(
        clients_count=("client_id", "count"),
        baseline_visits=("baseline_visits", "sum"),
        predicted_visits=("predicted_visits", "sum"),
        baseline_revenue=("baseline_revenue", "sum"),
        predicted_revenue=("predicted_revenue", "sum"),
    ).reset_index()
    agg["visits_change_abs"] = agg["predicted_visits"] - agg["baseline_visits"]
    agg["revenue_change_abs"] = agg["predicted_revenue"] - agg["baseline_revenue"]
    agg["visits_change_rel"] = agg["visits_change_abs"] / \
        (agg["baseline_visits"] + 1e-9)
    agg["revenue_change_rel"] = agg["revenue_change_abs"] / \
        (agg["baseline_revenue"] + 1e-9)

    # сохранение результатов
    if save_to:
        os.makedirs(save_to, exist_ok=True)
        client_path = os.path.join(
            save_to, f"forecast_clients_{feature_name}.csv")
        agg_path = os.path.join(
            save_to, f"forecast_portraits_{feature_name}.csv")
        df.to_csv(client_path, index=False)
        agg.to_csv(agg_path, index=False)

    # возвращаем детализированный прогноз и агрегат
    client_cols = [
        "client_id", "portrait_name", "baseline_visits", "predicted_visits", "delta_visits",
        "baseline_spend", "predicted_spend", "delta_spend", "baseline_revenue", "predicted_revenue", "revenue_change"
    ]
    return df[client_cols].copy(), agg.copy()


def generate_forecast_summary(clients_forecast, portraits_forecast, feature_name: str = ""):
    """
    Генерация красиво оформленного текстового отчёта
    по результатам прогнозирования для менеджеров.
    """

    try:
        total_visits_before = clients_forecast["baseline_visits"].sum()
        total_visits_after = clients_forecast["predicted_visits"].sum()
        total_spend_before = clients_forecast["baseline_revenue"].sum()
        total_spend_after = clients_forecast["predicted_revenue"].sum()
    except KeyError:
        raise KeyError(
            "Не найдены необходимые столбцы для анализа (baseline/predicted).")

    delta_visits = total_visits_after - total_visits_before
    delta_spend = total_spend_after - total_spend_before
    delta_spend_pct = (delta_spend / total_spend_before) * \
        100 if total_spend_before else 0

    # определяем общий эффект
    if delta_spend > 0:
        trend = "📈 Рост выручки"
        effect = "положительный"
        color = "🟢"
    elif delta_spend < 0:
        trend = "📉 Снижение выручки"
        effect = "негативный"
        color = "🔴"
    else:
        trend = "⚖️ Стабильный результат"
        effect = "нейтральный"
        color = "🟡"

    # формируем красиво оформленный отчёт
    summary = []
    summary.append(f"### Прогноз по фиче: **{feature_name}**\n")
    summary.append(f"**Общий тренд:** {trend} {color}")
    summary.append("")
    summary.append("**Основные показатели:**")
    summary.append(
        f"- Изменение выручки: **{delta_spend:,.0f} ₽ ({delta_spend_pct:+.1f}%)**".replace(",", " "))
    summary.append(
        f"- Изменение количества визитов: **{delta_visits:,.0f}**".replace(",", " "))
    summary.append("")
    summary.append(f"**Общий эффект:** {effect.capitalize()} для бизнеса.\n")

    # подсказка по ключевым сегментам
    try:
        top_segments = (
            portraits_forecast.sort_values(
                "revenue_change_rel", ascending=False)
            .head(2)["portrait_name"]
            .tolist()
        )
        bottom_segments = (
            portraits_forecast.sort_values(
                "revenue_change_rel", ascending=True)
            .head(1)["portrait_name"]
            .tolist()
        )
    except KeyError:
        top_segments, bottom_segments = [], []

    if top_segments:
        summary.append("**🔍 Наибольший рост ожидается в сегментах:**")
        for s in top_segments:
            summary.append(f"- {s}")
    if bottom_segments:
        summary.append("")
        summary.append("**⚠️ Возможное снижение в сегменте:**")
        for s in bottom_segments:
            summary.append(f"- {s}")

    summary.append("")
    summary.append(
        "_Рекомендация: стоит внимательно отследить реакцию клиентов в проблемных сегментах после релиза._")

    return "\n".join(summary)
