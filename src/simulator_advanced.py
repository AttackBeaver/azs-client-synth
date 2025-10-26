import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

# загрузка данных


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# генерация откликов


def simulate_feature_response(clients_df: pd.DataFrame,
                              portraits_rules: dict,
                              feature_hypotheses: list,
                              selected_feature: str):
    """
    Симуляция реакции клиентов на выбранную фичу.
    Возвращает датафрейм с откликами и статистикой по портретам.
    """
    df = clients_df.copy()

    # найдем гипотезу
    feature = next(
        (f for f in feature_hypotheses if f["feature_name"] == selected_feature), None)
    if feature is None:
        raise ValueError(
            f"Feature {selected_feature} not found in hypotheses.")

    target_metric = feature["target_metric"]
    applicable_portraits = feature["applicable_to"]

    # добавим колонку "реакция" (0-1)
    responses = []
    for idx, row in df.iterrows():
        portrait = row.get("portrait_name")
        base_rules = portraits_rules.get(portrait, {})
        # минимальная вероятность отклика
        base_prob = base_rules.get(target_metric, 0.05)
        # добавим случайность и зависимость от числовых признаков
        factor = np.mean([
            min(row.get("visits_per_month", 0)/10, 1),
            min(row.get("avg_spend_per_visit", 0)/10000, 1)
        ])
        # итоговая вероятность
        prob = base_prob + 0.5*factor
        prob = min(prob, 1.0)
        response = np.random.binomial(
            1, prob) if portrait in applicable_portraits else 0
        responses.append(response)

    df[f"response_to_{selected_feature}"] = responses

    # кластеризация по отклику для визуализации паттернов
    scaler = StandardScaler()
    num_features = ["visits_per_month",
                    "avg_liters_per_visit", "avg_spend_per_visit"]
    scaled = scaler.fit_transform(df[num_features])
    kmeans = KMeans(n_clusters=min(8, len(df)//50), random_state=42)
    df["response_cluster"] = kmeans.fit_predict(scaled)

    # метрики по портретам
    metrics = df.groupby("portrait_name")[f"response_to_{selected_feature}"].agg([
        "mean", "sum", "count"
    ]).rename(columns={"mean": "response_rate", "sum": "total_responses", "count": "total_clients"}).reset_index()

    return df, metrics
