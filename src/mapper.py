import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

def load_portraits(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def preprocess_data(df: pd.DataFrame):
    # Категориальные и числовые признаки
    cat_features = ["client_type", "fuel_type", "loyalty_card", "fuel_card", "contract"]
    num_features = ["visits_per_month", "avg_liters_per_visit", "avg_spend_per_visit"]

    # Заполняем пропуски
    df[cat_features] = df[cat_features].fillna("Неизвестно")
    df[num_features] = df[num_features].fillna(df[num_features].mean())

    # One-Hot кодирование категориальных признаков
    encoder = OneHotEncoder(sparse_output=False)
    cat_encoded = encoder.fit_transform(df[cat_features])
    cat_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_features))

    # Стандартизация числовых признаков
    scaler = StandardScaler()
    num_scaled = scaler.fit_transform(df[num_features])
    num_df = pd.DataFrame(num_scaled, columns=num_features)

    # Объединяем
    processed = pd.concat([num_df, cat_df], axis=1)
    return processed, encoder, scaler

def cluster_clients(processed_df: pd.DataFrame, n_clusters=15):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = model.fit_predict(processed_df)
    processed_df["cluster"] = clusters
    return model, processed_df

def compute_score(client_row, portrait_criteria):
    score = 0
    # Числовые признаки
    for field in ["visits_per_month", "avg_liters_per_visit", "avg_spend_per_visit"]:
        if field in portrait_criteria:
            val = client_row.get(field, 0)
            min_val, max_val = portrait_criteria[field]
            if min_val <= val <= max_val:
                score += 0.5  # числовой вес меньше, чем категориальный
    # Категориальные признаки
    for field in ["client_type", "fuel_type", "loyalty_card", "fuel_card", "contract"]:
        if field in portrait_criteria and client_row.get(field) == portrait_criteria[field]:
            score += 1
    return score

def assign_portraits(df: pd.DataFrame, portraits: list):
    assigned = []
    for idx, row in df.iterrows():
        best_score = -1
        best_portrait = None
        for p in portraits:
            score = compute_score(row, p["criteria"])
            if score > best_score:
                best_score = score
                best_portrait = p["portrait_name"]
        assigned.append(best_portrait or "Неопределенный тип")
    df["portrait_name"] = assigned
    return df

def map_clients_to_portraits(df: pd.DataFrame, portraits: list):
    processed, encoder, scaler = preprocess_data(df)
    model, processed_df = cluster_clients(processed, n_clusters=len(portraits)+5)
    mapped_df = assign_portraits(df, portraits)
    return mapped_df

if __name__ == "__main__":
    df = pd.read_csv("data/synthetic.csv")
    portraits = load_portraits("src/portraits.json")
    mapped = map_clients_to_portraits(df, portraits)
    mapped.to_csv("data/synthetic_mapped.csv", index=False)
    print("Mapping done. Saved to data/synthetic_mapped.csv")