import json
import pandas as pd
import numpy as np
from math import sqrt
from sklearn.preprocessing import StandardScaler

NUMERIC_FEATURES = ["age", "avg_check", "visits_per_month", "distance_km", "uses_app", "prefers_coffee"]

def load_portraits(path="src/portraits.json"):
    with open(path, "r", encoding="utf-8") as f:
        portraits = json.load(f)
    return portraits

def clients_to_matrix(df):
    return df[NUMERIC_FEATURES].astype(float).values

def portraits_to_matrix(portraits):
    rows = []
    for p in portraits:
        proto = p["prototype"]
        row = [float(proto.get(f, 0)) for f in NUMERIC_FEATURES]
        rows.append(row)
    return np.array(rows)

def fit_scaler(X_clients, X_portraits):
    scaler = StandardScaler()
    # fit on combined set to use same scale
    scaler.fit(np.vstack([X_clients, X_portraits]))
    return scaler

def map_clients_to_portraits(df_clients, portraits):
    Xc = clients_to_matrix(df_clients)
    Xp = portraits_to_matrix(portraits)
    scaler = fit_scaler(Xc, Xp)
    Xc_s = scaler.transform(Xc)
    Xp_s = scaler.transform(Xp)
    assignments = []
    for i, x in enumerate(Xc_s):
        # compute euclidean distances to prototypes
        dists = np.linalg.norm(Xp_s - x, axis=1)
        idx = int(np.argmin(dists))
        pat = portraits[idx]
        assignments.append({
            "client_index": i,
            "portrait_id": pat["id"],
            "portrait_name": pat["name"],
            "distance": float(dists[idx])
        })
    return pd.DataFrame(assignments)

if __name__ == "__main__":
    df = pd.read_csv("data/synthetic.csv")
    portraits = load_portraits("src\portraits.json")
    mapped = map_clients_to_portraits(df, portraits)
    # join for sample
    df["assigned_portrait"] = mapped["portrait_name"]
    df.to_csv("data/synthetic_mapped.csv", index=False)
    print("Mapped saved to synthetic_mapped.csv")
