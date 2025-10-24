import pandas as pd
import numpy as np
from faker import Faker

fake = Faker('ru_RU')

CAR_TYPES = ["седан", "внедорожник", "хэтчбек", "универсал", "грузовик"]
FUEL_TYPES = ["АИ-92", "АИ-95", "ДТ"]

def generate_clients(n=1000, seed=42):
    np.random.seed(seed)
    rows = []
    for i in range(n):
        age = int(np.clip(np.random.normal(35, 10), 18, 75))
        gender = np.random.choice(["М", "Ж"], p=[0.6, 0.4])
        car = np.random.choice(CAR_TYPES, p=[0.35,0.2,0.25,0.15,0.05])
        fuel = np.random.choice(FUEL_TYPES, p=[0.45,0.45,0.10])
        avg_check = int(np.clip(np.random.normal(1500 if fuel=="АИ-95" else 1000, 400), 200, 10000))
        visits_per_month = float(np.clip(np.random.exponential(1.0) + (0.5 if car=="грузовик" else 0.0), 0.1, 20))
        distance_km = float(np.clip(np.random.normal(5 if car!="грузовик" else 30, 8), 0.1, 200))
        # contextual features example (0/1)
        uses_app = int(np.random.rand() < 0.4)
        prefers_coffee = int(np.random.rand() < 0.25)
        rows.append({
            "client_id": f"synt_{i+1}",
            "age": age,
            "gender": gender,
            "car_type": car,
            "fuel_type": fuel,
            "avg_check": avg_check,
            "visits_per_month": round(visits_per_month,2),
            "distance_km": round(distance_km,2),
            "uses_app": uses_app,
            "prefers_coffee": prefers_coffee
        })
    df = pd.DataFrame(rows)
    return df

if __name__ == "__main__":
    df = generate_clients(1000)
    df.to_csv("data/synthetic.csv", index=False)
    print("synthetic.csv created with", len(df), "rows")