import pandas as pd
import random
from faker import Faker
import os

fake = Faker('ru_RU')


def generate_clients(n=1000):
    data = []
    for _ in range(n):
        client_type = random.choices(
            ["Физическое лицо", "Юридическое лицо"], weights=[0.8, 0.2])[0]

        if client_type == "Физическое лицо":
            loyalty_card = random.choices(["Да", "Нет"], weights=[0.7, 0.3])[0]
            fuel_card = random.choices(["Да", "Нет"], weights=[0.15, 0.85])[0]
            contract = "Нет"
            fuel_type = random.choices(
                ["Бензин", "Дизель"], weights=[0.7, 0.3])[0]
            tank_volume = random.randint(40, 70)
            avg_liters_per_visit = random.randint(20, 50)
            visits_per_month = random.randint(2, 10)
            avg_spend_per_visit = random.randint(1500, 4000)

        else:  # юрики
            loyalty_card = random.choices(["Да", "Нет"], weights=[0.1, 0.9])[0]
            fuel_card = random.choices(["Да", "Нет"], weights=[0.9, 0.1])[0]
            contract = "Да"
            fuel_type = random.choices(
                ["Дизель", "Газ", "Бензин"], weights=[0.6, 0.3, 0.1])[0]
            tank_volume = random.randint(80, 150)
            avg_liters_per_visit = random.randint(60, 120)
            visits_per_month = random.randint(5, 20)
            avg_spend_per_visit = random.randint(5000, 15000)

        client = {
            "client_id": fake.uuid4(),
            "client_type": client_type,
            "loyalty_card": loyalty_card,
            "fuel_card": fuel_card,
            "contract": contract,
            "fuel_type": fuel_type,
            "tank_volume": tank_volume,
            "avg_liters_per_visit": avg_liters_per_visit,
            "visits_per_month": visits_per_month,
            "avg_spend_per_visit": avg_spend_per_visit,
            "region": fake.city()
        }

        data.append(client)

    return pd.DataFrame(data)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df = generate_clients(1000)
    file_path = "data/synthetic.csv"
    df.to_csv(file_path, index=False, encoding='utf-8')
