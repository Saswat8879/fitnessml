import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

random.seed(42)
np.random.seed(42)

NUM_USERS = 500
NUM_PLATES = 200
IMPRS_PER_USER = 100

users = []
for uid in range(NUM_USERS):
    users.append({
        "user_id": f"user_{uid}",
        "age": int(np.clip(np.random.normal(30, 8), 18, 65)),
        "sex": random.choice(["male", "female"]),
        "goal": random.choice(["lose", "maintain", "gain"]),
        "diet_pref": random.choice(["no_pref", "veg", "vegan", "halal", "pescatarian", "kosher"]),
        "activity_level": random.choice(["sedentary","light","moderate","active"])
    })

plates = []
for pid in range(NUM_PLATES):
    protein = int(np.clip(np.random.normal(25, 10), 5, 80))
    carbs   = int(np.clip(np.random.normal(50, 25), 5, 300))
    fat     = int(np.clip(np.random.normal(20, 10), 0, 100))
    calories = int(protein*4 + carbs*4 + fat*9)
    cuisine = random.choice(["indian", "mediterranean", "american", "asian", "mexican"])
    plates.append({
        "plate_id": f"plate_{pid}",
        "protein": protein,
        "carbs": carbs,
        "fat": fat,
        "calories": calories,
        "cuisine": cuisine,
        "prep_time": random.choice([10, 15, 20, 30, 45]),
        "is_veg": random.choice([True, False]),
        "popularity": np.clip(np.random.beta(2,5), 0.01, 1.0)
    })

rows = []
start_ts = datetime.utcnow() - timedelta(days=30)
for user in users:
    for i in range(IMPRS_PER_USER):
        ts = (start_ts + timedelta(seconds=random.randint(0, 30*24*3600))).isoformat()
        meal_time = random.choice(["breakfast","lunch","dinner","snack"])
        calories_left_ratio = np.clip(np.random.normal(0.5, 0.25), 0.0, 1.5)

        plate = random.choice(plates)
        is_favorite = random.random() < 0.05 
        if user["goal"] == "lose":
            target_protein = 40
            target_carbs = 80
            target_fat = 40
        elif user["goal"] == "gain":
            target_protein = 60
            target_carbs = 250
            target_fat = 60
        else:
            target_protein = 45
            target_carbs = 180
            target_fat = 55

        macro_dist = abs(plate["protein"] - target_protein) + abs(plate["carbs"] - target_carbs) + abs(plate["fat"] - target_fat)
        base_prob = np.clip(0.5 - 0.002 * macro_dist + 0.2 * plate["popularity"], 0.01, 0.95)
        if is_favorite:
            base_prob += 0.12
        if meal_time == "breakfast" and plate["carbs"] > 120:
            base_prob -= 0.08

        label = 1 if random.random() < base_prob else 0

        rows.append({
            "user_id": user["user_id"],
            "age": user["age"],
            "sex": user["sex"],
            "goal": user["goal"],
            "diet_pref": user["diet_pref"],
            "activity_level": user["activity_level"],
            "timestamp": ts,
            "meal_time": meal_time,
            "calories_left_ratio": calories_left_ratio,
            "plate_id": plate["plate_id"],
            "protein": plate["protein"],
            "carbs": plate["carbs"],
            "fat": plate["fat"],
            "calories": plate["calories"],
            "prep_time": plate["prep_time"],
            "is_veg": plate["is_veg"],
            "popularity": plate["popularity"],
            "is_favorite": int(is_favorite),
            "outcome": label
        })

df = pd.DataFrame(rows)
df.to_csv("training_data.csv", index=False)
print("Wrote training_data.csv with", len(df), "rows")
