import pandas as pd
import numpy as np

def engineer(df):
    def target_for_goal(goal):
        if goal == "lose":
            return (40, 80, 40)
        if goal == "gain":
            return (60, 250, 60)
        return (45, 180, 55)
    prot_targets = []
    carb_targets = []
    fat_targets = []
    for g in df['goal']:
        p,c,f = target_for_goal(g)
        prot_targets.append(p)
        carb_targets.append(c)
        fat_targets.append(f)

    df['prot_target'] = prot_targets
    df['carb_target'] = carb_targets
    df['fat_target'] = fat_targets

    df['protein_diff'] = (df['protein'] - df['prot_target']).abs()
    df['carbs_diff']   = (df['carbs'] - df['carb_target']).abs()
    df['fat_diff']     = (df['fat'] - df['fat_target']).abs()

    df['calories_per_p'] = df['calories'] / (df['protein'].replace(0, 0.1))
    df['calories_left_ratio_clipped'] = df['calories_left_ratio'].clip(0, 2)

    df['is_male'] = (df['sex'] == 'male').astype(int)
    df['goal_lose'] = (df['goal'] == 'lose').astype(int)
    df['goal_gain'] = (df['goal'] == 'gain').astype(int)
    df['meal_breakfast'] = (df['meal_time'] == 'breakfast').astype(int)
    df['meal_lunch'] = (df['meal_time'] == 'lunch').astype(int)
    df['meal_dinner'] = (df['meal_time'] == 'dinner').astype(int)

    features = [
        'age', 'is_male', 'protein_diff', 'carbs_diff', 'fat_diff',
        'calories', 'calories_per_p', 'prep_time', 'is_veg', 'popularity',
        'is_favorite', 'calories_left_ratio_clipped', 'goal_lose', 'goal_gain',
        'meal_breakfast','meal_lunch','meal_dinner'
    ]
    features = [c for c in features if c in df.columns]
    X = df[features].fillna(0)
    y = df['outcome']

    return X, y, df

if __name__ == "__main__":
    df = pd.read_csv("training_data.csv")
    X, y, df_full = engineer(df)
    X.to_csv("features.csv", index=False)
    df_full.to_csv("training_data_with_targets.csv", index=False)
    print("Wrote features.csv and training_data_with_targets.csv")
