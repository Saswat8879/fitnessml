# model_server_sklearn.py
import os
from typing import List, Optional
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import uvicorn

MODEL_PATH = os.environ.get("MODEL_PATH", "models/plate_ranker_lgb.pkl")
if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Train model and place it there.")
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Plate Ranker (Sklearn LGBM)", version="1.0")
class Candidate(BaseModel):
    plate_id: str
    protein: int
    carbs: int
    fat: int
    calories: int
    prep_time: Optional[int] = 15
    is_veg: Optional[bool] = False
    popularity: Optional[float] = 0.01
    is_favorite: Optional[int] = 0

class Context(BaseModel):
    user_id: str
    age: int
    sex: str
    goal: str
    meal_time: str
    calories_left_ratio: float

class ScoreRequest(BaseModel):
    context: Context
    candidates: List[Candidate]
    top_k: Optional[int] = 5


def target_for_goal(goal: str):
    if goal == "lose":
        return (40, 80, 40)
    if goal == "gain":
        return (60, 250, 60)
    return (45, 180, 55)

def build_feature_row(context: Context, c: Candidate):
    p_target, c_target, f_target = target_for_goal(context.goal)
    protein_diff = abs(c.protein - p_target)
    carbs_diff = abs(c.carbs - c_target)
    fat_diff = abs(c.fat - f_target)
    calories_per_p = c.calories / (c.protein if c.protein > 0 else 0.1)
    is_male = 1 if context.sex.lower() == 'male' else 0

    meal_breakfast = 1 if context.meal_time == 'breakfast' else 0
    meal_lunch = 1 if context.meal_time == 'lunch' else 0
    meal_dinner = 1 if context.meal_time == 'dinner' else 0

    row = {
        'age': context.age,
        'is_male': is_male,
        'protein_diff': protein_diff,
        'carbs_diff': carbs_diff,
        'fat_diff': fat_diff,
        'calories': c.calories,
        'calories_per_p': calories_per_p,
        'prep_time': c.prep_time,
        'is_veg': int(c.is_veg),
        'popularity': c.popularity,
        'is_favorite': int(c.is_favorite),
        'calories_left_ratio_clipped': max(0.0, min(context.calories_left_ratio, 2.0)),
        'goal_lose': 1 if context.goal == 'lose' else 0,
        'goal_gain': 1 if context.goal == 'gain' else 0,
        'meal_breakfast': meal_breakfast,
        'meal_lunch': meal_lunch,
        'meal_dinner': meal_dinner
    }
    return row

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "model_path": MODEL_PATH}

@app.post("/score")
def score(req: ScoreRequest):
    if not req.candidates:
        raise HTTPException(status_code=400, detail="No candidates provided")

    rows = []
    id_map = []
    for c in req.candidates:
        rows.append(build_feature_row(req.context, c))
        id_map.append(c.plate_id)

    X = pd.DataFrame(rows)

    try:
        feat_names = model.booster_.feature_name() if hasattr(model, "booster_") else None
    except Exception:
        feat_names = None

    if feat_names:

        cols = [f for f in feat_names if f in X.columns]
        X = X[cols]

    try:
        probs = model.predict_proba(X)[:, 1]
    except Exception:
        probs = model.predict(X).astype(float)

    scored = list(zip(id_map, probs, req.candidates))
    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[: req.top_k]

    results = []
    for pid, score_val, c in top:
        results.append({
            "plate_id": pid,
            "score": float(score_val),
            "plate": c.dict()
        })

    return {"top_k": req.top_k, "results": results}

if __name__ == "__main__":
    uvicorn.run("model_server_sklearn:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
