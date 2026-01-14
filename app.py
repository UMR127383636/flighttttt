# app.py
# -*- coding: utf-8 -*-

import os, json
import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from feature_builder import FeatureBuilderNoDelayRate
import __main__
__main__.FeatureBuilderNoDelayRate = FeatureBuilderNoDelayRate  # 兼容 joblib 反序列化

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ✅ 用相对路径（部署/本地都通用）
RESULT_DIR = os.path.join(BASE_DIR, "results_cls_no_delayrate_fast_simple")
MODEL_PATH = os.path.join(RESULT_DIR, "best_model_classifier.joblib")
ROUTE_JSON = os.path.join(RESULT_DIR, "route_constraints.json")
INDEX_HTML = os.path.join(BASE_DIR, "index.html")

# 训练里出现过的列名（保留）
DEP_BIN_COL = "起飞时间离散化"
ARR_BIN_COL = "到达时间离散化"
ARR_TIME_COL = "到达时间"

app = FastAPI(title="Flight Delay Prediction API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"模型文件不存在: {MODEL_PATH}")
if not os.path.exists(ROUTE_JSON):
    raise FileNotFoundError(f"联动JSON不存在: {ROUTE_JSON}")
if not os.path.exists(INDEX_HTML):
    raise FileNotFoundError(f"index.html 不存在: {INDEX_HTML}")

model = joblib.load(MODEL_PATH)

with open(ROUTE_JSON, "r", encoding="utf-8") as f:
    route_constraints = json.load(f)

# ✅ 从 constraints 推导全量 airlines/airports/routes（你要的“所有飞机和所有航线”）
def build_all_lists(constraints: dict):
    airlines = sorted(constraints.keys())
    airports = set()
    routes = set()
    for air, mp in (constraints or {}).items():
        for frm, tos in (mp or {}).items():
            airports.add(frm)
            for to in (tos or []):
                airports.add(to)
                routes.add(f"{frm}-{to}")
    return airlines, sorted(airports), sorted(routes)

airlines_all, airports_all, routes_all = build_all_lists(route_constraints)

# ====== 折扣（8档更细） ======
def discount_suggestion(p: float):
    p = float(p)
    bins = [
        (0.10, "VeryLow",   0),
        (0.20, "Low",      -5),
        (0.30, "Medium",  -10),
        (0.40, "MedHigh", -15),
        (0.50, "High",    -25),
        (0.60, "VeryHigh",-35),
        (0.70, "Severe",  -45),
        (1.01, "Extreme", -55),
    ]
    for th, label, disc in bins:
        if p < th:
            return label, disc
    return "Extreme", -55

# ====== 时间段编码（上午/下午/晚上）=====
def label_to_code(label: str) -> int:
    s = str(label).strip()
    if s == "上午": return 0
    if s == "下午": return 1
    if s == "晚上": return 2
    raise HTTPException(status_code=422, detail="DepBin/ArrBin 只能是：上午 / 下午 / 晚上")

# 训练里“离散化列”可能是数字也可能是中文字符串
# ✅ 最稳：两种都发（模型需要哪个就会用哪个）
def bin_numeric_and_string(label: str):
    code = label_to_code(label)
    zh = {0: "上午", 1: "下午", 2: "晚上"}[code]
    return int(code), zh

class PredictIn(BaseModel):
    Airline: str
    AirportFrom: str
    AirportTo: str
    DayOfWeek: int
    Length: float
    DepBin: str   # 上午/下午/晚上
    ArrBin: str   # 上午/下午/晚上

@app.get("/")
def web():
    # ✅ 打开 Render 链接就看到网页
    return FileResponse(INDEX_HTML)

@app.get("/options")
def options():
    return {
        "constraints": route_constraints,
        "all": {
            "airlines": airlines_all,
            "airports": airports_all,
            "routes": routes_all
        }
    }

@app.get("/debug/expected")
def debug_expected():
    cols = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
    return {"expected_columns": cols, "has_feature_names_in": bool(cols)}

@app.post("/predict")
def predict(x: PredictIn):
    dep_num, dep_zh = bin_numeric_and_string(x.DepBin)
    arr_num, arr_zh = bin_numeric_and_string(x.ArrBin)

    row = {
        "Airline": str(x.Airline),
        "AirportFrom": str(x.AirportFrom),
        "AirportTo": str(x.AirportTo),
        "DayOfWeek": int(x.DayOfWeek),
        "Length": float(x.Length),

        # 训练里出现过的到达时间列，给个缺省（不影响）
        ARR_TIME_COL: np.nan,

        # ✅ 两种都给
        DEP_BIN_COL: dep_num,
        ARR_BIN_COL: arr_num,
        f"{DEP_BIN_COL}_str": dep_zh,
        f"{ARR_BIN_COL}_str": arr_zh,
    }

    df = pd.DataFrame([row])

    expected = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
    if expected:
        for c in expected:
            if c not in df.columns:
                df[c] = np.nan
        df = df[expected]

    try:
        p = float(model.predict_proba(df)[:, 1][0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {repr(e)}")

    level, disc = discount_suggestion(p)

    return {
        "delay_probability": p,
        "delay_probability_percent": round(p * 100, 2),
        "risk_level": level,
        "discount_percent": disc,
        "dep_bin": x.DepBin,
        "arr_bin": x.ArrBin,
    }
