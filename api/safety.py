from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
import hashlib
import json
import os
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

app = Flask(__name__)
CORS(app)

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = "YutJn0riht9IwHkUicoPXTqsZQXX4dZfpvxdcScE"
BASE_URL = "https://api.fda.gov/drug/event.json"
USER_DB_FILE = "/tmp/users.json" if os.environ.get('VERCEL') else "users.json"

# -----------------------------
# FETCH DATA
# -----------------------------
def fetch_fda_data(drug_name, limit=1000):
    params = {
        "api_key": API_KEY,
        "search": f"patient.drug.medicinalproduct:{drug_name}",
        "limit": limit
    }
    try:
        r = requests.get(BASE_URL, params=params, timeout=15)
        return r.json().get("results", [])
    except:
        return []

def fetch_baseline():
    params = {
        "api_key": API_KEY,
        "search": "_exists_:patient.reaction",
        "limit": 1000
    }
    try:
        r = requests.get(BASE_URL, params=params, timeout=15)
        return r.json().get("results", [])
    except:
        return []

@app.route('/api/safety/analyze', methods=['POST'])
def analyze():
    data = request.json
    drug_name = data.get('drug', 'aspirin')
    
    raw = fetch_fda_data(drug_name)
    if not raw:
        return jsonify({"error": "No records found"}), 404
        
    reports = []
    reactions = []
    seen_reports = set()
    for entry in raw:
        rid = entry.get("safetyreportid")
        
        # Deduplicate reports and apply both binary and weighted severity
        if rid and rid not in seen_reports:
            # Binary flag for table and counts
            is_serious = 1 if (
                entry.get("serious") == "1" or 
                entry.get("seriousnessdeath") == "1" or 
                entry.get("seriousnesslifethreatening") == "1" or 
                entry.get("seriousnesshospitalization") == "1" or 
                entry.get("seriousnessdisabling") == "1" or 
                entry.get("seriousnessother") == "1"
            ) else 0
            
            # Additive Intensity for the High Risk Signals graph
            score = 0
            if str(entry.get("serious")) == "1": score += 0.25
            if str(entry.get("seriousnessdeath")) == "1": score += 0.40
            if str(entry.get("seriousnesslifethreatening")) == "1": score += 0.20
            if str(entry.get("seriousnesshospitalization")) == "1": score += 0.10
            if str(entry.get("seriousnessdisabling")) == "1": score += 0.05
            
            reports.append({
                "report_id": rid,
                "serious": is_serious,
                "severity": round(min(score, 1.0), 3),
                "country": entry.get("primarysource", {}).get("reportercountry", "Unknown"),
                "date": entry.get("receiptdate")
            })
            seen_reports.add(rid)

        # Reactions are processed for all entries (can have multiple per report)
        for r in entry.get("patient", {}).get("reaction", []):
            name = r.get("reactionmeddrapt")
            if name:
                reactions.append({"report_id": rid, "reaction_name": name.upper()})
                
    df_reports = pd.DataFrame(reports)
    df_reactions = pd.DataFrame(reactions)
    
    accuracy = "0.00%"
    importance = 0.0
    
    if not df_reports.empty and not df_reactions.empty:
        # Ensure report_id types match for merge
        df_reports["report_id"] = df_reports["report_id"].astype(str)
        df_reactions["report_id"] = df_reactions["report_id"].astype(str)
        
        merged = df_reactions.merge(df_reports, on="report_id")
        
        if len(merged) > 20:
            reaction_counts = merged["reaction_name"].value_counts().to_dict()
            merged["reaction_freq"] = merged["reaction_name"].map(reaction_counts)
            
            X = merged[["reaction_freq"]].values
            y = merged["serious"].values
            
            if len(np.unique(y)) > 1:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    acc = accuracy_score(y_test, preds)
                    accuracy = f"{acc:.2%}"
                    # For a single feature, importance is either 1.0 (if split) or 0.0
                    importance = float(model.feature_importances_[0])
                except:
                    pass

    # Trends Logic
    df_reports['date'] = pd.to_datetime(df_reports['date'], errors='coerce')
    df_reports['month'] = df_reports['date'].dt.strftime('%Y-%m')
    trend = df_reports.groupby('month').size().tail(12).to_dict()

    # Baseline for PRR
    baseline_raw = fetch_baseline()
    baseline_reactions = []
    for entry in baseline_raw:
        for r in entry.get("patient", {}).get("reaction", []):
            name = r.get("reactionmeddrapt")
            if name: baseline_reactions.append(name.upper())

    return jsonify({
        "reports": reports,
        "reactions": reactions,
        "baseline_reactions": baseline_reactions,
        "accuracy": accuracy,
        "importance": round(importance, 3),
        "trend_data": trend
    })

if __name__ == "__main__":
    app.run(port=5001)
