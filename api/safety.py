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

app = Flask(__name__)
CORS(app)

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = "YutJn0riht9IwHkUicoPXTqsZQXX4dZfpvxdcScE"
BASE_URL = "https://api.fda.gov/drug/event.json"
USER_DB_FILE = "/tmp/users.json" if os.environ.get('VERCEL') else "users.json"

# -----------------------------
# AUTH LOGIC
# -----------------------------
def load_users():
    if os.path.exists(USER_DB_FILE):
        try:
            with open(USER_DB_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    return {
        "admin": {"password": hashlib.sha256("admin123".encode()).hexdigest(), "role": "admin"},
        "analyst": {"password": hashlib.sha256("analyst123".encode()).hexdigest(), "role": "analyst"}
    }

@app.route('/api/safety/auth', methods=['POST'])
def auth():
    data = request.json
    action = data.get('action')
    username = data.get('username')
    password = data.get('password')
    
    users = load_users()
    
    if action == 'login':
        user_data = users.get(username)
        if not user_data:
            return jsonify({"error": "User not found"}), 401
        
        hashed = hashlib.sha256(password.encode()).hexdigest()
        if hashed == user_data.get("password"):
            return jsonify({"username": username, "role": user_data.get("role")})
        return jsonify({"error": "Invalid credentials"}), 401
        
    elif action == 'signup':
        if username in users:
            return jsonify({"error": "User already exists"}), 400
        
        role = data.get('role', 'analyst')
        users[username] = {
            "password": hashlib.sha256(password.encode()).hexdigest(),
            "role": role
        }
        with open(USER_DB_FILE, "w") as f:
            json.dump(users, f)
        return jsonify({"message": "Account created"})

# -----------------------------
# ENGINE LOGIC
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
        
    # Process
    reports = []
    reactions = []
    for entry in raw:
        rid = entry.get("safetyreportid")
        reports.append({
            "id": rid,
            "serious": int(entry.get("serious", 0)),
            "country": entry.get("primarysource", {}).get("reportercountry", "Unknown"),
            "date": entry.get("receiptdate")
        })
        for r in entry.get("patient", {}).get("reaction", []):
            name = r.get("reactionmeddrapt")
            if name:
                reactions.append({"id": rid, "name": name.upper()})
                
    df_reports = pd.DataFrame(reports)
    df_reactions = pd.DataFrame(reactions)
    
    # 1. Top Reactions
    top_rx = df_reactions['name'].value_counts().head(10).to_dict()
    
    # 2. Risk Signals
    merged = df_reactions.merge(df_reports, on="id")
    risk = merged.groupby("name").agg(total=("serious", "count"), serious=("serious", "sum"))
    risk["score"] = (risk["serious"] / risk["total"]).round(3)
    top_risk = risk[risk["total"] > 2].sort_values("score", ascending=False).head(10)["score"].to_dict()
    
    # 3. PRR (Simulated with current batch)
    baseline_raw = fetch_baseline()
    b_rx = []
    for entry in baseline_raw:
        for r in entry.get("patient", {}).get("reaction", []):
            name = r.get("reactionmeddrapt")
            if name: b_rx.append(name.upper())
    
    b_counts = pd.Series(b_rx).value_counts()
    d_counts = df_reactions['name'].value_counts()
    
    prr_list = []
    total_b = len(b_rx)
    total_d = len(df_reactions)
    
    for rx in d_counts.index[:15]:
        A = d_counts[rx]
        C = b_counts.get(rx, 0)
        if C > 0:
            prr = (A/total_d) / (C/total_b)
            prr_list.append({"name": rx, "prr": round(prr, 2)})
            
    # 4. ML Prediction (Simplified for API speed)
    # We'll just return the reaction importance based on the current batch
    importance = d_counts.head(5).to_dict()
    
    return jsonify({
        "metrics": {
            "total": len(df_reports),
            "serious": int(df_reports['serious'].sum()),
            "ratio": f"{(df_reports['serious'].sum()/len(df_reports)*100):.1f}%"
        },
        "top_reactions": top_rx,
        "risk_signals": top_risk,
        "prr_signals": prr_list,
        "importance": importance
    })

if __name__ == "__main__":
    app.run(port=5001)
