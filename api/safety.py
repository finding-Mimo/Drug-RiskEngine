from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import pandas as pd
import numpy as np
import hashlib
import json
import os
import time
from sklearn.model_selection import train_test_splitimport requests
import pandas as pd
import time
import matplotlib.pyplot as plt

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = "YutJn0riht9IwHkUicoPXTqsZQXX4dZfpvxdcScE"
BASE_URL = "https://api.fda.gov/drug/event.json"
DRUG_NAME = "aspirin"
LIMIT = 1000
SLEEP_TIME = 1
TOTAL_DRUG_RECORDS = 500
TOTAL_BASELINE_RECORDS = 5000

# -----------------------------
# FETCH DATA
# -----------------------------
def fetch_data(search_query, total_records):
    """Generic function to fetch data from OpenFDA"""
    all_results = []
    skip = 0
    headers = {"User-Agent": "Mozilla/5.0"}

    while len(all_results) < total_records:
        params = {
            "api_key": API_KEY,
            "search": search_query,
            "limit": min(LIMIT, total_records - len(all_results)),
            "skip": skip
        }

        try:
            response = requests.get(BASE_URL, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()
            results = data.get("results", [])

            if not results:
                break

            all_results.extend(results)
            skip += LIMIT
            print(f"Fetched {len(all_results)} records for query: {search_query[:50]}...")
            
            if len(results) < LIMIT:
                break
                
            time.sleep(SLEEP_TIME)

        except Exception as e:
            print(f"Fetch error: {e}")
            break

    return all_results[:total_records]

# -----------------------------
# DATA NORMALIZATION
# -----------------------------
def process_data(raw_data):
    """Extracts reaction and report info"""
    reaction_rows = []
    report_rows = []
    seen_reports = set()
    for entry in raw_data:
        report_id = entry.get("safetyreportid")
        serious = entry.get("serious", "2")
        date_received = entry.get("receiptdate")
        
        for reaction in entry.get("patient", {}).get("reaction", []):
            name = reaction.get("reactionmeddrapt")
            if name:
                reaction_rows.append({
                    "report_id": report_id,
                    "reaction_name": name.upper()
                })
        
        if report_id and report_id not in seen_reports:
            # Additive scoring to ensure differentiation
            score = 0
            if entry.get("serious") == "1": score += 0.25
            if entry.get("seriousnessdeath") == "1": score += 0.40
            if entry.get("seriousnesslifethreatening") == "1": score += 0.20
            if entry.get("seriousnesshospitalization") == "1": score += 0.10
            if entry.get("seriousnessdisabling") == "1": score += 0.05

            report_rows.append({
                "report_id": report_id,
                "serious": round(min(score, 1.0), 3),
                "date_received": date_received
            })
            seen_reports.add(report_id)

    return pd.DataFrame(reaction_rows), pd.DataFrame(report_rows)

def extract_reactions_only(raw_data):
    """Simplified extractor for baseline"""
    rows = []
    for entry in raw_data:
        report_id = entry.get("safetyreportid")
        for reaction in entry.get("patient", {}).get("reaction", []):
            name = reaction.get("reactionmeddrapt")
            if name:
                rows.append({
                    "report_id": report_id,
                    "reaction_name": name.upper()
                })
    return pd.DataFrame(rows)

# -----------------------------
# VISUALIZATION & ANOMALIES
# -----------------------------
def plot_top_signals(strong_signals):
    if strong_signals.empty:
        print("No strong signals to plot.")
        return
        
    top10 = strong_signals.head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(top10["reaction"], top10["PRR"], color='skyblue')
    plt.xlabel("Proportional Reporting Ratio (PRR)")
    plt.ylabel("Reaction")
    plt.title(f"Top 10 Adverse Event Signals for {DRUG_NAME.capitalize()}")
    plt.gca().invert_yaxis()
    plt.tight_layout()

    plt.savefig("top_signals_chart.png")
    print("Chart saved: top_signals_chart.png")
    # plt.show() # Disabled to prevent blocking execution

def detect_anomalies(reports_df):
    print("\n--- ANOMALY DETECTION ---")

    # Convert date
    reports_df["date_received"] = pd.to_datetime(
        reports_df["date_received"], errors="coerce"
    )

    # Monthly aggregation
    reports_df["month"] = reports_df["date_received"].dt.to_period("M")
    monthly_counts = reports_df.groupby("month").size()

    # Convert to numeric index
    monthly_counts = monthly_counts.sort_index()

    if monthly_counts.empty:
        print("No date data available for anomaly detection.")
        return pd.Series()

    mean = monthly_counts.mean()
    std = monthly_counts.std()
    
    # Handle cases with very low variance
    if pd.isna(std): std = 0

    threshold = mean + 2 * std
    anomalies = monthly_counts[monthly_counts > threshold]

    print("\nMonthly Report Counts:")
    print(monthly_counts.tail(10))

    print("\nAnomaly Threshold:", round(threshold, 2))

    if not anomalies.empty:
        print("\nDetected Anomalies (Spikes):")
        print(anomalies)
    else:
        print("\nNo anomalies detected.")

    # Plot Trend
    plt.figure(figsize=(12, 6))
    plt.plot(monthly_counts.index.astype(str), monthly_counts.values, marker='o', linestyle='-')
    plt.xticks(rotation=45)
    plt.xlabel("Month")
    plt.ylabel("Number of Reports")
    plt.title(f"Monthly Adverse Event Trend for {DRUG_NAME.capitalize()}")

    # Highlight anomalies
    for i, val in enumerate(monthly_counts.values):
        if val > threshold:
            plt.scatter(monthly_counts.index.astype(str)[i], val, color='red', s=100, label='Anomaly')

    plt.tight_layout()
    plt.savefig("anomaly_trend.png")
    print("Trend chart saved: anomaly_trend.png")
    # plt.show()

    return anomalies

# -----------------------------
# CORE SIGNAL DETECTION
# -----------------------------
def compute_signals(reports_df, reactions_df):
    print("\n--- SIGNAL DETECTION ---")

    # Fetch baseline
    print(f"Fetching baseline data ({TOTAL_BASELINE_RECORDS} records)...")
    baseline_raw = fetch_data("_exists_:patient.reaction", TOTAL_BASELINE_RECORDS)
    baseline_df = extract_reactions_only(baseline_raw)

    baseline_counts = baseline_df["reaction_name"].value_counts()
    total_baseline = len(baseline_df)

    aspirin_counts = reactions_df["reaction_name"].value_counts()
    total_aspirin = len(reactions_df)

    # Merge with reports (for seriousness)
    merged = reactions_df.merge(reports_df, on="report_id")

    signal_rows = []
    for reaction in aspirin_counts.index:
        A = aspirin_counts.get(reaction, 0)
        B = total_aspirin - A
        C = baseline_counts.get(reaction, 0)
        D = total_baseline - C

        # Avoid division issues
        if C == 0 or (A+B) == 0 or (C+D) == 0:
            continue

        # PRR Calculation
        prr = (A/(A+B)) / (C/(C+D))

        # Risk Score Calculation
        subset = merged[merged["reaction_name"] == reaction]
        serious_cases = subset["serious"].sum()
        total_cases = len(subset)
        risk_score = serious_cases / total_cases if total_cases > 0 else 0

        signal_rows.append({
            "reaction": reaction,
            "count": A,
            "PRR": round(prr, 2),
            "risk_score": round(risk_score, 2)
        })

    signal_df = pd.DataFrame(signal_rows)

    # Filter Strong Signals
    strong_signals = signal_df[
        (signal_df["PRR"] > 2) & 
        (signal_df["count"] > 5)
    ].sort_values(by="PRR", ascending=False)

    return signal_df, strong_signals

# -----------------------------
# OUTPUT RESULTS
# -----------------------------
def generate_outputs(signal_df, strong_signals):
    print("\nTop Likely True Side Effects:")
    print(strong_signals.head(10))

    # Save outputs
    signal_df.to_csv("all_signals.csv", index=False)
    strong_signals.to_csv("top_signals.csv", index=False)

    print("\nSaved:")
    print("- all_signals.csv")
    print("- top_signals.csv")

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def main():
    print(f"Starting pipeline for {DRUG_NAME}...")
    
    # 1. Fetch Target Drug Data
    drug_raw = fetch_data(f"patient.drug.medicinalproduct:{DRUG_NAME}", TOTAL_DRUG_RECORDS)
    reactions_df, reports_df = process_data(drug_raw)
    
    # 2. Run Signal Detection
    signal_df, strong_signals = compute_signals(reports_df, reactions_df)
    
    # 3. Visualization and Anomaly Detection
    plot_top_signals(strong_signals)
    detect_anomalies(reports_df)
    
    # 4. Generate Outputs
    generate_outputs(signal_df, strong_signals)
    
    print("\nPipeline completed successfully.")

if __name__ == "__main__":
    main()
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
        
        # Deduplicate reports and apply Severity Weighting (0-1 scale)
        if rid and rid not in seen_reports:
            # Severity Weights: Death=1.0, Life-Threatening=0.8, Hospitalization/Disability=0.6, Other=0.4, Unspecified Serious=0.3
            severity = 0
            if entry.get("seriousnessdeath") == "1": severity = 1.0
            elif entry.get("seriousnesslifethreatening") == "1": severity = 0.8
            elif entry.get("seriousnesshospitalization") == "1": severity = 0.6
            elif entry.get("seriousnessdisabling") == "1": severity = 0.6
            elif entry.get("seriousnessother") == "1": severity = 0.4
            elif entry.get("serious") == "1": severity = 0.3
            
            reports.append({
                "report_id": rid,
                "serious": severity,
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
    
    # ML Logic - Perfectly as per pharmacovigilance_dashboard.py
    merged = df_reactions.merge(df_reports, on="report_id")
    reaction_counts = merged["reaction_name"].value_counts().to_dict()
    merged["reaction_freq"] = merged["reaction_name"].map(reaction_counts)
    
    X = merged[["reaction_freq"]]
    y = merged["serious"]
    
    accuracy = "0.00%"
    importance = 0.0
    if len(merged) > 10:
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestClassifier()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            accuracy = f"{acc:.2%}"
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
