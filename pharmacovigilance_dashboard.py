import streamlit as st
import requests
import pandas as pd
import time
import matplotlib.pyplot as plt
import plotly.express as px
import hashlib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# AUTH SYSTEM (Persistent with Roles)
# -----------------------------
USER_DB_FILE = "users.json"

def load_users():
    if os.path.exists(USER_DB_FILE):
        try:
            with open(USER_DB_FILE, "r") as f:
                return json.load(f)
        except:
            pass
    # Default roles for new installations
    return {
        "admin": {"password": hashlib.sha256("admin123".encode()).hexdigest(), "role": "admin"},
        "analyst": {"password": hashlib.sha256("analyst123".encode()).hexdigest(), "role": "analyst"}
    }

def signup_user(id_input, password, role="analyst"):
    users = load_users()
    if id_input in users:
        return False, "User already exists!"
    users[id_input] = {
        "password": hashlib.sha256(password.encode()).hexdigest(),
        "role": role
    }
    with open(USER_DB_FILE, "w") as f:
        json.dump(users, f)
    return True, "Account created successfully!"

def check_login(id_input, password):
    users = load_users()
    user_data = users.get(id_input)
    if not user_data: return None
    
    hashed = hashlib.sha256(password.encode()).hexdigest()
    # Support for old flat JSON format
    if isinstance(user_data, str):
        if hashed == user_data: return "admin"
        return None
    
    if hashed == user_data.get("password"):
        return user_data.get("role", "analyst")
    return None

def login_ui():
    st.container().markdown("<div style='height: 50px;'></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style='background-color: #0d1117; padding: 20px; border-radius: 20px; border: 1px solid #a855f7; text-align: center;'>
                <h1 style='color: white; margin: 0;'>🛡️ Clinical Gatekeeper</h1>
                <p style='color: #94a3b8; font-size: 0.9em;'>Secure access to medical intelligence</p>
            </div>
            """, unsafe_allow_html=True)
        
        tab_login, tab_signup = st.tabs(["🔑 Login", "📝 Sign Up"])
        
        with tab_login:
            login_id = st.text_input("Username / Email / Mobile", key="login_id")
            login_pass = st.text_input("Password", type="password", key="login_pass")
            if st.button("Access Dashboard", use_container_width=True):
                role = check_login(login_id, login_pass)
                if role:
                    st.session_state["logged_in"] = True
                    st.session_state["current_user"] = login_id
                    st.session_state["role"] = role
                    st.rerun()
                else:
                    st.error("Invalid credentials. Please check your details or Sign Up.")
        
        with tab_signup:
            st.info("Register to access the Intelligence System")
            new_id = st.text_input("Choose Username / Email / Mobile", key="signup_id")
            new_pass = st.text_input("Create Password", type="password", key="signup_pass")
            confirm_pass = st.text_input("Confirm Password", type="password", key="signup_confirm")
            reg_role = st.selectbox("Select Role", ["analyst", "admin"])
            
            if st.button("Create Account", use_container_width=True):
                if not new_id or not new_pass:
                    st.warning("Please fill all fields")
                elif new_pass != confirm_pass:
                    st.error("Passwords do not match")
                else:
                    success, msg = signup_user(new_id, new_pass, reg_role)
                    if success:
                        st.success(msg)
                        st.info("You can now switch to the Login tab.")
                    else:
                        st.error(msg)
    st.stop()

# Initialize session
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# Block app if not logged in
if not st.session_state["logged_in"]:
    login_ui()

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = "YutJn0riht9IwHkUicoPXTqsZQXX4dZfpvxdcScE"
BASE_URL = "https://api.fda.gov/drug/event.json"
LIMIT = 500

st.set_page_config(page_title="Adverse Event Intelligence Dashboard", layout="wide")

# Custom CSS for premium look
st.markdown("""
    <style>
    .main {
        background-color: #05070a;
        color: #f8fafc;
    }
    .stButton>button {
        background-color: #a855f7;
        color: white;
        border-radius: 12px;
        font-weight: bold;
        border: none;
        padding: 10px 20px;
    }
    .stTextInput>div>div>input {
        background-color: #0d1117;
        color: white;
        border: 1px solid #30363d;
    }
    .metric-card {
        background-color: #0d1117;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #30363d;
    }
    .user-badge {
        background: #a855f7;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.7em;
        font-weight: bold;
        vertical-align: middle;
        margin-left: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# -----------------------------
# FETCH DATA
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_data(drug_name):
    all_results = []
    skip = 0

    while len(all_results) < 2000:
        params = {
            "api_key": API_KEY,
            "search": f"patient.drug.medicinalproduct:{drug_name}",
            "limit": LIMIT,
            "skip": skip
        }

        try:
            r = requests.get(BASE_URL, params=params)
            r.raise_for_status()
            data = r.json()

            results = data.get("results", [])
            if not results:
                break

            all_results.extend(results)
            skip += LIMIT
            time.sleep(0.5)

        except:
            break

    return all_results

# -----------------------------
# PROCESS DATA
# -----------------------------
def process_data(raw):
    reports = []
    reactions = []

    for entry in raw:
        report_id = entry.get("safetyreportid")
        date = entry.get("receiptdate")
        serious = int(entry.get("serious", 0))
        country = entry.get("primarysource", {}).get("reportercountry")

        reports.append({
            "report_id": report_id,
            "date": date,
            "serious": serious,
            "country": country
        })

        for r in entry.get("patient", {}).get("reaction", []):
            name = r.get("reactionmeddrapt")
            if name:
                reactions.append({
                    "report_id": report_id,
                    "reaction_name": name.upper()
                })

    return pd.DataFrame(reports), pd.DataFrame(reactions)

# -----------------------------
# BASELINE FOR PRR
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_baseline():
    all_results = []
    skip = 0

    while len(all_results) < 3000:
        params = {
            "api_key": API_KEY,
            "search": "_exists_:patient.reaction",
            "limit": LIMIT,
            "skip": skip
        }

        try:
            r = requests.get(BASE_URL, params=params)
            r.raise_for_status()
            data = r.json()

            results = data.get("results", [])
            if not results:
                break

            all_results.extend(results)
            skip += LIMIT
            time.sleep(0.5)

        except:
            break

    return all_results

def extract_reactions(raw):
    rows = []
    for entry in raw:
        for r in entry.get("patient", {}).get("reaction", []):
            name = r.get("reactionmeddrapt")
            if name:
                rows.append(name.upper())
    return pd.Series(rows)

# -----------------------------
# ML MODEL
# -----------------------------
def train_ml_model(reports_df, reactions_df):
    # Merge
    df = reactions_df.merge(reports_df, on="report_id")

    # Feature: reaction frequency
    reaction_counts = df["reaction_name"].value_counts().to_dict()
    df["reaction_freq"] = df["reaction_name"].map(reaction_counts)

    # Target
    df["serious"] = df["serious"].astype(int)

    # Features
    X = df[["reaction_freq"]]
    y = df["serious"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    return model, acc

# -----------------------------
# UI
# -----------------------------
col_header, col_logout = st.columns([8, 2])
with col_header:
    role_display = st.session_state["role"].upper()
    st.markdown(f"<h1>💊 AE Intelligence Dashboard <span class='user-badge'>{role_display}</span></h1>", unsafe_allow_html=True)
with col_logout:
    st.write(f"Logged in: **{st.session_state['current_user']}**")
    if st.button("Logout"):
        st.session_state["logged_in"] = False
        st.rerun()

st.markdown("---")

# Sidebar Filters
st.sidebar.header("🔍 Controls & Filters")

drug = st.sidebar.text_input("Search Drug", value="aspirin")
run = st.sidebar.button("Run Intelligence Pipeline")

if drug:
    with st.spinner(f"Analyzing data for {drug.upper()}..."):
        raw = fetch_data(drug)
        if not raw:
            st.error(f"No records found for {drug}. Please check the drug name.")
        else:
            reports_df, reactions_df = process_data(raw)

            # -----------------------------
            # FILTERS
            # -----------------------------
            countries = reports_df["country"].dropna().unique()
            selected_country = st.sidebar.selectbox("Filter by Country", ["All"] + sorted(list(countries)))

            seriousness = st.sidebar.selectbox("Filter by Seriousness", ["All", "Serious", "Non-serious"])

            filtered_reports = reports_df.copy()
            if selected_country != "All":
                filtered_reports = filtered_reports[filtered_reports["country"] == selected_country]

            if seriousness == "Serious":
                filtered_reports = filtered_reports[filtered_reports["serious"] == 1]
            elif seriousness == "Non-serious":
                filtered_reports = filtered_reports[filtered_reports["serious"] == 0]

            filtered_reactions = reactions_df[reactions_df["report_id"].isin(filtered_reports["report_id"])]

            # -----------------------------
            # METRICS
            # -----------------------------
            total_reports = len(filtered_reports)
            serious_cases = filtered_reports["serious"].sum()

            m1, m2, m3 = st.columns(3)
            with m1: st.metric("Total Reports", f"{total_reports:,}")
            with m2: st.metric("Serious Cases", f"{serious_cases:,}")
            with m3: st.metric("Serious Ratio", f"{serious_cases/total_reports:.2%}" if total_reports else "0%")

            # -----------------------------
            # LAYOUT: REACTIONS & RISK
            # -----------------------------
            tab_list = ["🎯 Safety Signals", "📈 Trends & Anomalies"]
            if st.session_state["role"] == "admin":
                tab_list.append("🤖 AI Risk Prediction")
            tab_list.append("📋 Raw Insights")
            
            tabs = st.tabs(tab_list)

            # Assign tabs based on role
            if st.session_state["role"] == "admin":
                tab1, tab2, tab3, tab4 = tabs
            else:
                tab1, tab2, tab4 = tabs
                tab3 = None

            with tab1:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Top Reactions")
                    freq = filtered_reactions["reaction_name"].value_counts().head(15)
                    fig_freq = px.bar(x=freq.values, y=freq.index, orientation='h', 
                                      labels={'x':'Count', 'y':'Reaction'},
                                      color=freq.values, color_continuous_scale='Purples')
                    fig_freq.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_dark")
                    st.plotly_chart(fig_freq, use_container_width=True)

                with col2:
                    st.subheader("High Risk Signals (Severity Weighted)")
                    merged = filtered_reactions.merge(filtered_reports, on="report_id")
                    risk = merged.groupby("reaction_name").agg(
                        total=("reaction_name", "count"),
                        serious=("serious", "sum")
                    )
                    risk["risk_score"] = risk["serious"] / risk["total"]
                    risk = risk[risk["total"] > 2].sort_values(by="risk_score", ascending=False).head(15)
                    
                    fig_risk = px.bar(risk, x="risk_score", y=risk.index, orientation='h',
                                      color="risk_score", color_continuous_scale='Reds',
                                      labels={'risk_score':'Risk Score', 'reaction_name':'Reaction'})
                    fig_risk.update_layout(yaxis={'categoryorder':'total ascending'}, template="plotly_dark")
                    st.plotly_chart(fig_risk, use_container_width=True)

                st.subheader("PRR Signal Detection (vs Global Baseline)")
                baseline_raw = fetch_baseline()
                baseline_series = extract_reactions(baseline_raw)
                baseline_counts = baseline_series.value_counts()
                drug_counts = filtered_reactions["reaction_name"].value_counts()

                total_baseline = len(baseline_series)
                total_drug = len(filtered_reactions)

                prr_list = []
                for reaction in drug_counts.index:
                    A = drug_counts[reaction]
                    B = total_drug - A
                    C = baseline_counts.get(reaction, 0)
                    D = total_baseline - C
                    if C == 0: continue
                    prr = (A/(A+B)) / (C/(C+D))
                    prr_list.append({"reaction": reaction, "PRR": round(prr, 2), "count": A})

                prr_df = pd.DataFrame(prr_list).sort_values(by="PRR", ascending=False).head(15)
                st.dataframe(prr_df, use_container_width=True)
                
                # 📥 EXPORT SECTION
                st.markdown("---")
                st.subheader("📥 Export Intelligence")
                c1, c2 = st.columns(2)
                with c1:
                    st.download_button(label="Download Top Reactions (CSV)", 
                                       data=freq.to_csv().encode("utf-8"), 
                                       file_name=f"{drug}_reactions.csv", mime="text/csv")
                with c2:
                    st.download_button(label="Download PRR Signals (CSV)", 
                                       data=prr_df.to_csv().encode("utf-8"), 
                                       file_name=f"{drug}_signals.csv", mime="text/csv")

            with tab2:
                st.subheader("Adverse Event Volume Trends")
                filtered_reports["date"] = pd.to_datetime(filtered_reports["date"], errors="coerce")
                filtered_reports["month"] = filtered_reports["date"].dt.to_period("M")
                trend = filtered_reports.groupby("month").size()
                trend.index = trend.index.astype(str)

                if not trend.empty:
                    mean = trend.mean()
                    std = trend.std() if not pd.isna(trend.std()) else 0
                    threshold = mean + 2 * std
                    anomalies = trend[trend > threshold]

                    fig_trend = px.line(x=trend.index, y=trend.values, markers=True, title="Monthly Trend")
                    if not anomalies.empty:
                        fig_trend.add_scatter(x=anomalies.index, y=anomalies.values, mode='markers', 
                                              marker=dict(color='red', size=12), name='Anomaly')
                    
                    fig_trend.update_layout(template="plotly_dark", xaxis_title="Month", yaxis_title="Reports")
                    st.plotly_chart(fig_trend, use_container_width=True)
                    
                    if not anomalies.empty:
                        st.error(f"⚠️ Spikes detected in: {', '.join(anomalies.index.tolist())}")
                else:
                    st.info("Insufficient date data for trend analysis.")

            if tab3:
                with tab3:
                    st.subheader("🤖 AI Risk Prediction")
                    
                    if len(filtered_reports) > 50:
                        model, acc = train_ml_model(filtered_reports, filtered_reactions)
                        
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            st.metric("Model Prediction Accuracy", f"{acc:.2%}")
                            st.write("This model uses reaction frequency patterns to predict serious outcomes.")
                            
                            # 🔍 EXPLAINABILITY
                            st.markdown("---")
                            st.subheader("🔍 Model Explainability")
                            importance = model.feature_importances_[0]
                            st.write(f"**Reaction Frequency Importance:** `{importance:.3f}`")
                            st.info("Higher reaction frequency in this dataset strongly correlates with 'Serious' classification probability.")
                        
                        with c2:
                            reaction_input = st.selectbox(
                                "Select Reaction to Predict Severity",
                                sorted(filtered_reactions["reaction_name"].unique())
                            )
                            
                            if st.button("Predict Clinical Risk"):
                                freq_val = filtered_reactions["reaction_name"].value_counts().get(reaction_input, 1)
                                pred = model.predict([[freq_val]])[0]
                                
                                if pred == 1:
                                    st.error(f"⚠️ PREDICTION: High Clinical Risk (Serious) for {reaction_input}")
                                else:
                                    st.success(f"✅ PREDICTION: Low Clinical Risk (Non-serious) for {reaction_input}")
                    else:
                        st.warning("Insufficient filtered data to train the ML model.")

            with tab4:
                st.subheader("Raw Reaction Intelligence")
                st.dataframe(filtered_reactions.merge(filtered_reports, on="report_id"), use_container_width=True)

else:
    st.info("👈 Enter a drug name in the sidebar to begin production-level safety analysis.")
    st.image("https://images.unsplash.com/photo-1584036561566-baf8f5f1b144?auto=format&fit=crop&q=80&w=2000", caption="Precision Pharmacovigilance Engine")
