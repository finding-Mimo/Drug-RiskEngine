import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

class PatientRiskEngine:
    def __init__(self, baseline_window_days=3):
        self.baseline_window_days = baseline_window_days
        # Simple ML model mock
        self.scaler = StandardScaler()
        self.ml_model = LogisticRegression()
        self._is_model_trained = False
        
    def load_wearable_data(self, csv_path):
        """Ingest and preprocess wearable time-series data."""
        df = pd.read_csv(csv_path, parse_dates=['date'])
        df = df.sort_values('date').reset_index(drop=True)
        # Handle missing data using forward fill for critical metrics
        df[['bp_systolic_mmhg', 'bp_diastolic_mmhg']] = df[['bp_systolic_mmhg', 'bp_diastolic_mmhg']].ffill()
        return df

    def load_pharmacy_data(self, json_path):
        """Ingest pharmacy JSON data."""
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data['dispense_records']

    def calculate_wearable_features(self, df):
        """Feature engineering for physiological signals."""
        if df.empty:
            return df
            
        # 1. Establish Patient Baseline (first N days)
        baseline_df = df.head(self.baseline_window_days)
        baseline_weight = baseline_df['weight_kg'].mean()
        baseline_hr = baseline_df['resting_hr_bpm'].mean()
        baseline_steps = baseline_df['steps'].mean()
        baseline_spo2 = baseline_df['spo2_pct'].mean()
        
        # 2. Compute Rolling Deviations (Δ values)
        # Using 3-day rolling averages to smooth daily noise
        df['rolling_weight_3d'] = df['weight_kg'].rolling(window=3, min_periods=1).mean()
        df['rolling_hr_3d'] = df['resting_hr_bpm'].rolling(window=3, min_periods=1).mean()
        df['rolling_steps_3d'] = df['steps'].rolling(window=3, min_periods=1).mean()
        
        # Absolute deviations from baseline
        df['delta_weight_kg'] = df['rolling_weight_3d'] - baseline_weight
        df['delta_hr_bpm'] = df['rolling_hr_3d'] - baseline_hr
        
        # Relative deviations from baseline
        df['pct_change_steps'] = ((df['rolling_steps_3d'] - baseline_steps) / baseline_steps) * 100
        df['delta_spo2'] = df['spo2_pct'] - baseline_spo2
        
        return df

    def calculate_pharmacy_features(self, rx_records, current_date):
        """Feature engineering for medication adherence and risks."""
        adherence_issues = []
        med_risk_score = 0
        
        current_date_obj = pd.to_datetime(current_date)
        
        for rx in rx_records:
            # Check for generic substitution notes implying coverage issues
            notes = rx.get('notes', '').lower()
            if 'substituted' in notes or 'not covered' in notes or 'voicemail' in notes:
                med_risk_score += 0.3
                adherence_issues.append(f"Potential therapy mismatch/coverage issue: {rx['drug_name']}")
                
            # Check for adherence gaps
            if 'date_pickup' in rx and 'days_supply' in rx:
                pickup_date = pd.to_datetime(rx['date_pickup'])
                days_supply = rx['days_supply']
                expected_empty_date = pickup_date + pd.Timedelta(days=days_supply)
                
                # If we have refill history, check the latest fill
                if 'refill_history' in rx and len(rx['refill_history']) > 0:
                    latest_refill = rx['refill_history'][-1]
                    pickup_date = pd.to_datetime(latest_refill.get('date_pickup', latest_refill['date_filled']))
                    days_supply = latest_refill['days_supply']
                    expected_empty_date = pickup_date + pd.Timedelta(days=days_supply)
                
                days_since_empty = (current_date_obj - expected_empty_date).days
                if days_since_empty > 0:
                    med_risk_score += 0.5
                    adherence_issues.append(f"Medication gap detected for {rx['drug_name']} ({days_since_empty} days)")
                    
        return min(1.0, med_risk_score), adherence_issues

    def rule_based_risk_score(self, row, med_risk_score):
        """Hybrid Rule-Based Logic for Readmission Risk."""
        physio_risk = 0
        drivers = []
        
        # Rule 1: Fluid Retention (CHF specific but relative)
        if row['delta_weight_kg'] > 2.0:
            physio_risk += 0.4
            drivers.append(f"Weight increased {row['delta_weight_kg']:.1f}kg above baseline")
            
        # Rule 2: Cardiovascular Stress
        if row['delta_hr_bpm'] > 15:
            physio_risk += 0.3
            drivers.append(f"Resting HR increased {row['delta_hr_bpm']:.0f} bpm above baseline")
            
        # Rule 3: Functional Decline
        if row['pct_change_steps'] < -50:
            physio_risk += 0.2
            drivers.append(f"Activity dropped {abs(row['pct_change_steps']):.0f}% below baseline")
            
        # Rule 4: Hypoxia
        if row['spo2_pct'] < 92 and row['delta_spo2'] < -2:
            physio_risk += 0.3
            drivers.append(f"SpO2 dropped to {row['spo2_pct']}% (from baseline)")
            
        # Composite Triggers (Synergistic risk)
        if row['delta_weight_kg'] > 1.5 and row['pct_change_steps'] < -30:
            physio_risk += 0.2
            drivers.append("Composite: Weight increasing while activity decreasing")
            
        # Total Score (capped at 1.0)
        total_risk = min(1.0, (0.7 * physio_risk) + (0.3 * med_risk_score))
        
        risk_level = "LOW"
        if total_risk > 0.7:
            risk_level = "CRITICAL"
        elif total_risk > 0.4:
            risk_level = "HIGH"
        elif total_risk > 0.2:
            risk_level = "WARNING"
            
        return total_risk, risk_level, drivers

    def train_ml_model(self, historical_features, labels):
        """Train optional ML layer on historical population data."""
        # This is a placeholder for population-level ML training
        X = self.scaler.fit_transform(historical_features)
        self.ml_model.fit(X, labels)
        self._is_model_trained = True

    def process_patient(self, wearable_csv, pharmacy_json, target_date_str=None):
        """Main pipeline to process a patient's data and generate risk score."""
        wearable_df = self.load_wearable_data(wearable_csv)
        wearable_df = self.calculate_wearable_features(wearable_df)
        
        pharmacy_records = self.load_pharmacy_data(pharmacy_json)
        
        if target_date_str:
            target_date = pd.to_datetime(target_date_str)
            daily_data = wearable_df[wearable_df['date'] <= target_date].iloc[-1]
            eval_date = target_date
        else:
            daily_data = wearable_df.iloc[-1]
            eval_date = daily_data['date']
            
        med_risk_score, med_issues = self.calculate_pharmacy_features(pharmacy_records, eval_date)
        
        score, level, drivers = self.rule_based_risk_score(daily_data, med_risk_score)
        
        all_drivers = drivers + med_issues
        
        return {
            "patient_id": "Harmon, Robert",
            "evaluation_date": eval_date.strftime('%Y-%m-%d'),
            "risk_score": round(score, 2),
            "risk_level": level,
            "drivers": all_drivers,
            "recommended_action": "Immediate clinical review and patient outreach." if level in ["HIGH", "CRITICAL"] else "Monitor"
        }

if __name__ == "__main__":
    engine = PatientRiskEngine(baseline_window_days=3)
    
    # Process up to early April when deterioration is evident
    result = engine.process_patient(
        wearable_csv='wearable_export_bob_harmon.csv',
        pharmacy_json='pharmacy_feed_harmon.json',
        target_date_str='2024-04-05'
    )
    
    print(json.dumps(result, indent=2))
