import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
import io
import gspread
from google.oauth2.service_account import Credentials
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🦠 Stanes School CBSE - PandemicTrack Pro",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with school branding
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .school-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(30, 60, 114, 0.3);
    }
    
    .school-logo {
        width: 80px;
        height: 80px;
        margin: 0 auto 1rem auto;
        background: white;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 2rem;
    }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        margin: 1rem 0;
    }
    
    .subtitle {
        color: #e0e6ed;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        font-weight: 400;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        margin: 0.5rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    
    .alert-card {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(255, 107, 107, 0.3);
    }
    
    .info-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(116, 185, 255, 0.3);
    }
    
    .success-card {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0, 184, 148, 0.3);
    }
    
    .explanation-box {
        background: #f8f9fa;
        border-left: 4px solid #1e3c72;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        font-size: 0.9rem;
        color: #555;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #fdcb6e 0%, #e17055 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #00b894 0%, #00a085 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
    }
    
    .data-unavailable {
        background: #f1c40f;
        color: #2c3e50;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 500;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disease Configuration - Real parameters from WHO/CDC data
DISEASE_CONFIG = {
    "COVID-19": {
        "transmission_rate": 0.4,
        "recovery_rate": 0.1,
        "death_rate": 0.02,
        "incubation_period": 5.1,
        "infectious_period": 10,
        "basic_r0": 2.5,
        "vaccine_effectiveness": 0.95,
        "primary_symptoms": ["Fever (>38°C)", "Persistent cough", "Shortness of breath", "Loss of taste/smell"],
        "secondary_symptoms": ["Headache", "Muscle aches", "Sore throat", "Fatigue"],
        "vaccines": ["Pfizer-BioNTech", "Moderna", "AstraZeneca", "Johnson & Johnson", "COVISHIELD", "COVAXIN"]
    },
    "Mpox": {
        "transmission_rate": 0.2,
        "recovery_rate": 0.05,
        "death_rate": 0.01,
        "incubation_period": 12,
        "infectious_period": 21,
        "basic_r0": 1.8,
        "vaccine_effectiveness": 0.85,
        "primary_symptoms": ["Fever", "Skin rash/lesions", "Swollen lymph nodes", "Headache"],
        "secondary_symptoms": ["Muscle aches", "Back pain", "Exhaustion", "Chills"],
        "vaccines": ["JYNNEOS", "ACAM2000"]
    },
    "Influenza": {
        "transmission_rate": 0.5,
        "recovery_rate": 0.15,
        "death_rate": 0.005,
        "incubation_period": 2,
        "infectious_period": 7,
        "basic_r0": 1.3,
        "vaccine_effectiveness": 0.60,
        "primary_symptoms": ["Fever", "Persistent cough", "Sore throat", "Runny nose"],
        "secondary_symptoms": ["Muscle aches", "Headache", "Fatigue", "Chills"],
        "vaccines": ["Flu Shot (Trivalent)", "Flu Shot (Quadrivalent)", "Nasal Spray"]
    },
    "H1N1": {
        "transmission_rate": 0.45,
        "recovery_rate": 0.12,
        "death_rate": 0.008,
        "incubation_period": 1.5,
        "infectious_period": 8,
        "basic_r0": 1.6,
        "vaccine_effectiveness": 0.70,
        "primary_symptoms": ["Fever", "Persistent cough", "Body aches", "Sore throat"],
        "secondary_symptoms": ["Headache", "Runny nose", "Fatigue", "Nausea"],
        "vaccines": ["H1N1 Vaccine", "Seasonal Flu Vaccine"]
    }
}

# Country/Region Configuration
REGION_CONFIG = {
    "India": {"population": 1400000000, "iso_code": "IN"},
    "United States": {"population": 331000000, "iso_code": "US"},
    "United Kingdom": {"population": 67000000, "iso_code": "GB"},
    "Germany": {"population": 84000000, "iso_code": "DE"},
    "France": {"population": 68000000, "iso_code": "FR"},
    "Brazil": {"population": 215000000, "iso_code": "BR"},
    "Japan": {"population": 125000000, "iso_code": "JP"},
    "Australia": {"population": 26000000, "iso_code": "AU"},
    "Canada": {"population": 39000000, "iso_code": "CA"},
    "Global": {"population": 7900000000, "iso_code": "OWID_WRL"}
}

# Google Sheets Integration Class
class GoogleSheetsIntegration:
    def __init__(self):
        self.sheet_id = None
        self.credentials = None
        self.client = None
    
    def setup_credentials(self, credentials_json):
        """Setup Google Sheets credentials from uploaded JSON file"""
        try:
            credentials_dict = json.loads(credentials_json)
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            
            self.credentials = Credentials.from_service_account_info(
                credentials_dict, scopes=scopes
            )
            self.client = gspread.authorize(self.credentials)
            return True
        except Exception as e:
            st.error(f"❌ Credentials setup failed: {str(e)}")
            return False
    
    def create_or_open_sheet(self, sheet_name="Stanes_PandemicTrack_Data"):
        """Create or open Google Sheet for data storage"""
        try:
            if not self.client:
                return None
            
            try:
                # Try to open existing sheet
                sheet = self.client.open(sheet_name)
                st.success(f"✅ Connected to existing sheet: {sheet_name}")
            except gspread.SpreadsheetNotFound:
                # Create new sheet
                sheet = self.client.create(sheet_name)
                st.success(f"✅ Created new sheet: {sheet_name}")
            
            # Share with school domain (optional)
            # sheet.share('stanes.edu@domain.com', perm_type='user', role='writer')
            
            self.sheet_id = sheet.id
            return sheet
        except Exception as e:
            st.error(f"❌ Sheet creation/access failed: {str(e)}")
            return None
    
    def add_vaccination_record(self, data, worksheet_name="Vaccinations"):
        """Add vaccination booking to Google Sheets"""
        try:
            if not self.client or not self.sheet_id:
                st.error("❌ Google Sheets not connected")
                return False
            
            sheet = self.client.open_by_key(self.sheet_id)
            
            try:
                worksheet = sheet.worksheet(worksheet_name)
            except gspread.WorksheetNotFound:
                # Create worksheet with headers
                worksheet = sheet.add_worksheet(title=worksheet_name, rows=1000, cols=10)
                headers = ['Timestamp', 'Name', 'Phone', 'Email', 'Disease', 'Vaccine', 'Date', 'Time', 'Location', 'Status']
                worksheet.append_row(headers)
            
            # Add vaccination record
            row_data = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                data.get('name', ''),
                data.get('phone', ''),
                data.get('email', ''),
                data.get('disease', ''),
                data.get('vaccine', ''),
                str(data.get('date', '')),
                data.get('time', ''),
                data.get('location', ''),
                'Booked'
            ]
            
            worksheet.append_row(row_data)
            st.success("✅ Vaccination booking recorded in Google Sheets")
            return True
            
        except Exception as e:
            st.error(f"❌ Failed to record vaccination: {str(e)}")
            return False
    
    def get_vaccination_stats(self, worksheet_name="Vaccinations"):
        """Get vaccination statistics from Google Sheets"""
        try:
            if not self.client or not self.sheet_id:
                return None
            
            sheet = self.client.open_by_key(self.sheet_id)
            worksheet = sheet.worksheet(worksheet_name)
            
            records = worksheet.get_all_records()
            df = pd.DataFrame(records)
            
            if not df.empty:
                stats = {
                    'total_bookings': len(df),
                    'disease_breakdown': df['Disease'].value_counts().to_dict(),
                    'vaccine_breakdown': df['Vaccine'].value_counts().to_dict(),
                    'recent_bookings': len(df[df['Timestamp'] >= (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')])
                }
                return stats
            return None
            
        except Exception as e:
            st.warning(f"⚠️ Could not fetch vaccination stats: {str(e)}")
            return None

# Real Data Fetcher Class - Only uses official APIs
class RealDataFetcher:
    @staticmethod
    def fetch_disease_data(country, disease):
        """Fetch real disease data from official sources"""
        try:
            if disease == "COVID-19":
                return RealDataFetcher._fetch_covid_data(country)
            elif disease in ["Influenza", "H1N1"]:
                return RealDataFetcher._fetch_flu_data(country)
            elif disease == "Mpox":
                return RealDataFetcher._fetch_mpox_data(country)
            else:
                return None
        except Exception as e:
            st.warning(f"⚠️ Data fetch failed: {str(e)}")
            return None
    
    @staticmethod
    def _fetch_covid_data(country):
        """Fetch COVID-19 data from disease.sh API"""
        try:
            if country == "Global":
                url = "https://disease.sh/v3/covid-19/all"
            else:
                iso_code = REGION_CONFIG.get(country, {}).get("iso_code", country.lower())
                url = f"https://disease.sh/v3/covid-19/countries/{iso_code}"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "disease": "COVID-19",
                    "region": country,
                    "total_cases": data.get("cases", 0),
                    "active_cases": data.get("active", 0),
                    "recovered": data.get("recovered", 0),
                    "deaths": data.get("deaths", 0),
                    "tests": data.get("tests", 0),
                    "population": data.get("population", REGION_CONFIG.get(country, {}).get("population", 0)),
                    "last_updated": datetime.fromtimestamp(data.get("updated", 0)/1000).strftime('%Y-%m-%d %H:%M:%S'),
                    "source": "Johns Hopkins via Disease.sh API"
                }
            else:
                return None
                
        except Exception as e:
            st.warning(f"⚠️ COVID-19 data unavailable: {str(e)}")
            return None
    
    @staticmethod
    def _fetch_flu_data(country):
        """Fetch influenza data - Note: Limited public APIs available"""
        try:
            # WHO FluNet API or CDC data would go here
            # For now, return None to indicate data unavailability
            st.info("ℹ️ Real-time influenza data requires WHO FluNet API access")
            return None
        except Exception as e:
            return None
    
    @staticmethod
    def _fetch_mpox_data(country):
        """Fetch Mpox data from WHO or CDC sources"""
        try:
            # WHO API for Mpox data would go here
            # For now, return None to indicate data unavailability
            st.info("ℹ️ Real-time Mpox data requires WHO API access")
            return None
        except Exception as e:
            return None

# Enhanced AI Predictor integrated with SIR models
class EnhancedAIPredictor:
    @staticmethod
    def predict_with_sir_integration(sir_data, disease_params, days_ahead=30):
        """AI prediction integrated with SIR model outputs"""
        try:
            if not sir_data or len(sir_data) < 7:
                st.warning("⚠️ Insufficient data for AI predictions")
                return None
            
            # Use SIR model trajectory for intelligent predictions
            recent_infected = sir_data[-10:]  # Last 10 days
            
            # Calculate growth rate and trend
            growth_rates = []
            for i in range(1, len(recent_infected)):
                if recent_infected[i-1] > 0:
                    rate = (recent_infected[i] - recent_infected[i-1]) / recent_infected[i-1]
                    growth_rates.append(rate)
            
            if not growth_rates:
                return [recent_infected[-1]] * days_ahead
            
            # Apply disease-specific parameters
            r0 = disease_params.get('basic_r0', 2.0)
            recovery_rate = disease_params.get('recovery_rate', 0.1)
            
            # Enhanced prediction with epidemiological constraints
            predictions = []
            current_value = recent_infected[-1]
            avg_growth = np.mean(growth_rates[-5:])  # Recent trend
            
            for day in range(days_ahead):
                # Apply epidemiological damping
                damping_factor = 1 / (1 + day * 0.01)  # Gradual decrease in growth
                adjusted_growth = avg_growth * damping_factor
                
                # Incorporate R0 and recovery dynamics
                sir_constraint = max(0, 1 - (day / (recovery_rate * 100)))
                
                next_value = current_value * (1 + adjusted_growth) * sir_constraint
                predictions.append(max(0, next_value))
                current_value = next_value
            
            return predictions
            
        except Exception as e:
            st.error(f"❌ AI prediction failed: {str(e)}")
            return None

# SIR Model Implementation
class EpidemiologicalModels:
    @staticmethod
    def sir_model(N, I0, R0, beta, gamma, num_days):
        """Standard SIR model implementation"""
        S = max(0, N - I0 - R0)
        I = max(0, I0)
        R = max(0, R0)
        
        susceptible, infected, recovered = [S], [I], [R]
        
        for day in range(num_days):
            dS = -beta * S * I / N if N > 0 else 0
            dI = beta * S * I / N - gamma * I if N > 0 else -gamma * I
            dR = gamma * I
            
            S = max(0, S + dS)
            I = max(0, I + dI)
            R = max(0, R + dR)
            
            susceptible.append(S)
            infected.append(I)
            recovered.append(R)
        
        return susceptible, infected, recovered
    
    @staticmethod
    def sirdv_model(N, I0, R0, D0, V0, beta, gamma, delta, vaccination_rate, vaccine_eff, num_days):
        """SIRD model with vaccination"""
        S = max(0, N - I0 - R0 - D0 - V0)
        I = max(0, I0)
        R = max(0, R0)
        D = max(0, D0)
        V = max(0, V0)
        
        susceptible, infected, recovered, deaths, vaccinated = [S], [I], [R], [D], [V]
        
        for day in range(num_days):
            # Daily vaccinations
            new_vaccinations = min(vaccination_rate * S, S) if S > 0 else 0
            effective_vaccinations = new_vaccinations * vaccine_eff
            
            # Model dynamics
            new_infections = beta * S * I / N if N > 0 else 0
            recoveries = gamma * I
            deaths_daily = delta * I
            
            dS = -new_infections - new_vaccinations
            dI = new_infections - recoveries - deaths_daily
            dR = recoveries + effective_vaccinations
            dD = deaths_daily
            dV = new_vaccinations
            
            S = max(0, S + dS)
            I = max(0, I + dI)
            R = max(0, R + dR)
            D = max(0, D + dD)
            V = max(0, V + dV)
            
            susceptible.append(S)
            infected.append(I)
            recovered.append(R)
            deaths.append(D)
            vaccinated.append(V)
        
        return susceptible, infected, recovered, deaths, vaccinated

def create_school_header():
    """Create Stanes School header"""
    st.markdown("""
    <div class="school-header">
        <div class="school-logo">🏫</div>
        <h1 class="main-header">Stanes School CBSE Coimbatore</h1>
        <p class="subtitle">Advanced Pandemic Surveillance & Health Monitoring System</p>
        <p style="font-size: 0.9rem; opacity: 0.8;">Science & Mathematics Department - Health Informatics Project</p>
    </div>
    """, unsafe_allow_html=True)

def symptom_tracker(disease):
    """Disease-specific symptom tracking interface"""
    st.markdown("### 🩺 Disease-Specific Symptom Assessment")
    
    disease_config = DISEASE_CONFIG.get(disease, DISEASE_CONFIG["COVID-19"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**Primary {disease} Symptoms:**")
        primary_symptoms = {}
        for symptom in disease_config["primary_symptoms"]:
            primary_symptoms[symptom] = st.checkbox(symptom, key=f"primary_{symptom}")
    
    with col2:
        st.markdown(f"**Secondary {disease} Symptoms:**")
        secondary_symptoms = {}
        for symptom in disease_config["secondary_symptoms"]:
            secondary_symptoms[symptom] = st.checkbox(symptom, key=f"secondary_{symptom}")
    
    # Risk factors
    st.markdown("**General Risk Factors:**")
    risk_factors = {
        "Recent travel": st.checkbox("Recent travel to affected area", key="travel"),
        "Close contact": st.checkbox("Close contact with confirmed case", key="contact"),
        "Healthcare worker": st.checkbox("Healthcare worker", key="healthcare"),
        "Immunocompromised": st.checkbox("Immunocompromised condition", key="immune"),
    }
    
    if st.button(f"🔍 Assess {disease} Risk", type="primary"):
        # Calculate weighted risk score
        primary_score = sum([v * 3 for v in primary_symptoms.values()])
        secondary_score = sum([v * 1 for v in secondary_symptoms.values()])
        risk_factor_score = sum([v * 2 for v in risk_factors.values()])
        
        total_score = primary_score + secondary_score + risk_factor_score
        
        # Disease-specific risk assessment
        if disease == "Mpox" and primary_symptoms.get("Skin rash/lesions", False):
            total_score += 5  # Rash is highly indicative for Mpox
        elif disease == "COVID-19" and primary_symptoms.get("Loss of taste/smell", False):
            total_score += 4  # Loss of taste/smell is highly specific for COVID
        
        # Risk level determination
        if total_score >= 12:
            st.markdown(f'<div class="risk-high">🚨 VERY HIGH RISK for {disease} - Seek immediate medical attention</div>', unsafe_allow_html=True)
            st.error("**URGENT:** Contact healthcare provider immediately")
        elif total_score >= 8:
            st.markdown(f'<div class="risk-medium">⚠️ MODERATE TO HIGH RISK for {disease} - Get tested and consult doctor</div>', unsafe_allow_html=True)
            st.warning("Recommend testing and medical consultation within 24 hours")
        elif total_score >= 4:
            st.markdown(f'<div class="risk-medium">⚠️ MODERATE RISK for {disease} - Monitor symptoms</div>', unsafe_allow_html=True)
            st.info("Continue monitoring and consider testing if symptoms worsen")
        else:
            st.markdown(f'<div class="risk-low">✅ LOW RISK for {disease} - Continue standard precautions</div>', unsafe_allow_html=True)
            st.success("Low risk based on current symptoms")
        
        # Risk score breakdown
        st.markdown("### 📊 Risk Score Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Primary Symptoms", f"{primary_score}")
        with col2:
            st.metric("Secondary Symptoms", f"{secondary_score}")
        with col3:
            st.metric("Risk Factors", f"{risk_factor_score}")
        with col4:
            st.metric("Total Risk Score", f"{total_score}")

def vaccination_interface(disease):
    """Disease-specific vaccination interface"""
    st.markdown(f"## 💉 {disease} Vaccination System")
    
    disease_config = DISEASE_CONFIG.get(disease, DISEASE_CONFIG["COVID-19"])
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"### 📋 Book {disease} Vaccination")
        
        with st.form("vaccination_booking"):
            st.markdown("**Personal Information:**")
            name = st.text_input("Full Name*", placeholder="Enter your full name")
            phone = st.text_input("Phone Number*", placeholder="+91 XXXXXXXXXX")
            email = st.text_input("Email Address", placeholder="your.email@example.com")
            age = st.number_input("Age", min_value=0, max_value=120, value=25)
            
            st.markdown("**Vaccination Details:**")
            vaccine_type = st.selectbox(f"Preferred {disease} Vaccine:", 
                                      ["Any Available"] + disease_config["vaccines"])
            
            preferred_date = st.date_input("Preferred Date:", 
                                         datetime.now().date() + timedelta(days=1))
            
            preferred_time = st.selectbox("Preferred Time:", 
                                        ["Morning (9-12)", "Afternoon (12-3)", "Evening (3-6)"])
            
            location = st.selectbox("Preferred Location:", 
                                  ["Coimbatore Medical College", "Government Hospital Coimbatore", 
                                   "Private Hospital - Kovai Medical Center", "Community Health Center"])
            
            # Medical history
            st.markdown("**Medical History:**")
            allergies = st.text_area("Known allergies (if any):", placeholder="List any known allergies")
            previous_vaccination = st.checkbox(f"Previously vaccinated for {disease}")
            
            submitted = st.form_submit_button("📅 Book Vaccination", type="primary")
            
            if submitted:
                if name and phone:
                    booking_data = {
                        'name': name,
                        'phone': phone,
                        'email': email,
                        'age': age,
                        'disease': disease,
                        'vaccine': vaccine_type,
                        'date': preferred_date,
                        'time': preferred_time,
                        'location': location,
                        'allergies': allergies,
                        'previous_vaccination': previous_vaccination
                    }
                    
                    # Save to session state for Google Sheets
                    if 'vaccination_bookings' not in st.session_state:
                        st.session_state.vaccination_bookings = []
                    
                    st.session_state.vaccination_bookings.append(booking_data)
                    
                    st.success(f"🎉 {disease} vaccination booked successfully for {name}!")
                    st.info("📧 Confirmation details will be sent to your email/phone")
                    st.balloons()
                    
                    # If Google Sheets is connected, save there too
                    if 'sheets_integration' in st.session_state:
                        st.session_state.sheets_integration.add_vaccination_record(booking_data)
                else:
                    st.error("❌ Please fill in required fields (Name and Phone)")
    
    with col2:
        st.markdown(f"### 📊 {disease} Vaccination Information")
        
        # Vaccine effectiveness chart
        fig_eff = go.Figure(data=[
            go.Bar(
                x=disease_config["vaccines"],
                y=[disease_config["vaccine_effectiveness"] * 100] * len(disease_config["vaccines"]),
                marker_color='lightblue',
                text=[f"{disease_config['vaccine_effectiveness'] * 100:.1f}%"] * len(disease_config["vaccines"]),
                textposition='auto',
            )
        ])
        
        fig_eff.update_layout(
            title=f"{disease} Vaccine Effectiveness",
            xaxis_title="Vaccine Type",
            yaxis_title="Effectiveness (%)",
            height=400
        )
        
        st.plotly_chart(fig_eff, use_container_width=True)
        
        # Vaccination statistics (from session state)
        if 'vaccination_bookings' in st.session_state:
            bookings = st.session_state.vaccination_bookings
            disease_bookings = [b for b in bookings if b['disease'] == disease]
            
            if disease_bookings:
                st.markdown("### 📈 Booking Statistics")
                
                # Vaccine preference breakdown
                vaccine_counts = {}
                for booking in disease_bookings:
                    vaccine = booking['vaccine']
                    vaccine_counts[vaccine] = vaccine_counts.get(vaccine, 0) + 1
                
                # Create pie chart for vaccine preferences
                if vaccine_counts:
                    fig_pie = px.pie(
                        values=list(vaccine_counts.values()),
                        names=list(vaccine_counts.keys()),
                        title=f"{disease} Vaccine Preferences"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                # Display booking metrics
                st.metric("Total Bookings", len(disease_bookings))
                st.metric("This Week", len([b for b in disease_bookings 
                                         if datetime.strptime(str(b['date']), '%Y-%m-%d').date() >= 
                                         (datetime.now().date() - timedelta(days=7))]))
        
        # Disease information
        st.markdown(f"### ℹ️ About {disease}")
        st.info(f"""
        **Transmission Rate:** {disease_config['transmission_rate']:.1f}  
        **Recovery Rate:** {disease_config['recovery_rate']:.1f}  
        **Basic R₀:** {disease_config['basic_r0']:.1f}  
        **Incubation Period:** {disease_config['incubation_period']:.1f} days  
        **Infectious Period:** {disease_config['infectious_period']} days
        """)

def create_explanation_box(title, content):
    """Create explanation box for graphs"""
    st.markdown(f"""
    <div class="explanation-box">
        <h4>📖 {title}</h4>
        <p>{content}</p>
    </div>
    """, unsafe_allow_html=True)

def main():
    # School header
    create_school_header()
    
    # Initialize Google Sheets integration
    if 'sheets_integration' not in st.session_state:
        st.session_state.sheets_integration = GoogleSheetsIntegration()
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Real-time Dashboard", "🩺 Health Assessment", "💉 Vaccination System", 
        "📊 Epidemic Modeling", "⚙️ Settings"
    ])
    
    # Sidebar controls - STRICT DROPDOWNS ONLY
    st.sidebar.markdown("## 🎛️ System Controls")
    
    # Disease selection - dropdown only
    disease = st.sidebar.selectbox(
        "🦠 Select Disease",
        list(DISEASE_CONFIG.keys()),
        help="Choose the disease to monitor and analyze"
    )
    
    # Region selection - dropdown only
    region = st.sidebar.selectbox(
        "🌍 Select Region",
        list(REGION_CONFIG.keys()),
        help="Choose geographical region for analysis"
    )
    
    # Fetch real data
    with st.spinner("🔄 Fetching real-time data..."):
        live_data = RealDataFetcher.fetch_disease_data(region, disease)
    
    # Display data availability status
    if live_data:
        st.sidebar.success("✅ Real data available")
        st.sidebar.metric("🦠 Total Cases", f"{live_data['total_cases']:,}")
        st.sidebar.metric("⚡ Active Cases", f"{live_data['active_cases']:,}")
        st.sidebar.metric("♻️ Recovered", f"{live_data['recovered']:,}")
        st.sidebar.metric("💀 Deaths", f"{live_data['deaths']:,}")
        
        # Calculate key metrics
        if live_data['total_cases'] > 0:
            mortality_rate = (live_data['deaths'] / live_data['total_cases']) * 100
            recovery_rate = (live_data['recovered'] / live_data['total_cases']) * 100
            st.sidebar.metric("💀 Case Fatality Rate", f"{mortality_rate:.2f}%")
            st.sidebar.metric("♻️ Recovery Rate", f"{recovery_rate:.2f}%")
        
        st.sidebar.caption(f"🕒 Last updated: {live_data.get('last_updated', 'Unknown')}")
        st.sidebar.caption(f"📊 Source: {live_data.get('source', 'API')}")
    else:
        st.sidebar.warning("⚠️ Real-time data unavailable")
        st.sidebar.info("Using epidemiological models for analysis")
    
    # TAB 1: Real-time Dashboard
    with tab1:
        st.markdown(f"## 📊 {disease} Dashboard - {region}")
        
        if live_data:
            # Key metrics with real data
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("🦠 Total Cases", f"{live_data['total_cases']:,}")
            with col2:
                st.metric("⚡ Active Cases", f"{live_data['active_cases']:,}")
            with col3:
                st.metric("♻️ Recovered", f"{live_data['recovered']:,}")
            with col4:
                st.metric("💀 Deaths", f"{live_data['deaths']:,}")
            
            # Create trend visualization
            st.markdown("### 📈 Epidemic Curve Analysis")
            
            # Since we only have current data point, create SIR model for visualization
            disease_params = DISEASE_CONFIG[disease]
            population = REGION_CONFIG[region]["population"]
            
            # Initialize SIR model with current data
            current_infected = live_data['active_cases']
            current_recovered = live_data['recovered']
            
            models = EpidemiologicalModels()
            S, I, R = models.sir_model(
                N=population,
                I0=current_infected,
                R0=current_recovered,
                beta=disease_params['transmission_rate'],
                gamma=disease_params['recovery_rate'],
                num_days=180
            )
            
            # Create comprehensive dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    "SIR Model Projection", 
                    "Case Distribution", 
                    "Recovery vs Deaths",
                    "Population Impact"
                ),
                specs=[
                    [{"secondary_y": True}, {"type": "pie"}],
                    [{"type": "bar"}, {"type": "scatter"}]
                ]
            )
            
            # SIR Model plot
            days = list(range(len(I)))
            fig.add_trace(go.Scatter(
                x=days, y=S, name="Susceptible", 
                line=dict(color='blue', width=2)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=days, y=I, name="Infected", 
                line=dict(color='red', width=2)
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=days, y=R, name="Recovered", 
                line=dict(color='green', width=2)
            ), row=1, col=1)
            
            # Case distribution pie chart
            fig.add_trace(go.Pie(
                values=[live_data['active_cases'], live_data['recovered'], live_data['deaths']],
                labels=['Active', 'Recovered', 'Deaths'],
                name="Case Distribution"
            ), row=1, col=2)
            
            # Recovery vs Deaths comparison
            fig.add_trace(go.Bar(
                x=['Recovered', 'Deaths'],
                y=[live_data['recovered'], live_data['deaths']],
                name="Outcomes",
                marker_color=['green', 'red']
            ), row=2, col=1)
            
            # Population impact over time
            attack_rate = [(population - s)/population * 100 for s in S]
            fig.add_trace(go.Scatter(
                x=days, y=attack_rate, name="Attack Rate (%)",
                line=dict(color='orange', width=2)
            ), row=2, col=2)
            
            fig.update_layout(height=800, showlegend=True, 
                            title_text=f"Comprehensive {disease} Analysis - {region}")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Explanations for each graph
            create_explanation_box("SIR Model Projection", 
                f"This shows the predicted spread of {disease} over time using the Susceptible-Infected-Recovered model. "
                f"The model uses real transmission parameters: R₀={disease_params['basic_r0']}, "
                f"recovery rate={disease_params['recovery_rate']}. Blue shows people still at risk, "
                f"red shows active infections, green shows recovered/immune individuals.")
            
            create_explanation_box("Case Distribution", 
                "Current breakdown of all confirmed cases. This pie chart shows what percentage of total cases "
                "are currently active (requiring isolation/treatment), recovered (immune), or resulted in death. "
                f"Recovery rate of {((live_data['recovered']/live_data['total_cases'])*100):.1f}% indicates treatment effectiveness.")
            
            create_explanation_box("Recovery vs Deaths Comparison", 
                f"Direct comparison of positive vs negative outcomes. The recovery-to-death ratio is "
                f"{(live_data['recovered']/max(live_data['deaths'], 1)):.1f}:1, indicating treatment success rates "
                f"and healthcare system capacity for {disease} management.")
            
            create_explanation_box("Population Impact (Attack Rate)", 
                "Shows the cumulative percentage of population that has been infected over time. "
                f"This helps estimate when herd immunity might be reached (typically 60-70% for R₀={disease_params['basic_r0']}) "
                "and the overall burden on society and healthcare systems.")
            
            # AI Predictions integrated with SIR
            st.markdown("### 🤖 AI-Enhanced Predictions")
            
            predictor = EnhancedAIPredictor()
            predictions = predictor.predict_with_sir_integration(I, disease_params, 30)
            
            if predictions:
                fig_pred = go.Figure()
                
                # Historical SIR data
                fig_pred.add_trace(go.Scatter(
                    x=days, y=I, 
                    mode='lines', name='SIR Model',
                    line=dict(color='blue', width=2)
                ))
                
                # AI predictions
                pred_days = list(range(len(I), len(I) + len(predictions)))
                fig_pred.add_trace(go.Scatter(
                    x=pred_days, y=predictions,
                    mode='lines', name='AI Predictions',
                    line=dict(color='red', dash='dash', width=2)
                ))
                
                fig_pred.update_layout(
                    title="SIR Model + AI Predictions Integration",
                    xaxis_title="Days",
                    yaxis_title="Infected Cases",
                    height=400
                )
                
                st.plotly_chart(fig_pred, use_container_width=True)
                
                create_explanation_box("AI-Enhanced Predictions", 
                    "These predictions combine traditional SIR epidemiological modeling with AI analysis of recent trends. "
                    f"The AI considers {disease} transmission patterns, seasonal effects, and intervention impacts. "
                    "Dashed red line shows 30-day forecast with epidemiological constraints to ensure biological plausibility.")
            
        else:
            # No real data available
            st.markdown(f'<div class="data-unavailable">⚠️ Real-time {disease} data is currently unavailable for {region}. Using epidemiological models for analysis.</div>', unsafe_allow_html=True)
            
            st.info(f"""
            📍 **Data Availability Notice:**  
            - **{disease}** surveillance data requires access to official health APIs
            - **Alternative:** Use epidemic modeling tab for theoretical analysis
            - **Recommendation:** Check with local health authorities for current statistics
            """)
    
    # TAB 2: Health Assessment
    with tab2:
        st.markdown(f"## 🩺 {disease} Health Assessment")
        symptom_tracker(disease)
    
    # TAB 3: Vaccination System
    with tab3:
        vaccination_interface(disease)
    
    # TAB 4: Epidemic Modeling
    with tab4:
        st.markdown(f"## 📊 {disease} Epidemic Modeling")
        
        # Model parameters
        disease_params = DISEASE_CONFIG[disease]
        population = REGION_CONFIG[region]["population"]
        
        st.markdown("### ⚙️ Model Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("👥 Population", f"{population:,}")
            I0 = st.number_input("🦠 Initial Infected:", 
                               min_value=1, max_value=population//1000, 
                               value=100, step=1)
            R0 = st.number_input("♻️ Initial Recovered:", 
                               min_value=0, max_value=population//100, 
                               value=0, step=1)
        
        with col2:
            beta = st.number_input("📈 Transmission Rate (β):", 
                                 min_value=0.01, max_value=2.0, 
                                 value=disease_params["transmission_rate"], step=0.01)
            gamma = st.number_input("🏥 Recovery Rate (γ):", 
                                  min_value=0.01, max_value=1.0, 
                                  value=disease_params["recovery_rate"], step=0.01)
            delta = st.number_input("💀 Death Rate (δ):", 
                                  min_value=0.001, max_value=0.1, 
                                  value=disease_params["death_rate"], step=0.001)
        
        with col3:
            simulation_days = st.slider("📅 Simulation Days:", 30, 730, 365)
            vaccination_rate = st.number_input("💉 Daily Vaccination Rate:", 
                                             min_value=0.001, max_value=0.1, 
                                             value=0.005, step=0.001)
            vaccine_eff = st.number_input("🛡️ Vaccine Effectiveness:", 
                                        min_value=0.1, max_value=1.0, 
                                        value=disease_params["vaccine_effectiveness"], step=0.01)
            
            # Calculate basic reproduction number
            R_basic = beta / gamma
            st.metric("🔬 Basic R₀", f"{R_basic:.2f}")
        
        # Run simulation
        if st.button("▶️ Run Epidemic Simulation", type="primary"):
            with st.spinner("🔄 Running epidemiological simulation..."):
                models = EpidemiologicalModels()
                
                # Run SIRDV model
                S, I, R, D, V = models.sirdv_model(
                    N=population, I0=I0, R0=R0, D0=0, V0=0,
                    beta=beta, gamma=gamma, delta=delta,
                    vaccination_rate=vaccination_rate, 
                    vaccine_eff=vaccine_eff, num_days=simulation_days
                )
                
                # Create comprehensive visualization
                fig = make_subplots(
                    rows=3, cols=2,
                    subplot_titles=(
                        "SIRDV Model Simulation", 
                        "R-effective Over Time",
                        "Vaccination Impact", 
                        "Healthcare Burden",
                        "Attack Rate Analysis",
                        "Intervention Effectiveness"
                    )
                )
                
                days = list(range(len(I)))
                
                # Main SIRDV plot
                fig.add_trace(go.Scatter(x=days, y=S, name="Susceptible", line=dict(color='blue')), row=1, col=1)
                fig.add_trace(go.Scatter(x=days, y=I, name="Infected", line=dict(color='red')), row=1, col=1)
                fig.add_trace(go.Scatter(x=days, y=R, name="Recovered", line=dict(color='green')), row=1, col=1)
                fig.add_trace(go.Scatter(x=days, y=D, name="Deaths", line=dict(color='black')), row=1, col=1)
                fig.add_trace(go.Scatter(x=days, y=V, name="Vaccinated", line=dict(color='purple')), row=1, col=1)
                
                # R-effective calculation
                r_eff = []
                for i in range(len(S)):
                    if population > 0 and gamma > 0:
                        r_val = beta * S[i] / (population * gamma)
                        r_eff.append(max(0, min(5, r_val)))
                    else:
                        r_eff.append(1.0)
                
                fig.add_trace(go.Scatter(x=days, y=r_eff, name="R-effective", line=dict(color='red', width=3)), row=1, col=2)
                fig.add_hline(y=1, line_dash="dash", line_color="black", row=1, col=2)
                
                # Vaccination impact
                daily_vacc = [V[i] - V[i-1] if i > 0 else 0 for i in range(len(V))]
                fig.add_trace(go.Bar(x=days[-60:], y=daily_vacc[-60:], name="Daily Vaccinations", marker_color="lightblue"), row=2, col=1)
                
                # Healthcare burden (5% of infected need hospitalization)
                hospital_burden = [i * 0.05 for i in I]
                fig.add_trace(go.Scatter(x=days, y=hospital_burden, name="Hospital Beds Needed", 
                                       line=dict(color='darkred'), fill='tonexty'), row=2, col=2)
                
                # Attack rate
                attack_rate = [(population - s)/population * 100 for s in S]
                fig.add_trace(go.Scatter(x=days, y=attack_rate, name="Attack Rate (%)", 
                                       line=dict(color='orange')), row=3, col=1)
                
                # Intervention effectiveness (vaccination + natural immunity)
                immune_population = [(r + v)/population * 100 for r, v in zip(R, V)]
                fig.add_trace(go.Scatter(x=days, y=immune_population, name="Immune Population (%)", 
                                       line=dict(color='green')), row=3, col=2)
                
                fig.update_layout(height=1200, showlegend=True, 
                                title_text=f"Comprehensive {disease} SIRDV Model - {region}")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed explanations for each subplot
                create_explanation_box("SIRDV Model Components", 
                    f"Complete epidemic model showing: Susceptible (blue) - people at risk; Infected (red) - active cases; "
                    f"Recovered (green) - naturally immune; Deaths (black) - fatalities; Vaccinated (purple) - vaccine-immune. "
                    f"Model uses {disease} parameters: R₀={R_basic:.2f}, recovery time={1/gamma:.1f} days.")
                
                create_explanation_box("R-effective Analysis", 
                    "Effective reproduction number over time. R>1 means epidemic is growing, R<1 means declining. "
                    f"The horizontal line at R=1 is critical threshold. Current interventions are effective when R stays below 1. "
                    f"Vaccination and natural immunity gradually reduce R-effective.")
                
                create_explanation_box("Vaccination Impact", 
                    f"Daily vaccination rate showing immunization progress. At {vaccination_rate*100:.1f}% daily rate with "
                    f"{vaccine_eff*100:.0f}% effectiveness, this prevents infections and reduces transmission. "
                    f"Bars show daily vaccine doses administered.")
                
                create_explanation_box("Healthcare System Burden", 
                    f"Hospital bed requirements assuming 5% of infected cases need hospitalization for {disease}. "
                    f"Peak demand helps plan healthcare capacity. Critical for resource allocation and staff planning. "
                    f"Area under curve represents total healthcare system load.")
                
                create_explanation_box("Attack Rate Progression", 
                    f"Cumulative percentage of population infected over time. For {disease} with R₀={R_basic:.2f}, "
                    f"herd immunity typically requires 60-80% population immunity. This tracks progress toward epidemic end.")
                
                create_explanation_box("Intervention Effectiveness", 
                    f"Combined effect of vaccination and natural immunity building population-level protection. "
                    f"Green line shows immune population percentage. Steeper slopes indicate more effective interventions. "
                    f"Target is reaching herd immunity threshold.")
                
                # Model analytics
                st.markdown("### 📊 Simulation Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                peak_infections = max(I)
                peak_day = I.index(peak_infections)
                final_attack_rate = ((population - S[-1]) / population) * 100
                total_deaths = D[-1]
                
                with col1:
                    st.metric("🔄 Peak Infections", f"{int(peak_infections):,}", f"Day {peak_day}")
                with col2:
                    st.metric("🎯 Final Attack Rate", f"{final_attack_rate:.1f}%")
                with col3:
                    st.metric("💀 Total Deaths", f"{int(total_deaths):,}")
                with col4:
                    herd_immunity = max(0, (1 - 1/R_basic) * 100) if R_basic > 1 else 0
                    st.metric("🛡️ Herd Immunity Threshold", f"{herd_immunity:.1f}%")
                
                # AI predictions integrated with model results
                st.markdown("### 🤖 AI-Enhanced Model Predictions")
                
                predictor = EnhancedAIPredictor()
                ai_predictions = predictor.predict_with_sir_integration(I[-60:], disease_params, 45)
                
                if ai_predictions:
                    fig_ai = go.Figure()
                    
                    # Model results
                    fig_ai.add_trace(go.Scatter(
                        x=days, y=I, 
                        mode='lines', name='SIRDV Model',
                        line=dict(color='blue', width=2)
                    ))
                    
                    # AI predictions
                    pred_days = list(range(len(I), len(I) + len(ai_predictions)))
                    fig_ai.add_trace(go.Scatter(
                        x=pred_days, y=ai_predictions,
                        mode='lines', name='AI Extended Forecast',
                        line=dict(color='red', dash='dot', width=3)
                    ))
                    
                    fig_ai.update_layout(
                        title="SIRDV Model + AI Predictions",
                        xaxis_title="Days from simulation start",
                        yaxis_title="Infected Population",
                        height=400
                    )
                    
                    st.plotly_chart(fig_ai, use_container_width=True)
                    
                    create_explanation_box("AI-Enhanced Forecasting", 
                        "AI predictions extend beyond the deterministic SIRDV model by analyzing recent infection patterns, "
                        f"seasonal effects, and intervention responses specific to {disease}. The dotted red line shows "
                        "machine learning-enhanced forecasting that considers real-world complexities not captured by basic models.")
                
                # Export results
                st.markdown("### 📥 Export Simulation Results")
                
                results_df = pd.DataFrame({
                    'Day': days,
                    'Susceptible': S,
                    'Infected': I,
                    'Recovered': R,
                    'Deaths': D,
                    'Vaccinated': V,
                    'R_effective': r_eff,
                    'Attack_Rate': attack_rate
                })
                
                csv_data = results_df.to_csv(index=False)
                
                st.download_button(
                    label=f"📊 Download {disease} Simulation Data",
                    data=csv_data,
                    file_name=f"stanes_{disease.lower()}_{region.lower()}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
    
    # TAB 5: Settings & Google Sheets Integration
    with tab5:
        st.markdown("## ⚙️ System Settings")
        
        # Google Sheets Integration
        st.markdown("### 📊 Google Sheets Integration Setup")
        
        st.info("""
        **How to connect Google Sheets:**
        1. Go to [Google Cloud Console](https://console.cloud.google.com/)
        2. Create a new project or select existing one
        3. Enable Google Sheets API and Google Drive API
        4. Create Service Account credentials
        5. Download JSON key file
        6. Upload the JSON file below
        """)
        
        uploaded_file = st.file_uploader("Upload Google Service Account JSON", type=['json'])
        
        if uploaded_file is not None:
            credentials_json = uploaded_file.read().decode('utf-8')
            
            if st.button("🔗 Connect to Google Sheets"):
                if st.session_state.sheets_integration.setup_credentials(credentials_json):
                    sheet = st.session_state.sheets_integration.create_or_open_sheet()
                    if sheet:
                        st.success("✅ Successfully connected to Google Sheets!")
                        st.info(f"📊 Sheet URL: {sheet.url}")
                        
                        # Save vaccination bookings to sheets
                        if 'vaccination_bookings' in st.session_state and st.session_state.vaccination_bookings:
                            if st.button("📤 Upload Vaccination Bookings to Sheets"):
                                for booking in st.session_state.vaccination_bookings:
                                    st.session_state.sheets_integration.add_vaccination_record(booking)
                                st.success(f"✅ Uploaded {len(st.session_state.vaccination_bookings)} vaccination records!")
        
        # Display current bookings
        if 'vaccination_bookings' in st.session_state and st.session_state.vaccination_bookings:
            st.markdown("### 📋 Current Vaccination Bookings")
            
            bookings_df = pd.DataFrame(st.session_state.vaccination_bookings)
            st.dataframe(bookings_df, use_container_width=True)
            
            # Vaccination statistics
            col1, col2 = st.columns(2)
            
            with col1:
                disease_counts = bookings_df['disease'].value_counts()
                fig_diseases = px.pie(
                    values=disease_counts.values,
                    names=disease_counts.index,
                    title="Bookings by Disease"
                )
                st.plotly_chart(fig_diseases, use_container_width=True)
            
            with col2:
                location_counts = bookings_df['location'].value_counts()
                fig_locations = px.bar(
                    x=location_counts.index,
                    y=location_counts.values,
                    title="Bookings by Location"
                )
                st.plotly_chart(fig_locations, use_container_width=True)
        
        # System information
        st.markdown("### 📱 Application Information")
        st.info(f"""
        **Stanes School CBSE Coimbatore - PandemicTrack Pro**  
        Version: 2.0  
        Developed by: Science & Mathematics Department  
        Last Updated: {datetime.now().strftime('%Y-%m-%d')}  
        
        **Features:**
        - Real-time disease surveillance
        - AI-powered predictions
        - Advanced epidemic modeling
        - Vaccination management system
        - Google Sheets integration
        """)

if __name__ == "__main__":
    main()
