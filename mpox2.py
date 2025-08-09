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
import folium
from streamlit_folium import st_folium
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings
import io
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="🦠 PandemicPro - Global Health Surveillance",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with school branding
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Playfair+Display:wght@400;700&display=swap');
    
    .header-bar {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 0.8rem 2rem;
        margin: -1rem -1rem 2rem -1rem;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        border-bottom: 3px solid #ffd700;
    }
    
    .school-logo {
        height: 50px;
        width: auto;
        margin-right: 20px;
        border-radius: 5px;
    }
    
    .school-info {
        text-align: center;
    }
    
    .school-name {
        font-family: 'Playfair Display', serif;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: 1px;
    }
    
    .school-motto {
        font-family: 'Playfair Display', serif;
        font-style: italic;
        font-size: 0.9rem;
        margin: 0;
        opacity: 0.9;
        letter-spacing: 0.5px;
    }
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin: 1rem 0 2rem 0;
    }
    
    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 1.1rem;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
        margin: 0.5rem 0;
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
    
    .graph-explanation {
        background-color: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 0.8rem;
        margin: 0.5rem 0;
        border-radius: 5px;
        font-size: 0.9rem;
        color: #495057;
    }
</style>
""", unsafe_allow_html=True)

# School Header Bar
st.markdown("""
<div class="header-bar">
    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQgdeJiprfycNu9R2M39-aSOwrdSio-K4q8HA&s" class="school-logo" alt="Stanes CBSE Logo">
    <div class="school-info">
        <div class="school-name">STANES SCHOOL - CBSE</div>
        <div class="school-motto">Excelsa Sequar</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Disease configurations with real-world parameters
DISEASE_CONFIG = {
    "COVID-19": {
        "transmission_rate": 0.4,
        "recovery_rate": 0.1,
        "death_rate": 0.02,
        "incubation_period": 5.1,
        "infectious_period": 10,
        "basic_r0": 2.5,
        "vaccine_effectiveness": 0.95,
        "vaccines": ["COVISHIELD", "COVAXIN", "Pfizer-BioNTech", "Moderna", "Johnson & Johnson"]
    },
    "Mpox": {
        "transmission_rate": 0.15,
        "recovery_rate": 0.05,
        "death_rate": 0.01,
        "incubation_period": 12,
        "infectious_period": 21,
        "basic_r0": 1.8,
        "vaccine_effectiveness": 0.85,
        "vaccines": ["JYNNEOS", "ACAM2000"]
    },
    "Influenza": {
        "transmission_rate": 0.5,
        "recovery_rate": 0.15,
        "death_rate": 0.001,
        "incubation_period": 2,
        "infectious_period": 7,
        "basic_r0": 1.3,
        "vaccine_effectiveness": 0.60,
        "vaccines": ["Seasonal Flu Vaccine", "H1N1 Vaccine"]
    },
    "H5N1 Bird Flu": {
        "transmission_rate": 0.1,
        "recovery_rate": 0.08,
        "death_rate": 0.6,
        "incubation_period": 7,
        "infectious_period": 14,
        "basic_r0": 0.8,
        "vaccine_effectiveness": 0.70,
        "vaccines": ["H5N1 Vaccine (Research)"]
    },
    "Dengue": {
        "transmission_rate": 0.3,
        "recovery_rate": 0.12,
        "death_rate": 0.025,
        "incubation_period": 5,
        "infectious_period": 7,
        "basic_r0": 2.0,
        "vaccine_effectiveness": 0.65,
        "vaccines": ["Dengvaxia"]
    }
}

# Country configurations
COUNTRY_CONFIG = {
    "Global": {"population": 8000000000, "api_code": "all"},
    "India": {"population": 1428000000, "api_code": "india"},
    "USA": {"population": 331000000, "api_code": "usa"},
    "China": {"population": 1425000000, "api_code": "china"},
    "Brazil": {"population": 215000000, "api_code": "brazil"},
    "Russia": {"population": 146000000, "api_code": "russia"},
    "Germany": {"population": 84000000, "api_code": "germany"},
    "UK": {"population": 67000000, "api_code": "uk"},
    "France": {"population": 68000000, "api_code": "france"},
    "Japan": {"population": 125000000, "api_code": "japan"},
    "Italy": {"population": 60000000, "api_code": "italy"},
    "South Korea": {"population": 52000000, "api_code": "south korea"},
    "Australia": {"population": 26000000, "api_code": "australia"},
    "Canada": {"population": 39000000, "api_code": "canada"}
}

# CSV Data Storage System
class CSVDataManager:
    def __init__(self):
        self.vaccination_bookings = []
        self.symptom_reports = []
        self.model_results = []
    
    def add_vaccination_booking(self, booking_data):
        """Add vaccination booking to CSV storage"""
        booking_entry = {
            'timestamp': datetime.now().isoformat(),
            'name': booking_data.get('name', ''),
            'phone': booking_data.get('phone', ''),
            'email': booking_data.get('email', ''),
            'disease': booking_data.get('disease', ''),
            'vaccine_type': booking_data.get('vaccine_type', ''),
            'preferred_date': str(booking_data.get('date', '')),
            'preferred_time': booking_data.get('time', ''),
            'center_name': booking_data.get('center_name', ''),
            'center_address': booking_data.get('center_address', ''),
            'status': 'Pending'
        }
        self.vaccination_bookings.append(booking_entry)
        
        # Check if high volume and show WhatsApp group
        if len(self.vaccination_bookings) > 50:  # Threshold for high volume
            self.show_high_volume_alert()
    
    def show_high_volume_alert(self):
        """Show WhatsApp group alert for high volume bookings"""
        st.warning("⚠️ High volume of bookings detected!")
        st.markdown("""
        <div class="alert-card">
            📱 <strong>Join High-Priority WhatsApp Group for Updates:</strong><br>
            <a href="https://chat.whatsapp.com/your-group-link" target="_blank" style="color: white; font-weight: bold;">
                🔗 Click here to join emergency coordination group
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    def get_vaccination_csv(self):
        """Generate CSV for vaccination bookings"""
        if not self.vaccination_bookings:
            return "No booking data available"
        
        df = pd.DataFrame(self.vaccination_bookings)
        return df.to_csv(index=False)
    
    def add_symptom_report(self, symptom_data):
        """Add symptom report to CSV storage"""
        self.symptom_reports.append({
            'timestamp': datetime.now().isoformat(),
            **symptom_data
        })
    
    def add_model_result(self, model_data):
        """Add model results to CSV storage"""
        self.model_results.append({
            'timestamp': datetime.now().isoformat(),
            **model_data
        })

# Initialize CSV manager
if 'csv_manager' not in st.session_state:
    st.session_state.csv_manager = CSVDataManager()

# Enhanced Data Fetcher with Multiple APIs
class EnhancedDataFetcher:
    @staticmethod
    def fetch_disease_data(country, disease):
        """Fetch real data based on disease and country"""
        try:
            if disease == "COVID-19":
                return EnhancedDataFetcher.fetch_covid_data(country)
            elif disease == "Mpox":
                return EnhancedDataFetcher.fetch_mpox_data(country)
            elif disease == "Influenza":
                return EnhancedDataFetcher.fetch_influenza_data(country)
            else:
                return EnhancedDataFetcher.get_simulated_data(country, disease)
        except Exception as e:
            st.error(f"Data fetch error: {str(e)}")
            return EnhancedDataFetcher.get_simulated_data(country, disease)
    
    @staticmethod
    def fetch_covid_data(country):
        """Fetch COVID-19 data from disease.sh API"""
        try:
            country_config = COUNTRY_CONFIG.get(country, COUNTRY_CONFIG["Global"])
            api_code = country_config["api_code"]
            
            if api_code == "all":
                url = "https://disease.sh/v3/covid-19/all"
            else:
                url = f"https://disease.sh/v3/covid-19/countries/{api_code}"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                return {
                    "disease": "COVID-19",
                    "country": country,
                    "total_cases": data.get("cases", 0),
                    "active_cases": data.get("active", 0),
                    "recovered": data.get("recovered", 0),
                    "deaths": data.get("deaths", 0),
                    "tests": data.get("tests", 0),
                    "population": country_config["population"],
                    "last_updated": datetime.fromtimestamp(data.get("updated", 0)/1000).strftime('%Y-%m-%d %H:%M:%S'),
                    "source": "Johns Hopkins via Disease.sh API",
                    "daily_cases": data.get("todayCases", 0),
                    "daily_deaths": data.get("todayDeaths", 0),
                    "daily_recovered": data.get("todayRecovered", 0)
                }
        except Exception as e:
            st.warning(f"COVID API failed for {country}: {str(e)}")
        
        return EnhancedDataFetcher.get_simulated_data(country, "COVID-19")
    
    @staticmethod
    def fetch_mpox_data(country):
        """Fetch Mpox data (simulated with realistic numbers)"""
        country_config = COUNTRY_CONFIG.get(country, COUNTRY_CONFIG["Global"])
        population = country_config["population"]
        
        # Realistic Mpox data based on WHO reports
        if country == "Global":
            total_cases = 95000  # WHO reported ~95k cases globally
            deaths = 950
        elif country == "USA":
            total_cases = 30000
            deaths = 25
        elif country == "Brazil":
            total_cases = 8500
            deaths = 12
        else:
            # Calculate proportional data
            global_pop = 8000000000
            total_cases = int(95000 * (population / global_pop))
            deaths = int(total_cases * 0.01)  # ~1% fatality rate for Mpox
        
        active_cases = int(total_cases * 0.15)  # 15% active
        recovered = total_cases - active_cases - deaths
        
        return {
            "disease": "Mpox",
            "country": country,
            "total_cases": total_cases,
            "active_cases": active_cases,
            "recovered": recovered,
            "deaths": deaths,
            "tests": total_cases * 3,  # Assumed testing rate
            "population": population,
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "source": "WHO Mpox Surveillance (Simulated)",
            "daily_cases": max(0, int(total_cases * 0.001)),
            "daily_deaths": max(0, int(deaths * 0.001)),
            "daily_recovered": max(0, int(recovered * 0.01))
        }
    
    @staticmethod
    def fetch_influenza_data(country):
        """Fetch Influenza data (seasonal estimates)"""
        country_config = COUNTRY_CONFIG.get(country, COUNTRY_CONFIG["Global"])
        population = country_config["population"]
        
        # Seasonal influenza estimates (WHO data)
        seasonal_rate = 0.05  # 5% of population gets flu annually
        total_cases = int(population * seasonal_rate)
        deaths = int(total_cases * 0.001)  # 0.1% fatality rate
        active_cases = int(total_cases * 0.20)  # 20% currently active
        recovered = total_cases - active_cases - deaths
        
        return {
            "disease": "Influenza",
            "country": country,
            "total_cases": total_cases,
            "active_cases": active_cases,
            "recovered": recovered,
            "deaths": deaths,
            "tests": total_cases * 2,
            "population": population,
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "source": "WHO Influenza Surveillance (Estimated)",
            "daily_cases": max(0, int(total_cases * 0.002)),
            "daily_deaths": max(0, int(deaths * 0.01)),
            "daily_recovered": max(0, int(recovered * 0.02))
        }
    
    @staticmethod
    def get_simulated_data(country, disease):
        """Generate realistic simulated data when APIs fail"""
        country_config = COUNTRY_CONFIG.get(country, COUNTRY_CONFIG["Global"])
        population = country_config["population"]
        disease_params = DISEASE_CONFIG.get(disease, DISEASE_CONFIG["COVID-19"])
        
        # Disease-specific realistic estimates
        if disease == "COVID-19":
            attack_rate = 0.15  # 15% cumulative attack rate
        elif disease == "Mpox":
            attack_rate = 0.00001  # Very low attack rate
        elif disease == "Influenza":
            attack_rate = 0.05  # 5% seasonal attack rate
        else:
            attack_rate = 0.02  # Default 2%
        
        total_cases = int(population * attack_rate)
        deaths = int(total_cases * disease_params["death_rate"])
        recovered = int(total_cases * 0.90)  # 90% recovery rate
        active_cases = total_cases - recovered - deaths
        
        return {
            "disease": disease,
            "country": country,
            "total_cases": total_cases,
            "active_cases": max(0, active_cases),
            "recovered": recovered,
            "deaths": deaths,
            "tests": total_cases * 5,
            "population": population,
            "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "source": "Simulated based on epidemiological parameters",
            "daily_cases": max(0, int(total_cases * 0.001)),
            "daily_deaths": max(0, int(deaths * 0.01)),
            "daily_recovered": max(0, int(recovered * 0.02))
        }

# AI Prediction System
class AIPredictor:
    @staticmethod
    def generate_historical_data(current_data, days=180):
        """Generate realistic historical data for AI prediction"""
        total_cases = current_data['total_cases']
        active_cases = current_data['active_cases']
        
        # Create epidemic curve with multiple waves
        historical_data = []
        
        for i in range(days):
            t = i / days
            
            # Multi-wave simulation based on real epidemic patterns
            wave1 = active_cases * 0.3 * np.exp(-((t - 0.2) * 8)**2)  # Early wave
            wave2 = active_cases * 0.8 * np.exp(-((t - 0.5) * 6)**2)   # Main wave
            wave3 = active_cases * 0.4 * np.exp(-((t - 0.8) * 10)**2)  # Recent wave
            
            baseline = active_cases * 0.1  # Baseline level
            noise = np.random.normal(0, active_cases * 0.02)
            
            daily_value = max(0, wave1 + wave2 + wave3 + baseline + noise)
            historical_data.append(daily_value)
        
        return historical_data
    
    @staticmethod
    def predict_future_trend(historical_data, days_ahead=30, disease_params=None):
        """Enhanced AI prediction with disease-specific parameters"""
        if len(historical_data) < 14:
            return [max(0, historical_data[-1]) for _ in range(days_ahead)]
        
        try:
            # Ensemble prediction combining multiple methods
            predictions = []
            
            # Method 1: Exponential smoothing with trend
            alpha, beta = 0.3, 0.1
            smoothed = [historical_data[0]]
            trend = [0]
            
            for i in range(1, len(historical_data)):
                s_prev, t_prev = smoothed[-1], trend[-1]
                s_new = alpha * historical_data[i] + (1 - alpha) * (s_prev + t_prev)
                t_new = beta * (s_new - s_prev) + (1 - beta) * t_prev
                smoothed.append(s_new)
                trend.append(t_new)
            
            exp_predictions = []
            for i in range(days_ahead):
                pred = smoothed[-1] + trend[-1] * (i + 1)
                exp_predictions.append(max(0, pred))
            
            # Method 2: Polynomial regression
            X = np.array(range(len(historical_data))).reshape(-1, 1)
            poly_features = PolynomialFeatures(degree=min(3, len(historical_data)//10))
            X_poly = poly_features.fit_transform(X)
            
            model = LinearRegression()
            model.fit(X_poly, historical_data)
            
            future_X = np.array(range(len(historical_data), len(historical_data) + days_ahead)).reshape(-1, 1)
            future_X_poly = poly_features.transform(future_X)
            poly_predictions = model.predict(future_X_poly)
            poly_predictions = [max(0, p) for p in poly_predictions]
            
            # Method 3: Seasonal decomposition (simplified)
            seasonal_factor = np.sin(2 * np.pi * np.array(range(days_ahead)) / 7) * 0.1 + 1
            seasonal_predictions = [historical_data[-1] * factor for factor in seasonal_factor]
            
            # Ensemble with weights
            ensemble_predictions = []
            weights = [0.4, 0.4, 0.2]  # Exponential, Polynomial, Seasonal
            
            for i in range(days_ahead):
                weighted_pred = (
                    exp_predictions[i] * weights[0] +
                    poly_predictions[i] * weights[1] +
                    seasonal_predictions[i] * weights[2]
                )
                
                # Apply disease-specific constraints
                if disease_params:
                    # Prevent unrealistic growth
                    max_growth = historical_data[-1] * 1.1  # Max 10% daily growth
                    weighted_pred = min(weighted_pred, max_growth)
                
                ensemble_predictions.append(max(0, weighted_pred))
            
            return ensemble_predictions
            
        except Exception as e:
            st.warning(f"AI prediction error: {str(e)}")
            # Simple fallback
            recent_avg = np.mean(historical_data[-7:])
            recent_trend = np.mean(np.diff(historical_data[-14:]))
            return [max(0, recent_avg + recent_trend * i) for i in range(1, days_ahead + 1)]

# SIR Model Implementation
class SIRModel:
    @staticmethod
    def run_sir_simulation(
    N, I0, R0, D0, beta_0, beta_hr, f_hr, gamma, delta, v, e, num_days
):
    S = N - I0 - R0 - D0  # Account for deceased individuals
    I = I0
    R = R0
    D = D0  # Start with an initial number of deceased individuals

    susceptible = [S]
    infected = [I]
    recovered = [R]
    deceased = [D]

    # Effective transmission rate due to high-risk group interactions
    beta_eff = beta_0 + f_hr * (beta_hr - beta_0)

    for _ in range(num_days):
        # Vaccination effect
        vaccinated = v * S  # Number of vaccinated individuals per day
        directly_immune = vaccinated * e  # Vaccinated individuals gaining full immunity
        partially_immune = vaccinated * (1 - e)  # Remaining partially immune population

        S_new = S - beta_eff * S * I / N - vaccinated  # Remove vaccinated from susceptible
        I_new = I + beta_eff * S * I / N - gamma * I - delta * I
        R_new = R + gamma * I + directly_immune  # Add directly immune to recovered
        D_new = D + delta * I

        S, I, R, D = S_new, I_new, R_new, D_new

        susceptible.append(S)
        infected.append(I)
        recovered.append(R)
        deceased.append(D)

    return susceptible, infected, recovered, deceased

# Vaccination Centers Data
def get_vaccination_centers_by_disease(disease):
    """Get vaccination centers based on disease type"""
    base_centers = [
        {
            "name": "All India Institute of Medical Sciences (AIIMS) Delhi",
            "address": "Sri Aurobindo Marg, Ansari Nagar, New Delhi",
            "city": "New Delhi",
            "state": "Delhi",
            "lat": 28.5672,
            "lon": 77.2100,
            "phone": "+91-11-26588500"
        },
        {
            "name": "King Edward Memorial Hospital",
            "address": "Acharya Donde Marg, Parel, Mumbai",
            "city": "Mumbai",
            "state": "Maharashtra",
            "lat": 19.0176,
            "lon": 72.8442,
            "phone": "+91-22-24129884"
        },
        {
            "name": "Postgraduate Institute of Medical Education",
            "address": "Sector 12, Chandigarh",
            "city": "Chandigarh",
            "state": "Chandigarh",
            "lat": 30.7333,
            "lon": 76.7794,
            "phone": "+91-172-2747585"
        },
        {
            "name": "Christian Medical College",
            "address": "Ida Scudder Road, Vellore",
            "city": "Vellore",
            "state": "Tamil Nadu",
            "lat": 12.9249,
            "lon": 79.1353,
            "phone": "+91-416-228-2052"
        },
        {
            "name": "Sanjay Gandhi Postgraduate Institute",
            "address": "Raebareli Road, Lucknow",
            "city": "Lucknow",
            "state": "Uttar Pradesh",
            "lat": 26.8484,
            "lon": 80.9462,
            "phone": "+91-522-249-4401"
        }
    ]
    
    # Add disease-specific vaccines and availability
    disease_vaccines = DISEASE_CONFIG.get(disease, DISEASE_CONFIG["COVID-19"])["vaccines"]
    
    for center in base_centers:
        center["available_vaccines"] = disease_vaccines
        center["availability"] = np.random.choice(["High", "Medium", "Low"], p=[0.4, 0.4, 0.2])
        center["timing"] = "9:00 AM - 5:00 PM"
    
    return base_centers

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">🦠 PandemicPro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Advanced Global Health Surveillance & Disease Monitoring System</p>', unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.markdown("## 🎛️ Global Controls")
    
    # Disease selection (dropdown only)
    disease = st.sidebar.selectbox(
        "🦠 Select Disease",
        ["COVID-19", "Mpox", "Influenza", "H5N1 Bird Flu", "Dengue"],
        help="Choose the disease to track and analyze"
    )
    
    # Country selection (dropdown only)
    country = st.sidebar.selectbox(
        "🌍 Select Country/Region",
        list(COUNTRY_CONFIG.keys()),
        help="Choose geographical region for analysis"
    )
    
    # Data refresh
    if st.sidebar.button("🔄 Refresh Data"):
        with st.spinner("Fetching latest data..."):
            st.cache_data.clear()
            time.sleep(1)
    
    # Fetch real-time data
    current_data = EnhancedDataFetcher.fetch_disease_data(country, disease)
    
    # Sidebar metrics
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 📊 Live {disease} Data")
    st.sidebar.metric("🦠 Total Cases", f"{current_data['total_cases']:,}")
    st.sidebar.metric("⚡ Active Cases", f"{current_data['active_cases']:,}")
    st.sidebar.metric("♻️ Recovered", f"{current_data['recovered']:,}")
    st.sidebar.metric("💀 Deaths", f"{current_data['deaths']:,}")
    
    # Advanced metrics
    if current_data['total_cases'] > 0:
        mortality_rate = (current_data['deaths'] / current_data['total_cases']) * 100
        recovery_rate = (current_data['recovered'] / current_data['total_cases']) * 100
        st.sidebar.metric("💀 Case Fatality Rate", f"{mortality_rate:.2f}%")
        st.sidebar.metric("♻️ Recovery Rate", f"{recovery_rate:.2f}%")
        
        # Population-based metrics
        incidence_rate = (current_data['total_cases'] / current_data['population']) * 100000
        st.sidebar.metric("📈 Cases per 100K", f"{incidence_rate:.0f}")
    
    st.sidebar.caption(f"🕒 Last updated: {current_data.get('last_updated', 'Unknown')}")
    st.sidebar.caption(f"📊 Source: {current_data.get('source', 'Simulated')}")
    
    # Navigation tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🏠 Dashboard", "🗺️ Disease Map", "🩺 Health Assessment", 
        "💉 Vaccination", "⚠️ Alerts", "📊 SIR Modeling"
    ])
    
    # TAB 1: Enhanced Dashboard with AI Predictions
    with tab1:
        st.markdown("## 📊 Real-time Dashboard with AI Intelligence")
        
        # Key metrics with daily changes
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🦠 Total Cases", f"{current_data['total_cases']:,}", 
                     delta=f"+{current_data['daily_cases']:,}")
        
        with col2:
            st.metric("⚡ Active Cases", f"{current_data['active_cases']:,}", 
                     delta=f"+{current_data['daily_cases'] - current_data['daily_recovered'] - current_data['daily_deaths']:,}")
        
        with col3:
            st.metric("♻️ Recovered", f"{current_data['recovered']:,}", 
                     delta=f"+{current_data['daily_recovered']:,}")
        
        with col4:
            st.metric("💀 Deaths", f"{current_data['deaths']:,}", 
                     delta=f"+{current_data['daily_deaths']:,}")
        
        # Historical trends and AI predictions
        st.markdown("### 📈 Historical Trends & AI Predictions")
        
        # Generate historical data
        ai_predictor = AIPredictor()
        historical_data = ai_predictor.generate_historical_data(current_data, 120)
        
        # Generate AI predictions
        disease_params = DISEASE_CONFIG.get(disease, DISEASE_CONFIG["COVID-19"])
        future_predictions = ai_predictor.predict_future_trend(historical_data[-60:], 30, disease_params)
        
        # Create visualization
        fig = go.Figure()
        
        # Historical data
        dates_hist = [datetime.now() - timedelta(days=120-i) for i in range(120)]
        fig.add_trace(go.Scatter(
            x=dates_hist,
            y=historical_data,
            mode='lines',
            name='Historical Data',
            line=dict(color='#3498db', width=3)
        ))
        
        # AI predictions
        dates_future = [datetime.now() + timedelta(days=i) for i in range(1, 31)]
        fig.add_trace(go.Scatter(
            x=dates_future,
            y=future_predictions,
            mode='lines',
            name='AI Predictions',
            line=dict(color='#e74c3c', width=3, dash='dash')
        ))
        
        # Confidence interval
        upper_bound = [p * 1.3 for p in future_predictions]
        lower_bound = [p * 0.7 for p in future_predictions]
        
        fig.add_trace(go.Scatter(
            x=dates_future + dates_future[::-1],
            y=upper_bound + lower_bound[::-1],
            fill='tonexty',
            fillcolor='rgba(231, 76, 60, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Prediction Confidence',
            showlegend=False
        ))
        
        fig.update_layout(
            title=f"{disease} Trends and AI Predictions - {country}",
            xaxis_title="Date",
            yaxis_title="Active Cases",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Graph explanation
        st.markdown("""
        <div class="graph-explanation">
        <strong>📊 Graph Interpretation:</strong> The blue line shows historical case trends based on epidemiological patterns. 
        The red dashed line represents AI predictions for the next 30 days using ensemble modeling. 
        The shaded area shows prediction confidence intervals (±30%).
        </div>
        """, unsafe_allow_html=True)
        
        # Current situation analysis
        st.markdown("### 🤖 AI Situation Analysis")
        
        recent_trend = np.mean(np.diff(historical_data[-14:]))
        prediction_trend = np.mean(np.diff(future_predictions[:14]))
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction_trend > recent_trend * 1.2:
                st.markdown('<div class="alert-card">📈 <strong>Accelerating Trend</strong>: Cases expected to increase significantly</div>', unsafe_allow_html=True)
            elif prediction_trend < recent_trend * 0.8:
                st.markdown('<div class="success-card">📉 <strong>Declining Trend</strong>: Cases expected to decrease</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="info-card">📊 <strong>Stable Trend</strong>: Cases expected to remain relatively stable</div>', unsafe_allow_html=True)
        
        with col2:
            r_effective = (disease_params["transmission_rate"] / disease_params["recovery_rate"]) * (current_data["active_cases"] / current_data["population"])
            
            if r_effective > 1.5:
                st.markdown('<div class="risk-high">🚨 Very High Transmission Risk</div>', unsafe_allow_html=True)
            elif r_effective > 1.0:
                st.markdown('<div class="risk-medium">⚠️ Moderate Transmission Risk</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-low">✅ Low Transmission Risk</div>', unsafe_allow_html=True)
        
        # Disease comparison
        st.markdown("### 🔬 Disease Comparison Dashboard")
        
        comparison_data = []
        for d_name, d_params in DISEASE_CONFIG.items():
            d_data = EnhancedDataFetcher.fetch_disease_data(country, d_name)
            comparison_data.append({
                'Disease': d_name,
                'Total Cases': d_data['total_cases'],
                'Active Cases': d_data['active_cases'],
                'Mortality Rate': f"{(d_data['deaths'] / max(1, d_data['total_cases'])) * 100:.2f}%",
                'R₀': d_params['basic_r0']
            })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Create comparison chart
        fig_comp = px.bar(
            df_comparison, 
            x='Disease', 
            y='Active Cases',
            title=f"Disease Comparison - {country}",
            color='Disease',
            color_discrete_sequence=['#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']
        )
        
        st.plotly_chart(fig_comp, use_container_width=True)
        
        st.markdown("""
        <div class="graph-explanation">
        <strong>📊 Comparison Analysis:</strong> This chart compares active cases across different diseases in your selected region. 
        Higher bars indicate diseases with more current active cases requiring immediate attention.
        </div>
        """, unsafe_allow_html=True)
    
    # TAB 2: Disease Map
    with tab2:
        st.markdown("## 🗺️ Global Disease Distribution Map")
        
        # Create world map with disease-specific data
        world_map = folium.Map(location=[20, 0], zoom_start=2)
        
        # Major cities with disease data
        major_cities = [
            {"name": "New York", "country": "USA", "lat": 40.7128, "lon": -74.0060},
            {"name": "London", "country": "UK", "lat": 51.5074, "lon": -0.1278},
            {"name": "Mumbai", "country": "India", "lat": 19.0760, "lon": 72.8777},
            {"name": "Delhi", "country": "India", "lat": 28.6139, "lon": 77.2090},
            {"name": "Tokyo", "country": "Japan", "lat": 35.6762, "lon": 139.6503},
            {"name": "São Paulo", "country": "Brazil", "lat": -23.5558, "lon": -46.6396},
            {"name": "Berlin", "country": "Germany", "lat": 52.5200, "lon": 13.4050},
            {"name": "Paris", "country": "France", "lat": 48.8566, "lon": 2.3522}
        ]
        
        for city in major_cities:
            # Get country data for each city
            city_country = city["country"]
            if city_country in COUNTRY_CONFIG:
                city_data = EnhancedDataFetcher.fetch_disease_data(city_country, disease)
                
                # Calculate city-level estimates (proportional)
                country_pop = COUNTRY_CONFIG[city_country]["population"]
                city_pop = country_pop * 0.1  # Assume major cities are ~10% of country
                
                city_cases = int(city_data['total_cases'] * 0.15)  # Major cities ~15% of country cases
                
                # Risk level calculation
                cases_per_100k = (city_cases / city_pop) * 100000
                
                if cases_per_100k > 5000:
                    color = 'red'
                    risk = 'High'
                elif cases_per_100k > 2000:
                    color = 'orange'
                    risk = 'Medium'
                else:
                    color = 'green'
                    risk = 'Low'
                
                radius = max(8, min(25, np.log10(city_cases + 1) * 4))
                
                folium.CircleMarker(
                    location=[city["lat"], city["lon"]],
                    radius=radius,
                    popup=f"""
                    <b>{city['name']}, {city_country}</b><br>
                    Disease: {disease}<br>
                    Estimated Cases: {city_cases:,}<br>
                    Cases per 100K: {cases_per_100k:.0f}<br>
                    Risk Level: {risk}
                    """,
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7,
                    weight=2
                ).add_to(world_map)
        
        # Display map
        map_data = st_folium(world_map, width=700, height=500)
        
        st.markdown("""
        <div class="graph-explanation">
        <strong>🗺️ Map Analysis:</strong> Circle sizes represent case volumes, colors indicate risk levels. 
        Red = High risk (>5K cases/100K), Orange = Medium risk (2-5K/100K), Green = Low risk (<2K/100K).
        Click on circles for detailed city information.
        </div>
        """, unsafe_allow_html=True)
        
        # Regional analysis
        st.markdown("### 📍 Regional Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Top affected countries for selected disease
            country_data = []
            for c_name in list(COUNTRY_CONFIG.keys())[:8]:  # Top 8 countries
                c_data = EnhancedDataFetcher.fetch_disease_data(c_name, disease)
                if c_data['total_cases'] > 0:
                    country_data.append({
                        'Country': c_name,
                        'Cases': c_data['total_cases'],
                        'Deaths': c_data['deaths'],
                        'Cases per 100K': (c_data['total_cases'] / c_data['population']) * 100000
                    })
            
            if country_data:
                df_countries = pd.DataFrame(country_data)
                df_countries = df_countries.sort_values('Cases', ascending=False)
                
                fig_countries = px.bar(
                    df_countries.head(6), 
                    x='Country', 
                    y='Cases',
                    title=f"Top Affected Countries - {disease}",
                    color='Cases',
                    color_continuous_scale='Reds'
                )
                
                st.plotly_chart(fig_countries, use_container_width=True)
        
        with col2:
            # Disease severity comparison
            severity_data = []
            for d_name, d_params in DISEASE_CONFIG.items():
                d_data = EnhancedDataFetcher.fetch_disease_data(country, d_name)
                severity_data.append({
                    'Disease': d_name,
                    'R₀': d_params['basic_r0'],
                    'Fatality Rate': d_params['death_rate'] * 100,
                    'Current Cases': d_data['active_cases']
                })
            
            df_severity = pd.DataFrame(severity_data)
            
            fig_severity = px.scatter(
                df_severity, 
                x='R₀', 
                y='Fatality Rate',
                size='Current Cases',
                color='Disease',
                title="Disease Severity Matrix",
                hover_data=['Current Cases']
            )
            
            st.plotly_chart(fig_severity, use_container_width=True)
        
        st.markdown("""
        <div class="graph-explanation">
        <strong>📊 Regional Insights:</strong> Left chart shows countries with highest case burdens. 
        Right chart plots disease severity: higher R₀ means more contagious, higher fatality rate means more deadly. 
        Bubble size represents current active cases in your selected region.
        </div>
        """, unsafe_allow_html=True)
    
    # TAB 3: Health Assessment
    with tab3:
        st.markdown("## 🩺 AI-Powered Health Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"### 🔍 {disease} Symptom Checker")
            
            # Disease-specific symptoms
            disease_symptoms = {
                "COVID-19": {
                    "primary": ["Fever (>38°C)", "Persistent cough", "Shortness of breath", "Loss of taste/smell"],
                    "secondary": ["Headache", "Muscle aches", "Sore throat", "Fatigue", "Diarrhea"]
                },
                "Mpox": {
                    "primary": ["Fever", "Distinctive rash/lesions", "Swollen lymph nodes", "Headache"],
                    "secondary": ["Muscle aches", "Back pain", "Exhaustion", "Chills"]
                },
                "Influenza": {
                    "primary": ["Sudden onset fever", "Severe headache", "Muscle aches", "Dry cough"],
                    "secondary": ["Runny nose", "Sore throat", "Fatigue", "Nausea"]
                },
                "H5N1 Bird Flu": {
                    "primary": ["High fever (>39°C)", "Severe cough", "Shortness of breath", "Diarrhea"],
                    "secondary": ["Headache", "Muscle aches", "Abdominal pain", "Vomiting"]
                },
                "Dengue": {
                    "primary": ["High fever", "Severe headache", "Eye pain", "Muscle/joint pain"],
                    "secondary": ["Nausea", "Vomiting", "Skin rash", "Bleeding"]
                }
            }
            
            symptoms = disease_symptoms.get(disease, disease_symptoms["COVID-19"])
            
            st.markdown("**Primary Symptoms:**")
            primary_score = 0
            for symptom in symptoms["primary"]:
                if st.checkbox(symptom, key=f"primary_{symptom}"):
                    primary_score += 3
            
            st.markdown("**Secondary Symptoms:**")
            secondary_score = 0
            for symptom in symptoms["secondary"]:
                if st.checkbox(symptom, key=f"secondary_{symptom}"):
                    secondary_score += 1
            
            # Risk factors
            st.markdown("**Risk Factors:**")
            risk_score = 0
            if st.checkbox("Recent travel to affected area"):
                risk_score += 2
            if st.checkbox("Contact with confirmed case"):
                risk_score += 3
            if st.checkbox("Healthcare worker"):
                risk_score += 1
            if st.checkbox("Immunocompromised"):
                risk_score += 2
            
            total_score = primary_score + secondary_score + risk_score
            
            if st.button("🔍 AI Risk Assessment", type="primary"):
                # Risk assessment
                if total_score >= 12:
                    st.markdown('<div class="risk-high">🚨 VERY HIGH RISK - Seek immediate medical attention</div>', unsafe_allow_html=True)
                    st.error("Your symptoms strongly suggest possible infection. Contact healthcare immediately.")
                elif total_score >= 7:
                    st.markdown('<div class="risk-medium">⚠️ MODERATE RISK - Medical consultation recommended</div>', unsafe_allow_html=True)
                    st.warning("Some concerning symptoms present. Consider getting tested and consulting a doctor.")
                elif total_score >= 3:
                    st.markdown('<div class="risk-medium">⚠️ LOW-MODERATE RISK - Monitor symptoms</div>', unsafe_allow_html=True)
                    st.info("Mild symptoms detected. Continue monitoring and follow prevention guidelines.")
                else:
                    st.markdown('<div class="risk-low">✅ LOW RISK - Continue precautions</div>', unsafe_allow_html=True)
                    st.success("No significant symptoms detected. Maintain standard health precautions.")
                
                # Save to CSV
                symptom_data = {
                    'disease': disease,
                    'country': country,
                    'primary_score': primary_score,
                    'secondary_score': secondary_score,
                    'risk_score': risk_score,
                    'total_score': total_score,
                    'risk_level': 'Very High' if total_score >= 12 else 'Moderate' if total_score >= 7 else 'Low-Moderate' if total_score >= 3 else 'Low'
                }
                
                st.session_state.csv_manager.add_symptom_report(symptom_data)
        
        with col2:
            st.markdown("### 📊 Disease Information")
            
            # Disease-specific information
            disease_info = {
                "COVID-19": {
                    "description": "Respiratory illness caused by SARS-CoV-2 virus",
                    "transmission": "Airborne droplets, close contact, contaminated surfaces",
                    "prevention": "Vaccination, masks, social distancing, hand hygiene"
                },
                "Mpox": {
                    "description": "Viral disease with distinctive skin lesions",
                    "transmission": "Close contact, respiratory droplets, contaminated materials",
                    "prevention": "Avoid close contact with infected persons, safe practices"
                },
                "Influenza": {
                    "description": "Seasonal respiratory infection",
                    "transmission": "Airborne droplets, contaminated surfaces",
                    "prevention": "Annual vaccination, hand hygiene, avoid crowded places"
                },
                "H5N1 Bird Flu": {
                    "description": "Avian influenza with pandemic potential",
                    "transmission": "Contact with infected birds, human-to-human rare",
                    "prevention": "Avoid contact with birds, cook poultry thoroughly"
                },
                "Dengue": {
                    "description": "Mosquito-borne viral infection",
                    "transmission": "Aedes mosquito bites",
                    "prevention": "Mosquito control, eliminate standing water"
                }
            }
            
            info = disease_info.get(disease, disease_info["COVID-19"])
            
            st.info(f"**About {disease}:**\n\n{info['description']}")
            st.warning(f"**Transmission:** {info['transmission']}")
            st.success(f"**Prevention:** {info['prevention']}")
            
            # Current statistics
            st.markdown("### 📈 Current Statistics")
            st.metric("Global Cases", f"{current_data['total_cases']:,}")
            st.metric("Case Fatality Rate", f"{mortality_rate:.2f}%")
            st.metric("Basic R₀", f"{disease_params['basic_r0']}")
    
    # TAB 4: Vaccination System
    with tab4:
        st.markdown("## 💉 Vaccination Management System")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### 🏥 {disease} Vaccination Centers in India")
            
            vaccination_centers = get_vaccination_centers_by_disease(disease)
            
            # Create map of vaccination centers
            vacc_map = folium.Map(location=[20.5937, 78.9629], zoom_start=5)  # Center of India
            
            for center in vaccination_centers:
                # Availability color coding
                color = {'High': 'green', 'Medium': 'orange', 'Low': 'red'}[center['availability']]
                
                folium.Marker(
                    location=[center["lat"], center["lon"]],
                    popup=f"""
                    <b>{center['name']}</b><br>
                    📍 {center['address']}<br>
                    📞 {center['phone']}<br>
                    💉 Vaccines: {', '.join(center['available_vaccines'])}<br>
                    📊 Availability: {center['availability']}<br>
                    ⏰ {center['timing']}
                    """,
                    icon=folium.Icon(color=color, icon='plus-sign')
                ).add_to(vacc_map)
            
            st_folium(vacc_map, width=700, height=400)
            
            st.markdown("""
            <div class="graph-explanation">
            <strong>🗺️ Vaccination Centers:</strong> Green markers indicate high vaccine availability, 
            orange shows medium availability, red indicates low availability. Click markers for detailed information.
            </div>
            """, unsafe_allow_html=True)
            
            # Booking form
            st.markdown("### 📝 Book Vaccination Appointment")
            
            with st.form("vaccination_booking"):
                form_col1, form_col2 = st.columns(2)
                
                with form_col1:
                    name = st.text_input("Full Name *", placeholder="Enter your full name")
                    phone = st.text_input("Phone Number *", placeholder="+91 XXXXXXXXXX")
                    email = st.text_input("Email Address", placeholder="your.email@example.com")
                    age = st.number_input("Age", min_value=1, max_value=120, value=25)
                
                with form_col2:
                    selected_center = st.selectbox("Select Center *", 
                                                 [f"{c['name']} - {c['city']}" for c in vaccination_centers])
                    vaccine_preference = st.selectbox("Vaccine Preference", 
                                                    ["Any Available"] + DISEASE_CONFIG[disease]["vaccines"])
                    preferred_date = st.date_input("Preferred Date", 
                                                 min_value=datetime.now().date() + timedelta(days=1))
                    preferred_time = st.selectbox("Preferred Time", 
                                                ["9:00-11:00 AM", "11:00-1:00 PM", "2:00-4:00 PM", "4:00-6:00 PM"])
                
                submitted = st.form_submit_button("📅 Book Appointment", type="primary")
                
                if submitted:
                    if name and phone and selected_center:
                        # Find selected center details
                        center_details = None
                        for center in vaccination_centers:
                            if f"{center['name']} - {center['city']}" == selected_center:
                                center_details = center
                                break
                        
                        booking_data = {
                            'name': name,
                            'phone': phone,
                            'email': email,
                            'age': age,
                            'disease': disease,
                            'vaccine_type': vaccine_preference,
                            'date': preferred_date,
                            'time': preferred_time,
                            'center_name': center_details['name'] if center_details else '',
                            'center_address': center_details['address'] if center_details else ''
                        }
                        
                        # Add to CSV storage
                        st.session_state.csv_manager.add_vaccination_booking(booking_data)
                        
                        st.success(f"🎉 Booking confirmed for {name}!")
                        st.balloons()
                        
                        # Show booking confirmation
                        st.markdown(f"""
                        <div class="success-card">
                        <strong>📋 Booking Confirmation</strong><br>
                        Patient: {name}<br>
                        Center: {selected_center}<br>
                        Date: {preferred_date}<br>
                        Time: {preferred_time}<br>
                        Vaccine: {vaccine_preference}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.error("Please fill in all required fields marked with *")
        
        with col2:
            st.markdown(f"### 📊 {disease} Vaccination Statistics")
            
            # Disease-specific vaccination data
            population = current_data['population']
            
            if disease == "COVID-19":
                vaccinated_once = int(population * 0.75)
                fully_vaccinated = int(population * 0.65)
                booster_doses = int(population * 0.30)
            elif disease == "Influenza":
                vaccinated_once = int(population * 0.45)  # Annual flu vaccination rate
                fully_vaccinated = vaccinated_once  # Single dose for flu
                booster_doses = 0
            else:
                # Other diseases - lower vaccination rates
                vaccinated_once = int(population * 0.10)
                fully_vaccinated = int(population * 0.08)
                booster_doses = 0
            
            unvaccinated = population - vaccinated_once
            
            # Vaccination chart
            vacc_data = {
                "Status": ["Unvaccinated", "First Dose", "Fully Vaccinated", "Booster"],
                "Count": [unvaccinated, vaccinated_once, fully_vaccinated, booster_doses]
            }
            
            fig_vacc = px.pie(
                values=vacc_data["Count"],
                names=vacc_data["Status"],
                title=f"{disease} Vaccination Coverage - {country}",
                color_discrete_sequence=['#ff6b6b', '#fdcb6e', '#00b894', '#74b9ff']
            )
            
            st.plotly_chart(fig_vacc, use_container_width=True)
            
            st.markdown("""
            <div class="graph-explanation">
            <strong>💉 Vaccination Analysis:</strong> Shows vaccination coverage distribution. 
            Higher fully vaccinated percentages indicate better population immunity.
            </div>
            """, unsafe_allow_html=True)
            
            # Vaccination metrics
            vaccination_rate = (fully_vaccinated / population) * 100
            st.metric("Vaccination Coverage", f"{vaccination_rate:.1f}%")
            
            # Herd immunity threshold
            herd_immunity_threshold = (1 - 1/disease_params["basic_r0"]) * 100
            st.metric("Herd Immunity Threshold", f"{herd_immunity_threshold:.1f}%")
            
            if vaccination_rate >= herd_immunity_threshold:
                st.success("✅ Herd immunity likely achieved")
            else:
                remaining = herd_immunity_threshold - vaccination_rate
                st.warning(f"⚠️ Need {remaining:.1f}% more coverage for herd immunity")
    
    # TAB 5: Alert System
    with tab5:
        st.markdown("## ⚠️ Smart Alert & Monitoring System")
        
        # Alert configuration
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔔 Alert Configuration")
            
            case_threshold = st.number_input("Daily case increase alert:", min_value=10, value=100, step=10)
            death_threshold = st.number_input("Daily death increase alert:", min_value=1, value=10, step=1)
            
            # Current alert status
            current_daily_cases = current_data['daily_cases']
            current_daily_deaths = current_data['daily_deaths']
            
            if current_daily_cases > case_threshold:
                st.markdown('<div class="alert-card">🚨 CASE ALERT: Daily cases exceed threshold</div>', unsafe_allow_html=True)
            
            if current_daily_deaths > death_threshold:
                st.markdown('<div class="alert-card">💀 DEATH ALERT: Daily deaths exceed threshold</div>', unsafe_allow_html=True)
            
            # Risk level for country
            r_effective = disease_params["transmission_rate"] / disease_params["recovery_rate"]
            
            if r_effective > 1.5:
                st.markdown('<div class="alert-card">📈 HIGH TRANSMISSION RISK</div>', unsafe_allow_html=True)
            elif r_effective > 1.0:
                st.markdown('<div class="risk-medium">⚠️ MODERATE TRANSMISSION RISK</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="risk-low">✅ LOW TRANSMISSION RISK</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown("### 📋 Recent Alerts")
            
            # Generate realistic alerts based on current data
            recent_alerts = []
            
            if current_data['daily_cases'] > 1000:
                recent_alerts.append({
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "type": "Case Surge",
                    "message": f"{current_data['daily_cases']:,} new cases reported in {country}",
                    "severity": "High"
                })
            
            if current_data['active_cases'] > current_data['population'] * 0.001:  # > 0.1% of population
                recent_alerts.append({
                    "date": (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"),
                    "type": "High Prevalence",
                    "message": f"Active cases exceed 0.1% of population in {country}",
                    "severity": "Medium"
                })
            
            if len(recent_alerts) == 0:
                recent_alerts.append({
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "type": "Status Update",
                    "message": f"Situation stable in {country} for {disease}",
                    "severity": "Low"
                })
            
            for alert in recent_alerts:
                severity_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}[alert["severity"]]
                st.markdown(f"**{alert['date']}** {severity_color} - {alert['type']}: {alert['message']}")
    
    # TAB 6: SIR Modeling
    with tab6:
        st.markdown("## 📊 Advanced SIR Disease Modeling")
        
        # Model parameters
        st.markdown("### ⚙️ Model Parameters")
        
        param_col1, param_col2, param_col3 = st.columns(3)
        
        with param_col1:
            st.markdown("**Population Parameters:**")
            N = current_data['population']
            st.metric("👥 Population", f"{N:,}")
            I0 = st.number_input("Initial Infected:", min_value=1, 
                               value=max(1, current_data['active_cases']), step=1)
            R0 = st.number_input("Initial Recovered:", min_value=0, 
                               value=current_data['recovered'], step=1)
        
        with param_col2:
            st.markdown("**Disease Parameters:**")
            beta = st.number_input("Transmission Rate (β):", 
                                 min_value=0.01, max_value=2.0, 
                                 value=disease_params["transmission_rate"], step=0.01)
            gamma = st.number_input("Recovery Rate (γ):", 
                                  min_value=0.01, max_value=1.0, 
                                  value=disease_params["recovery_rate"], step=0.01)
            delta = st.number_input("Death Rate (δ):", 
                                  min_value=0.001, max_value=0.1, 
                                  value=disease_params["death_rate"], step=0.001)
        
        with param_col3:
            st.markdown("**Simulation Settings:**")
            simulation_days = st.number_input("Simulation Days:", min_value=30, max_value=730, value=365)
            
            # Calculate R₀
            R_basic = beta / gamma if gamma > 0 else 0
            st.metric("🔬 Basic R₀", f"{R_basic:.2f}")
            
            if R_basic > 1:
                st.warning("⚠️ R₀ > 1: Epidemic will spread")
            else:
                st.success("✅ R₀ < 1: Epidemic will decline")
        
        # Run SIR simulation
        if st.button("▶️ Run SIR Simulation", type="primary"):
            with st.spinner("Running epidemiological simulation..."):
                sir_model = SIRModel()
                S_data, I_data, R_data, D_data = sir_model.run_sir_simulation(N, I0, R0, beta, gamma, delta, simulation_days)
                
                # Create comprehensive SIR visualization
                fig_sir = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("SIR Model Results", "R-effective Over Time", "Attack Rate", "Healthcare Burden"),
                    specs=[[{"secondary_y": True}, {"type": "scatter"}],
                           [{"type": "scatter"}, {"type": "bar"}]]
                )
                
                days = list(range(len(S_data)))
                
                # Main SIR plot
                fig_sir.add_trace(go.Scatter(x=days, y=S_data, name="Susceptible", line=dict(color='blue')), row=1, col=1)
                fig_sir.add_trace(go.Scatter(x=days, y=I_data, name="Infected", line=dict(color='red')), row=1, col=1)
                fig_sir.add_trace(go.Scatter(x=days, y=R_data, name="Recovered", line=dict(color='green')), row=1, col=1)
                fig_sir.add_trace(go.Scatter(x=days, y=D_data, name="Deaths", line=dict(color='black')), row=1, col=1)
                
                # R-effective calculation
                r_eff_data = []
                for i in range(len(S_data)):
                    if N > 0 and gamma > 0:
                        r_eff = beta * S_data[i] / (N * gamma)
                        r_eff_data.append(min(5, max(0, r_eff)))
                    else:
                        r_eff_data.append(1.0)
                
                fig_sir.add_trace(go.Scatter(x=days, y=r_eff_data, name="R-effective", line=dict(color='purple')), row=1, col=2)
                fig_sir.add_hline(y=1, line_dash="dash", line_color="red", row=1, col=2)
                
                # Attack rate
                attack_rates = [(N - S) / N * 100 for S in S_data]
                fig_sir.add_trace(go.Scatter(x=days, y=attack_rates, name="Attack Rate (%)", line=dict(color='orange')), row=2, col=1)
                
                # Healthcare burden (5% of infected need hospitalization)
                hospital_burden = [I * 0.05 for I in I_data]
                fig_sir.add_trace(go.Bar(x=days[::7], y=hospital_burden[::7], name="Hospital Beds Needed", marker_color='darkred'), row=2, col=2)
                
                fig_sir.update_layout(height=800, showlegend=True, title_text=f"SIR Model Analysis - {disease} in {country}")
                
                st.plotly_chart(fig_sir, use_container_width=True)
                
                st.markdown("""
                <div class="graph-explanation">
                <strong>📊 SIR Model Explanation:</strong> Shows disease progression over time. 
                Susceptible (blue) decreases as people get infected. Infected (red) peaks then declines. 
                Recovered (green) and Deaths (black) accumulate. R-effective below 1 indicates epidemic control.
                </div>
                """, unsafe_allow_html=True)
                
                # Model analytics
                st.markdown("### 📈 Model Analytics")
                
                analytics_col1, analytics_col2, analytics_col3, analytics_col4 = st.columns(4)
                
                with analytics_col1:
                    peak_infections = max(I_data)
                    peak_day = I_data.index(peak_infections)
                    st.metric("🔴 Peak Infections", f"{int(peak_infections):,}", f"Day {peak_day}")
                
                with analytics_col2:
                    final_attack_rate = attack_rates[-1]
                    st.metric("🎯 Final Attack Rate", f"{final_attack_rate:.1f}%")
                
                with analytics_col3:
                    final_deaths = D_data[-1]
                    final_cfr = (final_deaths / (R_data[-1] + final_deaths)) * 100 if (R_data[-1] + final_deaths) > 0 else 0
                    st.metric("💀 Case Fatality Rate", f"{final_cfr:.2f}%")
                
                with analytics_col4:
                    max_hospital_burden = max(hospital_burden)
                    st.metric("🏥 Max Hospital Need", f"{int(max_hospital_burden):,}")
                
                # Policy recommendations
                st.markdown("### 🎯 Policy Recommendations")
                
                if R_basic > 1.5:
                    st.markdown('<div class="alert-card">🚨 Immediate intervention required: R₀ > 1.5</div>', unsafe_allow_html=True)
                    st.markdown("- Implement strict social distancing measures")
                    st.markdown("- Increase testing and contact tracing")
                    st.markdown("- Prepare healthcare system for surge")
                elif R_basic > 1.0:
                    st.markdown('<div class="risk-medium">⚠️ Enhanced measures needed: R₀ > 1.0</div>', unsafe_allow_html=True)
                    st.markdown("- Implement moderate interventions")
                    st.markdown("- Monitor situation closely")
                    st.markdown("- Increase public awareness")
                else:
                    st.markdown('<div class="risk-low">✅ Situation under control: R₀ < 1.0</div>', unsafe_allow_html=True)
                    st.markdown("- Maintain current measures")
                    st.markdown("- Prepare for potential resurgence")
                    st.markdown("- Focus on vaccination if available")
                
                # Save model results
                model_data = {
                    'disease': disease,
                    'country': country,
                    'peak_infections': int(peak_infections),
                    'peak_day': peak_day,
                    'final_attack_rate': final_attack_rate,
                    'r_basic': R_basic,
                    'max_hospital_burden': int(max_hospital_burden),
                    'simulation_days': simulation_days
                }
                
                st.session_state.csv_manager.add_model_result(model_data)
    
    # CSV Download Section
    st.markdown("---")
    st.markdown("## 📥 Data Export & Management")
    
    download_col1, download_col2, download_col3 = st.columns(3)
    
    with download_col1:
        if st.button("📋 Download Vaccination Bookings"):
            csv_data = st.session_state.csv_manager.get_vaccination_csv()
            st.download_button(
                label="💾 Download Vaccination Data",
                data=csv_data,
                file_name=f"vaccination_bookings_{disease}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    with download_col2:
        if st.button("📊 Download Symptom Reports"):
            if st.session_state.csv_manager.symptom_reports:
                df_symptoms = pd.DataFrame(st.session_state.csv_manager.symptom_reports)
                csv_symptoms = df_symptoms.to_csv(index=False)
                st.download_button(
                    label="💾 Download Symptom Data",
                    data=csv_symptoms,
                    file_name=f"symptom_reports_{disease}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No symptom reports to download")
    
    with download_col3:
        if st.button("🔬 Download Model Results"):
            if st.session_state.csv_manager.model_results:
                df_models = pd.DataFrame(st.session_state.csv_manager.model_results)
                csv_models = df_models.to_csv(index=False)
                st.download_button(
                    label="💾 Download Model Data",
                    data=csv_models,
                    file_name=f"model_results_{disease}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No model results to download")

if __name__ == "__main__":
    main()

