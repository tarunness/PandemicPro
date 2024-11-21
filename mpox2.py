import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def neo_dynamic_sird_model(N, I0, R0, D0, beta_0, beta_hr, f_hr, gamma, delta, num_days):
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
        S_new = S - beta_eff * S * I / N
        I_new = I + beta_eff * S * I / N - gamma * I - delta * I
        R_new = R + gamma * I
        D_new = D + delta * I

        S, I, R, D = S_new, I_new, R_new, D_new

        susceptible.append(S)
        infected.append(I)
        recovered.append(R)
        deceased.append(D)

    return susceptible, infected, recovered, deceased

# Function for the SIR Model
def sir_model(N, I0, R0, beta, gamma, num_days):
    S = N - I0 - R0
    I = I0
    R = R0

    susceptible = [S]
    infected = [I]
    recovered = [R]

    for _ in range(num_days):
        S_new = S - beta * S * I / N
        I_new = I + beta * S * I / N - gamma * I
        R_new = R + gamma * I

        S, I, R = S_new, I_new, R_new

        susceptible.append(S)
        infected.append(I)
        recovered.append(R)

    return susceptible, infected, recovered

# Function for the SIRD Model
def sird_model(N, I0, R0, beta, gamma, delta, num_days):
    S = N - I0 - R0
    I = I0
    R = R0
    D = 0

    susceptible = [S]
    infected = [I]
    recovered = [R]
    deceased = [D]

    for _ in range(num_days):
        S_new = S - beta * S * I / N
        I_new = I + beta * S * I / N - gamma * I - delta * I
        R_new = R + gamma * I
        D_new = D + delta * I

        S, I, R, D = S_new, I_new, R_new, D_new

        susceptible.append(S)
        infected.append(I)
        recovered.append(R)
        deceased.append(D)

    return susceptible, infected, recovered, deceased

# Function for the SIS Model
def sis_model(N, I0, beta, gamma, num_days):
    S = N - I0
    I = I0

    susceptible = [S]
    infected = [I]

    for _ in range(num_days):
        S_new = S - beta * S * I / N + gamma * I
        I_new = I + beta * S * I / N - gamma * I

        S, I = S_new, I_new

        susceptible.append(S)
        infected.append(I)

    return susceptible, infected

# Function for the SEIR Model
def seir_model(N, E0, I0, R0, beta, gamma, alpha, num_days):
    S = N - E0 - I0 - R0
    E = E0
    I = I0
    R = R0

    susceptible = [S]
    exposed = [E]
    infected = [I]
    recovered = [R]

    for _ in range(num_days):
        S_new = S - beta * S * I / N
        E_new = E + beta * S * I / N - alpha * E
        I_new = I + alpha * E - gamma * I
        R_new = R + gamma * I

        S, E, I, R = S_new, E_new, I_new, R_new

        susceptible.append(S)
        exposed.append(E)
        infected.append(I)
        recovered.append(R)

    return susceptible, exposed, infected, recovered

# Streamlit page setup
st.title("Epidemiological Models")
model_option = st.sidebar.selectbox(
    "Choose a model",
    ("SIR Model", "SIRD Model", "SIS Model", "SEIR Model", "Neo-Dynamic Model")
)

# Common inputs
location = st.text_input("Enter the name of the Region:", "Example Region")
N = st.number_input("Enter the total population:", min_value=1, value=1000000)
I0 = st.number_input("Enter the initial number of infected individuals:", min_value=0, value=1)
num_days = st.slider("Simulation duration (days):", min_value=1, max_value=500, value=300)

# Model-specific inputs and plots
if model_option == "SIR Model":
    R0 = st.number_input("Enter the initial number of recovered individuals:", min_value=0, value=0)
    beta = st.number_input("Enter the infection rate (beta):", min_value=0.0, value=0.3)
    gamma = st.number_input("Enter the recovery rate (gamma):", min_value=0.0, value=0.1)
    
    susceptible, infected, recovered = sir_model(N, I0, R0, beta, gamma, num_days)
    
    # Convert to DataFrame for plotting
    data = pd.DataFrame({
        "Susceptible": susceptible,
        "Infected": infected,
        "Recovered": recovered
    })
    
    # Plotting
    st.line_chart(data)

elif model_option == "SIRD Model":
    R0 = st.number_input("Enter the initial number of recovered individuals:", min_value=0, value=0)
    beta = st.number_input("Enter the infection rate (beta):", min_value=0.0, value=0.3)
    gamma = st.number_input("Enter the recovery rate (gamma):", min_value=0.0, value=0.1)
    delta = st.number_input("Enter the death rate (delta):", min_value=0.0, value=0.01)
    
    susceptible, infected, recovered, deceased = sird_model(N, I0, R0, beta, gamma, delta, num_days)
    
    # Convert to DataFrame for plotting
    data = pd.DataFrame({
        "Susceptible": susceptible,
        "Infected": infected,
        "Recovered": recovered,
        "Deceased": deceased
    })
    
    # Plotting
    st.line_chart(data)

elif model_option == "SIS Model":
    beta = st.number_input("Enter the infection rate (beta):", min_value=0.0, value=0.3)
    gamma = st.number_input("Enter the recovery rate (gamma):", min_value=0.0, value=0.1)
    
    susceptible, infected = sis_model(N, I0, beta, gamma, num_days)
    
    # Convert to DataFrame for plotting
    data = pd.DataFrame({
        "Susceptible": susceptible,
        "Infected": infected
    })
    
    # Plotting
    st.line_chart(data)

elif model_option == "SEIR Model":
    E0 = st.number_input("Enter the initial number of exposed individuals:", min_value=0, value=0)
    R0 = st.number_input("Enter the initial number of recovered individuals:", min_value=0, value=0)
    beta = st.number_input("Enter the infection rate (beta):", min_value=0.0, value=0.3)
    gamma = st.number_input("Enter the recovery rate (gamma):", min_value=0.0, value=0.1)
    alpha = st.number_input("Enter the rate at which exposed individuals become infectious (alpha):", min_value=0.0, value=0.2)
    
    susceptible, exposed, infected, recovered = seir_model(N, E0, I0, R0, beta, gamma, alpha, num_days)
    
    # Convert to DataFrame for plotting
    data = pd.DataFrame({
        "Susceptible": susceptible,
        "Exposed": exposed,
        "Infected": infected,
        "Recovered": recovered
    })
    
    # Plotting
    st.line_chart(data)

elif model_option == "Neo-Dynamic Model":
    R0 = st.number_input("Enter the initial number of recovered individuals:", min_value=0, value=0)
    beta_0 = st.number_input("Enter the baseline infection rate:", min_value=0.0, value=0.3)
    beta_hr = st.number_input("Enter the high-risk infection rate:", min_value=0.0, value=0.5)
    f_hr = st.number_input("Enter the fraction of high-risk individuals:", min_value=0.0, max_value=1.0, value=0.1)
    gamma = st.number_input("Enter the recovery rate:", min_value=0.0, value=0.1)
    
    susceptible, infected, recovered = neo_dynamic_model(N, I0, R0, beta_0, beta_hr, f_hr, gamma, num_days)
    
    # Convert to DataFrame for plotting
    data = pd.DataFrame({
        "Susceptible": susceptible,
        "Infected": infected,


        "Recovered": recovered
    })
    
    # Plotting
    st.line_chart(data)
