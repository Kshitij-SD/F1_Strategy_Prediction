import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ------------------------------
# Load models and encoders
# ------------------------------
save_dir = r'C:\Users\Kshitij\Downloads\F1\saved_models'

# Load models (already saved previously)
stint_count_model = joblib.load(os.path.join(save_dir, 'stint_count_model'))

stint_len_models = [
    joblib.load(os.path.join(save_dir, 'stint_1_len_model')),
    joblib.load(os.path.join(save_dir, 'stint_2_len_model')),
    joblib.load(os.path.join(save_dir, 'stint_3_len_model')),
    joblib.load(os.path.join(save_dir, 'stint_4_len_model')),
]

stint_compound_models = [
    joblib.load(os.path.join(save_dir, 'stint_1_compound_model')),
    joblib.load(os.path.join(save_dir, 'stint_2_compound_model')),
    joblib.load(os.path.join(save_dir, 'stint_3_compound_model')),
    joblib.load(os.path.join(save_dir, 'stint_4_compound_model')),
]

le_event = joblib.load(os.path.join(save_dir, 'le_event.joblib')) 
le_compound = joblib.load(os.path.join(save_dir, 'le_compound.joblib'))

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🏁 F1 Pit Stop Strategy Predictor")

st.sidebar.header("Race Input Parameters")

# Collect input data from user
event_name = st.sidebar.selectbox("Event", le_event.classes_)

circuit_length = st.sidebar.number_input("Circuit Length (meters)", value=4309)
designed_laps = st.sidebar.number_input("Designed Laps", value=71)
track_temp = st.sidebar.number_input("Track Temp (°C)", value=51.56)
air_temp = st.sidebar.number_input("Air Temp (°C)", value=23.14)
humidity = st.sidebar.number_input("Humidity (%)", value=60.1)
rainfall = st.sidebar.number_input("Rainfall (mm)", value=0)
safety_car = st.sidebar.selectbox("Safety Car Deployed?", ['No', 'Yes'])
safety_car = 1 if safety_car == 'Yes' else 0

# Degradation
degradation_slopes = {
    'SOFT': st.sidebar.number_input("Degradation Slope (SOFT)", value=-23.0),
    'MEDIUM': st.sidebar.number_input("Degradation Slope (MEDIUM)", value=-1.5),
    'HARD': st.sidebar.number_input("Degradation Slope (HARD)", value=-0.8),
}

degradation_biases = {
    'SOFT': st.sidebar.number_input("Degradation Bias (SOFT)", value=94.2),
    'MEDIUM': st.sidebar.number_input("Degradation Bias (MEDIUM)", value=95.0),
    'HARD': st.sidebar.number_input("Degradation Bias (HARD)", value=83.0),
}

# Prediction Trigger
if st.sidebar.button("Predict Strategy"):

    # ------------------------------
    # Prepare Input DataFrame
    # ------------------------------
    new_race_data = {
        'CircuitLength': circuit_length,
        'DesignedLaps': designed_laps,
        'TrackTemp': track_temp,
        'AirTemp': air_temp,
        'Humidity': humidity,
        'Rainfall': rainfall,
        'SafetyCar': safety_car,
        'DegradationSlope_s': degradation_slopes['SOFT'],
        'DegradationBias_s': degradation_biases['SOFT'],
        'DegradationSlope_m': degradation_slopes['MEDIUM'],
        'DegradationBias_m': degradation_biases['MEDIUM'],
        'DegradationSlope_h': degradation_slopes['HARD'],
        'DegradationBias_h': degradation_biases['HARD'],
        'EventEncoded': le_event.transform([event_name])[0]
    }

    new_race_df = pd.DataFrame([new_race_data])

    # ------------------------------
    # Prediction Logic 
    # ------------------------------
    features_stint_num = ['CircuitLength', 'DesignedLaps','TrackTemp', 'AirTemp','EventEncoded'] 
    features_stint_compound = ['CircuitLength', 'cumulative_laps', 'TrackTemp', 'AirTemp','stint_num','EventEncoded','Humidity', 'Rainfall','SafetyCar']
    features_stint_length = ['CircuitLength', 'TrackTemp', 'AirTemp','prev_stint_length','EventEncoded','DegradationSlope', 'DegradationBias','DesignedLaps','Humidity', 'Rainfall','SafetyCar']
    
    degradation_mapping = {
        'SOFT': ('DegradationSlope_s', 'DegradationBias_s'),
        'MEDIUM': ('DegradationSlope_m', 'DegradationBias_m'),
        'HARD': ('DegradationSlope_h', 'DegradationBias_h')
    }

    X_stint_count = new_race_df[features_stint_num]
    predicted_total_stints = int(stint_count_model.predict(X_stint_count)[0])

    st.success(f"Predicted Total Stints: {predicted_total_stints}")

    cum_laps = 0
    prev_stint_len = 0
    predicted_stint_lengths = []
    predicted_compounds = []

    for stint_num in range(predicted_total_stints):
        new_race_df['stint_num'] = stint_num + 1
        new_race_df['cumulative_laps'] = cum_laps
        new_race_df['prev_stint_length'] = prev_stint_len

        compound_model = stint_compound_models[stint_num]
        stint_compound_encoded = compound_model.predict(new_race_df[features_stint_compound])[0]
        compound_name = le_compound.inverse_transform([stint_compound_encoded])[0]
        predicted_compounds.append(compound_name)

        deg_slope_col, deg_bias_col = degradation_mapping[compound_name]
        new_race_df['DegradationSlope'] = new_race_df[deg_slope_col]
        new_race_df['DegradationBias'] = new_race_df[deg_bias_col]
        new_race_df['CompoundEncoded'] = stint_compound_encoded

        length_model = stint_len_models[stint_num]
        stint_length = int(length_model.predict(new_race_df[features_stint_length])[0])

        if stint_length <= 0:
            stint_length = 1

        cum_laps += stint_length
        prev_stint_len = stint_length
        predicted_stint_lengths.append(stint_length)

    # Fix last stint
    total_laps = designed_laps
    laps_difference = total_laps - cum_laps

    if laps_difference != 0 and predicted_stint_lengths:
        predicted_stint_lengths[-1] += laps_difference
        cum_laps = total_laps

    # Enforce Different Compounds 
    unique_compounds = set(predicted_compounds)

    if len(unique_compounds) == 1:
        current_compound = predicted_compounds[0]
        longest_stint_index = predicted_stint_lengths.index(max(predicted_stint_lengths))
        shortest_stint_index = predicted_stint_lengths.index(min(predicted_stint_lengths))

        switch_made = False

        if current_compound != 'HARD':
            predicted_compounds[longest_stint_index] = 'HARD'
            switch_made = True

        if not switch_made and current_compound != 'SOFT':
            predicted_compounds[shortest_stint_index] = 'SOFT'
            switch_made = True

    # ------------------------------
    # Final Output
    # ------------------------------
    st.header("📋 Predicted Strategy")
    strategy_data = []
    for i, (compound, length) in enumerate(zip(predicted_compounds, predicted_stint_lengths), start=1):
        strategy_data.append({
            'Stint': i,
            'Compound': compound,
            'Length (laps)': length
        })

    strategy_df = pd.DataFrame(strategy_data)
    st.table(strategy_df)

    st.info(f"Total Laps Covered: {cum_laps} / {total_laps}")
