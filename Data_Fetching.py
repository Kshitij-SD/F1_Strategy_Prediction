import fastf1
import pandas as pd
import numpy as np
import os

# Set up cache directory
cache_dir = r'C:\Users\Kshitij\Downloads\F1\final_data_cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

fastf1.Cache.enable_cache(cache_dir)

circuit_lengths = {
    'Bahrain Grand Prix': 5412,
    'Saudi Arabian Grand Prix': 6174,
    'Australian Grand Prix': 5303,
    'Emilia Romagna Grand Prix': 4909,
    'Miami Grand Prix': 5412,
    'Spanish Grand Prix': 4657,
    'Monaco Grand Prix': 3337,
    'Canadian Grand Prix': 4361,
    'Austrian Grand Prix': 4318,
    'British Grand Prix': 5891,
    'Hungarian Grand Prix': 4381,
    'Belgian Grand Prix': 7004,
    'Dutch Grand Prix': 4259,
    'Italian Grand Prix': 5793,
    'Singapore Grand Prix': 5063,
    'Japanese Grand Prix': 5807,
    'United States Grand Prix': 5513,
    'Mexico City Grand Prix': 4304,
    'São Paulo Grand Prix': 4309,
    'Abu Dhabi Grand Prix': 5281,
}

def calculate_degradation(lap_times):
    if len(lap_times) < 2:
        return None, None

    lap_times_sec = [lt.total_seconds() for lt in lap_times if pd.notnull(lt)]
    if len(lap_times_sec) < 2:
        return None, None

    x = np.arange(len(lap_times_sec))
    y = np.array(lap_times_sec)

    slope, bias = np.polyfit(x, y, 1)
    
    return slope, bias

def load_f1_data(years=None, races=None):
    if years is None:
        years = [2020, 2021, 2022, 2023, 2024]
    if races is None:
        races = list(range(1, 25))

    all_data = []

    for year in years:
        for round in races:
            try:
                print(f"\nLoading {year} Round {round}...")

                try:
                    session = fastf1.get_session(year, round, 'R')
                    session.load(telemetry=False, weather=True, messages=False)
                except Exception as e:
                    print(f"Failed to load session {year} R{round}: {e}")
                    continue

                event_name = session.event.get('EventName', f"Round {round}")
                print(f"Event: {event_name}")

                circuit_length = session.event.get('CircuitLength', None)
                if circuit_length is None or pd.isna(circuit_length):
                    circuit_length = circuit_lengths.get(event_name, 5000)

                total_laps = session.total_laps
                results = session.results
                if results is None or results.empty:
                    print(f"No race results for {event_name} ({year} R{round})")
                    continue

                # Weather Info
                weather = session.weather_data
                avg_track_temp = weather['TrackTemp'].mean() if 'TrackTemp' in weather.columns else None
                avg_air_temp = weather['AirTemp'].mean() if 'AirTemp' in weather.columns else None
                avg_humidity = weather['Humidity'].mean() if 'Humidity' in weather.columns else None
                rainfall = 1 if ('Rainfall' in weather.columns and (weather['Rainfall'] > 0).any()) else 0

                print(f"TrackTemp: {avg_track_temp}, Rainfall: {rainfall}")

                # Process laps to detect Safety Car periods
                all_laps = session.laps
                if not all_laps.empty and 'TrackStatus' in all_laps.columns:
                    sc_flags = all_laps['TrackStatus'].fillna(1)
                    safety_car = 1 if any(sc_flags.isin([4, 5])) else 0
                else:
                    print(f"No TrackStatus data for {event_name} ({year} R{round})")
                    safety_car = 0

                print(f"Safety Car deployed: {safety_car}")

                # Process each driver
                for idx, driver_result in results.iterrows():
                    driver = driver_result['Abbreviation']
                    grid_pos = int(driver_result['GridPosition'])
                    final_pos = int(driver_result['Position'])
                    team = driver_result['TeamName']
                    driver_name = driver_result['FullName']

                    driver_laps = all_laps.pick_driver(driver)
                    if driver_laps.empty:
                        print(f"No laps for driver {driver} in {year} R{round}")
                        continue

                    stints = driver_laps[['LapNumber', 'Stint', 'Compound', 'LapTime']].copy()

                    stints = stints.groupby('Stint').agg({
                        'LapNumber': ['count', 'min', 'max'],
                        'Compound': 'first',
                        'LapTime': lambda x: list(x)
                    }).reset_index()

                    for _, stint in stints.iterrows():
                        stint_len = stint['LapNumber']['count']
                        stint_start = stint['LapNumber']['min']
                        stint_end = stint['LapNumber']['max']
                        compound = stint['Compound']['first']
                        lap_times = stint['LapTime']['<lambda>']

                        avg_lap_time = (
                            np.mean([lt.total_seconds() for lt in lap_times if pd.notnull(lt)])
                            if lap_times else None
                        )

                        deg_slope, deg_bias = calculate_degradation(lap_times)

                        all_data.append({
                            'EventName': event_name,
                            'RoundNumber': round,
                            'EventYear': year,
                            'Team': team,
                            'Driver': driver_name,
                            'GridPosition': grid_pos,
                            'FinalPosition': final_pos,
                            'Compound': compound,
                            'StintLen': stint_len,
                            'CircuitLength': circuit_length,
                            'DesignedLaps': total_laps,
                            'StintStartLap': stint_start,
                            'StintEndLap': stint_end,
                            'AvgLapTime': avg_lap_time,
                            'TrackTemp': avg_track_temp,
                            'AirTemp': avg_air_temp,
                            'Humidity': avg_humidity,
                            'Rainfall': rainfall,
                            'DegradationSlope': deg_slope,
                            'DegradationBias': deg_bias,
                            'SafetyCar': safety_car
                        })

            except Exception as e:
                print(f"Unhandled error loading {year} R{round}: {e}")
                continue

    df = pd.DataFrame(all_data)

    if not df.empty:
        df['PositionsDelta'] = df['GridPosition'] - df['FinalPosition']

    return df

# Load data
df = load_f1_data(years=[2020, 2021, 2022, 2023, 2024])

print("\nSample data:")
print(df.head())
print(f"\nTotal records: {len(df)}")

df.to_csv("f1_stint_data_2020_2024.csv", index=False)
