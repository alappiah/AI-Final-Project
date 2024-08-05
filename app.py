import pandas as pd
import requests
import numpy as np
import pickle
import streamlit as st
from pulp import LpProblem, LpVariable, lpSum, LpMaximize, LpStatus
import pymongo

# MongoDB connection parameters
mongodb_uri = "mongodb://localhost:27017/"  # Change if using a remote MongoDB instance
database_name = "fpl"
collection_name = "players"

# Connect to MongoDB
client = pymongo.MongoClient(mongodb_uri)
db = client[database_name]
collection = db[collection_name]

# Load the saved Random Forest model
with open('C:\\Users\\al200\\Downloads\\Intro to AI\\Final project\\gdr_model.pkl', 'rb') as model_file:
    rf_model = pickle.load(model_file)

# Step 1: Download and Prepare Data
def prepare_data():
    # Download the latest API data from FPL website
    fpl_events_url = 'https://fantasy.premierleague.com/api/bootstrap-static/'
    fpl_events_resp = requests.get(fpl_events_url)
    fpl_events_json = fpl_events_resp.json()

    # Convert to DataFrame and save to CSV
    df_elements = pd.DataFrame(fpl_events_json['elements'])
    df_elements.to_csv('elements.csv', index=False)

    # Load data from CSV
    df = pd.read_csv('elements.csv')

    # Create a 'Player' column and prepare the DataFrame for optimization
    df['Player'] = df['first_name'] + ' ' + df['second_name']
    df['now_cost'] = df['now_cost'] / 10
    df['total_points'] = df['total_points'] / 38  # Example conversion, replace with actual calculation if needed
    df = df[['Player', 'now_cost', 'total_points', 'team', 'element_type']]
    
    return df

# Function to add player constraints for a given gameweek
def add_player_constraints(model, df, player_vars, players, budget_min, budget_max):
    # Budget constraints
    model += lpSum(df['now_cost'][i] * player_vars[i] for i in players) <= budget_max
    model += lpSum(df['now_cost'][i] * player_vars[i] for i in players) >= budget_min

    # Position constraints
    model += lpSum(player_vars[i] for i in players if df['element_type'][i] == 1) == 1  # Goalkeeper
    model += lpSum(player_vars[i] for i in players if df['element_type'][i] == 2) <= 5  # Defenders
    model += lpSum(player_vars[i] for i in players if df['element_type'][i] == 2) >= 3  # Minimum 3 Defenders
    model += lpSum(player_vars[i] for i in players if df['element_type'][i] == 3) <= 5  # Midfielders
    model += lpSum(player_vars[i] for i in players if df['element_type'][i] == 3) >= 3  # Minimum 3 Midfielders
    model += lpSum(player_vars[i] for i in players if df['element_type'][i] == 4) <= 3  # Forwards
    model += lpSum(player_vars[i] for i in players if df['element_type'][i] == 4) >= 1  # Minimum 1 Forward

    # Exactly 11 players must be selected
    model += lpSum(player_vars[i] for i in players) == 11

    # Exclude players with predicted points less than 3
    for player in players:
        if df['total_points'][player] < 3:
            model += player_vars[player] == 0

    # No more than 3 players from a single team
    team_players = df.groupby('team').apply(lambda x: x.index.tolist()).to_dict()
    for team, players_list in team_players.items():
        model += lpSum(player_vars[i] for i in players_list) <= 3

    # Add bench constraints
    bench_players = {i: LpVariable(f"bench_{i}", cat='Binary') for i in players}
    model += lpSum(bench_players[i] for i in players) == 4  # Total bench players including goalkeeper

    # Ensure at least one goalkeeper is on the bench
    model += lpSum(bench_players[i] for i in players if df['element_type'][i] == 1) >= 1

    # Ensure exactly one goalkeeper in the starting 11
    model += lpSum(player_vars[i] for i in players if df['element_type'][i] == 1) == 1

# Function to insert data into MongoDB
def insert_data(df):
    # Rename columns for MongoDB insertion
    df.rename(columns={'Player': 'player_name', 'total_points': 'expected_points'}, inplace=True)
    data_dict = df.to_dict(orient='records')
    collection.insert_many(data_dict)

# Insert data into MongoDB (Only needed once or if you want to update)
# df = prepare_data()
# insert_data(df)

# Function to get player data from MongoDB
def get_player_from_mongo(player_name):
    return collection.find_one({"player_name": player_name})

# Main Streamlit App
def main():
    st.set_page_config(page_title="FPL Points Predictor", page_icon=":soccer:", layout="wide")

    st.markdown("""
    <style>
    .main { background-color: #000000; }
    .header { background-color: #296EB4; padding: 20px; border-radius: 10px; }
    .header h2 { color: white; text-align: center; margin: 0; }
    .input-box { margin: 10px 0; }
    .input-box label { display: block; margin-bottom: 5px; }
    .input-box input { width: 100%; padding: 10px; border-radius: 5px; border: 1px solid #ddd; }
    .button { background-color: #296EB4; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; }
    .button:hover { background-color: #1e5f8c; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="header"><h2>Fantasy Premier League Points Predictor</h2></div>', unsafe_allow_html=True)

    # Sidebar for user options
    option = st.sidebar.selectbox("Choose an option", ["Predict Points", "Display Team of the Week", "Search Player"])

    if option == "Predict Points":
        # Input fields for features
        def get_float_input(label, default_value):
            try:
                return float(st.text_input(label, default_value))
            except ValueError:
                st.error(f"Invalid value for {label}. Please enter a numerical value.")
                return 0

        bps = get_float_input("BPS", "0")
        minutes = get_float_input("Minutes", "0")
        assists = get_float_input("Assists", "0")
        goals_conceded = get_float_input("Goals Conceded", "0")
        goals_scored = get_float_input("Goals Scored", "0")

        # Prediction button
        if st.button("Predict Points", key="predict", help="Click to predict FPL points"):
            try:
                # Create feature array
                features = np.array([
                    bps,
                    minutes,
                    assists,
                    goals_conceded,
                    goals_scored,
                ])

                # Make prediction
                prediction = rf_model.predict(features.reshape([1, -1]))

                # Display the result
                st.success(f'The predicted FPL points are: {prediction[0]}')

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

    elif option == "Display Team of the Week":
        # Get user inputs
        gameweek = int(st.text_input("Enter the gameweek number:", "1"))
        budget_min = float(st.text_input("Enter the minimum budget:", "20"))
        budget_max = float(st.text_input("Enter the maximum budget:", "60"))

        # Display Team button
        if st.button("Display Team of the Week", key="display_team"):
            # Prepare data
            df = prepare_data()

            # Create the model
            model = LpProblem("Fantasy_Football_Team_Selection", LpMaximize)

            # Define player variables
            players = df.index
            player_vars = LpVariable.dicts("Player", players, cat="Binary")

            # Objective: Maximize total predicted points
            model += lpSum(df['total_points'][i] * player_vars[i] for i in players)

            # Add player constraints for the specified gameweek and budget
            add_player_constraints(model, df, player_vars, players, budget_min, budget_max)

            # Solve the optimization problem
            model.solve()

            # Check the status
            if LpStatus[model.status] == "Optimal":
                st.success("Optimal team found!")

                # Display team in the formation
                formation = {"Goalkeeper": [], "Defenders": [], "Midfielders": [], "Forwards": []}
                for i in players:
                    if player_vars[i].varValue == 1:
                        position = df['element_type'][i]
                        if position == 1:
                            formation["Goalkeeper"].append(df['Player'][i])
                        elif position == 2:
                            formation["Defenders"].append(df['Player'][i])
                        elif position == 3:
                            formation["Midfielders"].append(df['Player'][i])
                        elif position == 4:
                            formation["Forwards"].append(df['Player'][i])

                # Display formation
                st.write("Team of the Week:")
                for role, players in formation.items():
                    st.write(f"**{role}:**")
                    for player in players:
                        st.write(f"- {player}")

            else:
                st.error("No optimal solution found.")

    elif option == "Search Player":
        player_name = st.text_input("Enter player name to search:")
        
        if st.button("Search", key="search"):
            if player_name:
                player = get_player_from_mongo(player_name)
                if player:
                    st.write(f"Player: {player['player_name']}")
                    st.write(f"Predicted Points: {player['expected_points']}")
                else:
                    st.error("Player not found.")
            else:
                st.error("Please enter a player name.")

if __name__ == "__main__":
    main()
