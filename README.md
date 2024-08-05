# AI-Final-Project
Link to the video: https://youtu.be/JzvcJ7E2RTc
---

# Fantasy Premier League (FPL) Prediction App

This application provides features to predict Fantasy Premier League (FPL) points, search for player predictions, and display the team of the week based on specific criteria. 

## Features

1. **Predict FPL Points**: Predict the points a player will score based on key features.
2. **Search for a Player**: Search for a player by name and display their predicted points.
3. **Display Team of the Week**: View the optimal team for the week, considering game week, minimum budget, and maximum budget.

## Getting Started

To run the application locally, follow these steps:

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository
```

### 2. Install Required Packages

Ensure you have Python 3.x installed. Install the necessary dependencies using pip:

```bash
pip install -r requirements.txt
```

### 3. Run the Application

Start the application by running:

```bash
python app.py
```

### 4. Access the App

Open your terminal and run:

```bash
streamlit run app.py
```

This will start a local server. You can access the application in your web browser at `http://localhost:8501`.

## Using the Application

1. **Select an Option**: In the sidebar, choose one of the following options:
   - **Predict Points**: Enter values into the fields and click **Predict Points**.
   - **Display Team of the Week**: Enter the gameweek, minimum budget, and maximum budget, then click **Display Team of the Week**.
   - **Search Players**: Search for a player by name as shown in the FPL app and click **Search**.

2. **Interact with Features**:
   - For **Predict Points**, input the required values and press **Predict Points** to view the prediction.
   - For **Display Team of the Week**, provide the gameweek and budget constraints, then press **Display Team of the Week** to see the recommended team.
   - For **Search Players**, enter the player's name as it appears in the Fantasy Premier League app and press **Search** to view the player's predicted points.
