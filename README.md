# MLB Game Score Prediction with a Multi-Output Neural Network
This repository contains the code and analysis for a deep learning project aimed at predicting multiple outcomes of Major League Baseball (MLB) games. Instead of a simple win/loss prediction, this model simultaneously forecasts four related values: home team score, away team score, total runs, and margin of victory.

The project is detailed in the Jupyter Notebook: mlb-project.ipynb.

Project Overview
The core of this project is a multi-output neural network built with TensorFlow and Keras. It leverages extensive feature engineering on historical game data to learn the complex patterns that drive game scores. The model's architecture and training process are systematically optimized using Keras Tuner, and a custom loss function is implemented to ensure the predicted scores are mathematically consistent.

Key Features
Multi-Output Prediction: A single model predicts four interconnected targets.
Extensive Feature Engineering: Over 144 features were engineered, including 10-game moving averages for team statistics to capture recent form and momentum.
Custom Loss Function: A "consistency-aware" loss function penalizes the model for predictions where home_score + away_score does not equal total_score.
Hyperparameter Tuning: Bayesian Optimization was used via Keras Tuner to find the optimal model architecture and learning parameters.
Temporal Data Split: The dataset was split chronologically to ensure the model is validated on data that occurs after the training data, simulating a real-world prediction scenario.
Dataset
The project uses the MLB Game Data dataset from Kaggle, which contains detailed game, player, and team statistics from April 2016 to October 2021.

Source: Kaggle: MLB Game Data
Files Used: games.csv, hittersByGame.csv, and pitchersByGame.csv.
Methodology
Data Cleaning & Preprocessing: The raw data was filtered to include only standard 9-inning regular season games. The three primary data files were then merged into a single comprehensive dataset.
Feature Engineering: Team statistics were aggregated on a per-game basis. Ten-game rolling averages were calculated for all key metrics to provide the model with a dynamic view of a team's performance. Ratio-based features (e.g., Offensive Power Ratio) were also created to model matchups more effectively.
Model Architecture: A sequential neural network was designed with a flexible number of hidden layers and a final 4-neuron output layer corresponding to the four prediction targets.
Training & Evaluation: The model was trained on 70% of the data, validated on the next 15%, and tested on the final 15% (chronologically). Performance was primarily measured by Mean Absolute Error (MAE).
