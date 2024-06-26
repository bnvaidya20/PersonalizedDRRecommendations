# Implementation of Personalized Demand Response (DR) Recommendation System

## Project Overview
This project implements a Personalized Demand Response (DR) Recommendation System aimed at optimizing energy consumption and enhancing user engagement with demand response programs. The system leverages advanced machine learning models and data analytics to provide tailored DR recommendations to individual households or entities.

Use Case: ML algorithms can analyze individual energy consumption patterns to provide personalized recommendations for energy reduction during DR events.

Benefits: Tailored recommendations can improve participant engagement by showing them the most effective ways to reduce their energy usage, based on their habits and the devices they use.

## Features
- Data loading, preprocessing and exploratory data analysis (EDA)
- Hyperparameter tuning for model optimization
- Implementation of predictive models
- Implementation of DR recommendations
- Visualization tools for data and prediction insights

## Dataset
This study uses a dataset that can be found at the following UCI ML repository link:

[Individual household electric power consumption](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)

The dataset has measurements of electric power consumption in one household with a one-minute sampling rate over almost 4 years with different electrical quantities and some sub-metering values. 

It contains 2075259 measurements gathered in a house located in Sceaux (7 km from Paris, France) between December 2006 and November 2010 (47 months).

## Directory Structure
PersonalizedDRRecommendations/

- utils/ # Utility scripts

  |- model.py # Model implementation
  
  |- recommender_scheme.py # Recommendation scheme logic
  
  |- utils.py # General utility functions
  
  |- utils_plot.py # Data visualization functions

- config.py # Configuration settings and parameters
- eda_preprocessing.py # Script for data loading, preprocessing and EDA
- prediction.py # Predictive modeling and analysis
- model_hyperparam_tuning.py # Model Hyperparameter tuning
- recommender_initiator.py # Initiates the recommendation process
- recommender_query.py # Handles queries for recommendations
- README.md # Project documentation

