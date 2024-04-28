# CRR-prediction - Bachelor's Thesis 

This repository contains the Python code developed for my Bachelor's thesis, which focuses on the implementation and evaluation of two advanced statistical models: the Shifted Beta Geometric (SBG) model and the Beta Discrete Weibull (BDW) model. 
## Project Structure

sbg.py            - Implementation of the Shifted Beta Geometric model.
bdw.py            - Implementation of the Beta Discrete Weibull model.
performance2.py   - Script for assessing the performance of both models.
preprocess.py     - Data preprocessing script for model preparation.


## sbg.py: Shifted Beta Geometric Model

This script contains the implementation of the SBG model.

### Key Functions:
- `fit(df_data)`: Fits the model to the data, optimizing model parameters to maximize the likelihood.
- `ll(t, delta, alpha, beta)`: Log-likelihood function that calculates the likelihood of a customer churning at time t.
- `pdf(t, alpha, beta)`: Probability density function, calculating the probability of churn at time t.
- `survivor(t, alpha, beta)`: Survivor function, determining the probability of a customer not churning by time t.
- `coeff()`: Returns the cumulative sum of retention rates up to the time t.

## bdw.py: Beta Discrete Weibull Model

This script handles the implementation of the BDW model.

### Key Functions:
- `fit(df_data)`: Fits the BDW model to the data, optimizing the parameters for the best model fit.
- `ll(t, delta, gamma, delta, c)`: Log-likelihood function that calculates the likelihood of an event occurring at time t.
- `pdf(t, gamma, delta, c)`: Probability density function for the occurrence of an event at time t.
- `survivor(t, gamma, delta, c)`: Survivor function, calculating the probability of an event not occurring by time t.
- `coeff()`: Returns the cumulative sum of retention rates up to the time t.

## performance2.py: Data collection and preprocessing, model's performance assessment 

This Python script is designed for analytics in subscription-based businesses. It integrates data collection, preprocessing, visualization, and prediction tasks. It heavily utilizes pandas for data manipulation, numpy for numerical operations, and matplotlib for plotting. The script also interacts with databases using a custom database client and handles dynamic SQL query generation for robust data retrieval and processing.

### Key Components:
- **Data Collection**: Connects to a PostgreSQL database to fetch subscription data based on dynamic intervals and conditions.
- **Data Preprocessing**: Implements sophisticated data transformations, including moving calculations and pivoting, to prepare data for analysis.
- **Visualization**: Plots data to compare model predictions against actual data, providing visual insights into model performance.
- **Prediction**: Utilizes fitted models to predict renewals and churn, evaluating these predictions against historical data to compute RMSE (Root Mean Square Error) as a performance metric.

### Key Functions:

#### `plot_individual_bdw_distributions()` and `plot_individual_sbg_distributions()`
- **Purpose**: Plots the fitted Beta Discrete Weibull (BDW) and Shifted Beta Geometric (sBG) distributions against the training and validation data to visualize model fits.
- **Parameters**: Includes segment info and processed data for fitting and plotting.
- **Returns**: RMSE for the model fits to assess prediction accuracy.

## preprocess.py

### Key Functions:
#### moving_calculation(df, window_size)
- **Purpose:** Calculates moving sums over a specified window size on a pivot table of renewals by tenure.
- **Parameters:** 
  - `df`: Dataframe containing the renewal data.
  - `window_size`: Size of the window over which the moving sum is calculated.
- **Returns:** A dataframe with adjusted renewals over the specified window, providing a view of aggregated customer actions over time.

#### preprocess_level(df_level, level_info)
- **Purpose:** Preprocesses and evaluates data for a specific subscription level, handling different stages of data cleanliness and completeness.
- **Parameters:**
  - `df_level`: Dataframe at a particular subscription level.
  - `level_info`: Dictionary containing specific configurations, such as `duration_interval`.
- **Operations:** 
  - Fills missing data, calculates good tenures, adjusts data based on minimum subscriptions, and checks for data sufficiency.
  - Handles data shortages by applying a moving calculation.
- **Returns:** A dataframe with the preprocessed and restructured subscription data, ready for further analysis.

### Usage:
Preprocessed the data to be ready for the model fitting.

