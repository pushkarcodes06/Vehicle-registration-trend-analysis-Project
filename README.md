VEHICLE REGISTRATION TREND ANALYSIS IN INDIA USING DATA ANALYSIS, VISUALIZATION AND MACHINE LEARNING
1. Abstract

Vehicle registration data provides useful information about transportation demand, vehicle ownership, fuel preferences, and changes in mobility. This project analyzes vehicle registration trends in India using Python, data visualization, and machine learning techniques.

The project uses historical vehicle registration data obtained from the VAHAN system through a publicly accessible dataset. Data preprocessing is performed using Pandas and NumPy. Matplotlib and Seaborn are used to visualize annual trends, vehicle classes, fuel categories, and electric vehicle registrations.

Ridge Regression and Random Forest Regression are implemented to forecast annual vehicle registrations. Their predictions are evaluated against a held-out historical year using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).

The project demonstrates how data analytics and machine learning can support the understanding of vehicle registration patterns and provide exploratory estimates of future demand.

2. Introduction

Vehicle registration is an important indicator of transportation activity and mobility demand. Analyzing registrations over time can reveal long-term growth patterns, changes in vehicle categories, and the adoption of alternative fuel technologies.

India has a diverse vehicle market that includes two-wheelers, passenger cars, commercial vehicles, and electric vehicles. Analyzing registration records can help researchers and policymakers understand changes in this market.

This project uses Python-based data analysis and machine learning to study historical registration records and produce visual summaries and forecasts.

3. Problem Statement

Historical vehicle registration data is available in large tabular datasets, making manual analysis difficult. It is necessary to organize and clean the data, identify meaningful trends, compare vehicle and fuel categories, and investigate whether historical patterns can be used to estimate future registrations.

The project addresses these requirements through exploratory data analysis, visualization, and predictive modeling.

4. Objectives
Collect vehicle registration data relevant to India.
Clean and preprocess the dataset.
Analyze annual vehicle registration trends.
Compare registrations across vehicle classes and fuel types.
Examine the recorded trend in electric vehicle registrations.
Build and evaluate machine learning forecasting models.
Visualize historical observations and future estimates.
5. Dataset Description

Dataset: Year, Month, Vehicle Class and Fuel Type-wise Total Number of Vehicles Registered at All India Level.

Source: Dataful dataset sourced from the Ministry of Road Transport and Highways' VAHAN portal.

Dataset link: https://dataful.in/datasets/18261/

The dataset documentation describes monthly vehicle registration records from 2014 onward, with fields including year, month, state, RTO, vehicle class, fuel type, registration count, unit, and notes.

Attribute	Description
year	Calendar year of registration
month	Month of registration
state	Geographic reporting scope
rto	Regional Transport Office scope
vehicle_class	Registered vehicle class
fuel	Fuel or powertrain category
value	Number of registrations
unit	Measurement unit
note	Additional remarks

The downloaded dataset should be inspected to verify its actual coverage, geographic scope, missing values, and reporting conventions before analysis.

6. Tools and Technologies
Python: Programming language.
Pandas: Data manipulation and aggregation.
NumPy: Numerical operations.
Matplotlib: Chart creation.
Seaborn: Statistical visualization.
Scikit-learn: Machine learning and evaluation.
CSV: Data storage and exchange.
7. Methodology
7.1 Data collection

The dataset is downloaded from the public dataset page and saved as a CSV file in the project's data folder.

7.2 Data preprocessing

The following preprocessing operations are performed:

Standardize column names.
Convert year and registration counts to numeric types.
Remove records with missing essential fields.
Remove duplicate rows.
Validate month values.
Exclude negative registration counts.
Select national-level records when the dataset contains national and state aggregates.
Identify complete calendar years for annual comparisons.

Data cleaning is important because incomplete periods and duplicate aggregate records can distort registration totals.

7.3 Exploratory data analysis

The analysis examines annual registration totals, year-over-year growth, the most frequently recorded vehicle classes, fuel category distributions, and electric vehicle registrations.

7.4 Data visualization

The project produces the following visualizations:

Annual vehicle registration trend line chart.
Horizontal bar chart of the top ten vehicle classes.
Horizontal bar chart of the top ten fuel categories.
Electric vehicle registration trend chart.
Actual versus predicted registrations chart.
Historical and forecast trend chart.
7.5 Machine learning

Two regression models are used:

Ridge Regression: A regularized linear model used as a baseline.

Random Forest Regression: An ensemble model that can represent nonlinear relationships in the lagged historical data.

The forecasting features include the previous three annual registration totals and the target year. The latest complete year is held out for testing, while earlier years are used for training.

After evaluation, the models are refitted using the available historical observations to produce exploratory estimates for the following three years.

8. Evaluation Metrics
Mean Absolute Error (MAE)

MAE measures the average absolute difference between actual and predicted registration counts. Lower values indicate smaller prediction errors.

Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared prediction error. It penalizes larger errors more strongly than MAE.

Both metrics are expressed in registration-count units.

9. Results and Discussion

The Python program generates the annual summary, category comparisons, fuel analysis, model evaluation, and future forecast files.

The following results must be completed after running the program on the downloaded dataset:

Historical trend: [Insert the observed pattern in annual registrations.]
Highest-registration year: [Insert year and total from the annual summary.]
Vehicle class with the highest recorded count: [Insert result from the class analysis.]
Leading fuel category: [Insert result from the fuel analysis.]
Electric vehicle trend: [Describe the observed trend if the available categories support a valid comparison.]
Best-performing model: [Compare the held-out-year MAE and RMSE.]
Future estimates: [Insert the values from future_forecast.csv.]

No specific model accuracy or data-derived finding should be reported until the program has been executed and the results verified.

10. Applications
Transportation planning and infrastructure studies.
Analysis of vehicle-market trends.
Exploration of fuel and powertrain adoption.
Historical registration reporting.
Educational demonstrations of time-series forecasting.
Preliminary estimates of future registration demand.
11. Limitations
The accuracy of the analysis depends on the completeness and consistency of the source data.
The current-year data may represent only a partial year.
A national aggregate dataset cannot independently establish state-level differences.
Fuel-category labels may require normalization before calculating electric vehicle totals.
A single held-out test year provides limited evidence of model reliability.
Historical registration patterns may change because of economic conditions, regulations, technology, or other external factors.
Forecasts are estimates and should not be treated as official projections.
12. Future Enhancements

The project can be improved by incorporating state-level datasets, monthly time-series forecasting, rolling-origin validation, additional models, and an interactive dashboard using Streamlit. External variables such as fuel prices, population, and economic indicators may also be explored if suitable historical data is available.

13. Conclusion

This project provides an end-to-end workflow for analyzing vehicle registration trends in India using Python. It combines data cleaning, exploratory data analysis, visualization, and machine learning forecasting in a single implementation.

The project helps identify historical patterns and compare regression models for educational forecasting. The reliability of the conclusions depends on the source data, appropriate validation, and careful interpretation of incomplete periods and aggregated records.

14. References
Ministry of Road Transport and Highways, Government of India, VAHAN Public Dashboard: https://www.analytics.parivahan.gov.in/analytics/publicdashboard/vahan
Dataful, Master Data: Year, Month, Vehicle Class and Fuel Type-wise Total Number of Vehicles Registered at All India Level: https://dataful.in/datasets/18261/
Pandas documentation: https://pandas.pydata.org/docs/
Matplotlib documentation: https://matplotlib.org/stable/
Seaborn documentation: https://seaborn.pydata.org/
Scikit-learn documentation: https://scikit-learn.org/stable/
