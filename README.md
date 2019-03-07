# ML_Project
Kaggle competition for Ames Iowa real estate data

The goal of this project/competition was to create a model to predict a set of housing prices in Ames, Iowa.
The training data included 80 house features and 1460 samples of houses. Our group consisted of three members.
We perfored EDA, feature engineering, log/box-cox transformation and imputation on the data set. We created a function to do this for us.
For the modeling we used regularized regression techniques, inclduing ridge, lasso and elastic net. 
We also used Random Forest, XGBoost and Light GBM as well as a stacked average of our models.
Our best score was a RMSE of .11781 which ranked in the top 19% of the Kaggle leaderboard.
