# Load required packages
library(rsample)   # For data splitting 
library(caret)     # For automating the tuning process
library(ggplot2)   # plotting
library(earth)     # fit MARS models
library(vip)       # For variable importance
library(pdp)       # For variable relationships
library(Metrics)   # For evaluation metrics
library(dplyr)     # For data manipulation
library(glmnet)    # For fitting GLM models
library(mgcv)      # For fitting GAM models
library(caTools)   # For data splitting
library(segmented) # For fitting segmented regression models

# Define function to make predictions using a saved model
model <- function(input_model_path, input_data_path){
  
  # Load tuned GAM model from saved file
  tuned_model <- readRDS(input_model_path) # Path to the saved model
  
  # Load test data for predictions
  data_test <- read.csv(input_data_path) # Path to the dataset for predictions
  
  # Make predictions using model
  pred <- predict(tuned_model, data_test)
  prediction_table <- cbind("actual" = data_test$proportion, 'predicted' = pred)
  
  # Calculate and print evaluation metrics
  RMSE <- rmse(data_test$proportion, pred) # Calculate RMSE
  RMSE
  
  rss <- sum((pred - data_test$proportion) ^ 2) # Calculate residual sum of squares
  tss <- sum((data_test$proportion - mean(data_test$proportion)) ^ 2) # Calculate total sum of squares
  rsq <- 1 - (rss/tss) # Calculate R-squared
  rsq
  
  # Store predictions and evaluation metrics in a list and return
  results <- list(prediction_table, RMSE, rsq)
  return(results)
}

# Call the function with the file paths for the model and test data
Results <- model("model.rds", 'df_quantiles.csv')

# Print the results
Results