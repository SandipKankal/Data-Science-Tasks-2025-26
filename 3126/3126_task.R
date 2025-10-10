options(warn = -1)
suppressMessages(suppressWarnings({
  
  library(tidyverse)
  library(caret)
  library(rpart)
  library(rpart.plot)
  library(randomForest)
  library(ggplot2)
  library(gridExtra)
  library(pROC)
  
  set.seed(42)
  n_samples <- 2000
  
  data <- data.frame(
    Age = sample(18:60, n_samples, replace = TRUE),
    Gender = sample(c("Male", "Female"), n_samples, replace = TRUE),
    MaritalStatus = sample(c("Single", "Married", "Divorced"), n_samples, replace = TRUE, prob = c(0.4, 0.5, 0.1)),
    Education = sample(1:5, n_samples, replace = TRUE, prob = c(0.1, 0.2, 0.3, 0.25, 0.15)),
    Department = sample(c("Sales", "R&D", "HR"), n_samples, replace = TRUE, prob = c(0.5, 0.35, 0.15)),
    JobLevel = sample(1:5, n_samples, replace = TRUE, prob = c(0.3, 0.25, 0.2, 0.15, 0.1)),
    MonthlyIncome = round(rnorm(n_samples, 6500, 2500)),
    YearsAtCompany = sample(0:30, n_samples, replace = TRUE),
    YearsInCurrentRole = sample(0:20, n_samples, replace = TRUE),
    YearsSinceLastPromotion = sample(0:15, n_samples, replace = TRUE),
    NumCompaniesWorked = sample(0:9, n_samples, replace = TRUE),
    TrainingTimesLastYear = sample(0:6, n_samples, replace = TRUE),
    WorkLifeBalance = sample(1:4, n_samples, replace = TRUE),
    JobSatisfaction = sample(1:4, n_samples, replace = TRUE),
    EnvironmentSatisfaction = sample(1:4, n_samples, replace = TRUE),
    JobInvolvement = sample(1:4, n_samples, replace = TRUE),
    PerformanceRating = sample(3:4, n_samples, replace = TRUE, prob = c(0.85, 0.15)),
    OverTime = sample(c("Yes", "No"), n_samples, replace = TRUE, prob = c(0.3, 0.7)),
    DistanceFromHome = sample(1:29, n_samples, replace = TRUE),
    StockOptionLevel = sample(0:3, n_samples, replace = TRUE, prob = c(0.6, 0.2, 0.15, 0.05))
  )
  
  data$MonthlyIncome <- data$MonthlyIncome + (data$JobLevel * 1000) + (data$Education * 500)
  data$MonthlyIncome <- pmax(data$MonthlyIncome, 1500)
  
  churn_prob <- rep(0.16, n_samples)
  churn_prob[data$JobSatisfaction == 1] <- churn_prob[data$JobSatisfaction == 1] + 0.20
  churn_prob[data$WorkLifeBalance == 1] <- churn_prob[data$WorkLifeBalance == 1] + 0.15
  churn_prob[data$YearsAtCompany < 2] <- churn_prob[data$YearsAtCompany < 2] + 0.25
  churn_prob[data$OverTime == "Yes"] <- churn_prob[data$OverTime == "Yes"] + 0.15
  churn_prob[data$YearsSinceLastPromotion > 5] <- churn_prob[data$YearsSinceLastPromotion > 5] + 0.10
  churn_prob[data$EnvironmentSatisfaction == 1] <- churn_prob[data$EnvironmentSatisfaction == 1] + 0.12
  churn_prob[data$MonthlyIncome < 3000] <- churn_prob[data$MonthlyIncome < 3000] + 0.18
  churn_prob[data$StockOptionLevel > 0] <- churn_prob[data$StockOptionLevel > 0] - 0.10
  churn_prob[data$JobInvolvement == 4] <- churn_prob[data$JobInvolvement == 4] - 0.08
  churn_prob <- pmin(pmax(churn_prob, 0), 1)
  
  data$Attrition <- ifelse(runif(n_samples) < churn_prob, "Yes", "No")
  data$Attrition <- factor(data$Attrition, levels = c("No", "Yes"))
  
  pdf(NULL)
  categorical_vars <- c("Gender", "MaritalStatus", "Department", "OverTime")
  data[categorical_vars] <- lapply(data[categorical_vars], as.factor)
  
  data$IncomePerYear <- data$MonthlyIncome / pmax(data$YearsAtCompany, 1)
  data$PromotionRate <- data$YearsAtCompany / pmax(data$YearsSinceLastPromotion + 1, 1)
  data$TenureRatio <- data$YearsInCurrentRole / pmax(data$YearsAtCompany, 1)
  data$IncomePerYear[is.infinite(data$IncomePerYear)] <- data$MonthlyIncome[is.infinite(data$IncomePerYear)]
  data$PromotionRate[is.infinite(data$PromotionRate)] <- 0
  data$TenureRatio[is.infinite(data$TenureRatio)] <- 0
  
  set.seed(42)
  train_index <- createDataPartition(data$Attrition, p = 0.8, list = FALSE)
  train_data <- data[train_index, ]
  test_data <- data[-train_index, ]
  
  dt_model <- rpart(Attrition ~ ., data = train_data, method = "class",
                    control = rpart.control(cp = 0.01, maxdepth = 10, minsplit = 20, minbucket = 10))
  
  rf_model <- randomForest(Attrition ~ ., data = train_data, ntree = 500,
                           mtry = sqrt(ncol(train_data) - 1), importance = TRUE, nodesize = 5)
  
  dt_test_acc <- confusionMatrix(predict(dt_model, test_data, type = "class"), test_data$Attrition)$overall['Accuracy']
  rf_test_acc <- confusionMatrix(predict(rf_model, test_data), test_data$Attrition)$overall['Accuracy']
  
  if (dt_test_acc > rf_test_acc) {
    best_model <- dt_model
  } else {
    best_model <- rf_model
  }
  
  saveRDS(best_model, "best_churn_model.rds")
  saveRDS(train_data, "train_structure.rds")
}))

predict_churn <- function(employee_data) {
  model <- readRDS("best_churn_model.rds")
  train_structure <- readRDS("train_structure.rds")
  
  new_df <- as.data.frame(employee_data)
  
  categorical_vars <- c("Gender", "MaritalStatus", "Department", "OverTime")
  new_df[categorical_vars] <- lapply(new_df[categorical_vars], as.factor)
  
  new_df$IncomePerYear <- new_df$MonthlyIncome / pmax(new_df$YearsAtCompany, 1)
  new_df$PromotionRate <- new_df$YearsAtCompany / pmax(new_df$YearsSinceLastPromotion + 1, 1)
  new_df$TenureRatio <- new_df$YearsInCurrentRole / pmax(new_df$YearsAtCompany, 1)
  new_df$IncomePerYear[is.infinite(new_df$IncomePerYear)] <- new_df$MonthlyIncome[is.infinite(new_df$IncomePerYear)]
  new_df$PromotionRate[is.infinite(new_df$PromotionRate)] <- 0
  new_df$TenureRatio[is.infinite(new_df$TenureRatio)] <- 0
  
  for (col in names(train_structure)) {
    if (!(col %in% names(new_df))) new_df[[col]] <- NA
    if (is.factor(train_structure[[col]])) {
      new_df[[col]] <- factor(new_df[[col]], levels = levels(train_structure[[col]]))
    }
  }
  
  new_df <- new_df[, names(train_structure)[names(train_structure) != "Attrition"], drop = FALSE]
  
  if (inherits(model, "rpart")) {
    prediction <- predict(model, new_df, type = "class")
    probabilities <- predict(model, new_df, type = "prob")
  } else {
    prediction <- predict(model, new_df)
    probabilities <- predict(model, new_df, type = "prob")
  }
  
  prob_stay <- probabilities[1, "No"]
  prob_leave <- probabilities[1, "Yes"]
  
  if (prob_leave < 0.3) risk_level <- "LOW"
  else if (prob_leave < 0.6) risk_level <- "MEDIUM"
  else risk_level <- "HIGH"
  
  churn_status <- ifelse(prediction == "Yes", "WILL LEAVE", "WILL STAY")
  
  return(list(
    status = churn_status,
    prob_stay = prob_stay,
    prob_leave = prob_leave,
    risk_level = risk_level
  ))
}

test_case_1 <- list(
  Age = 28, Gender = "Female", MaritalStatus = "Single", Education = 3, Department = "Sales",
  JobLevel = 1, MonthlyIncome = 2500, YearsAtCompany = 1, YearsInCurrentRole = 1,
  YearsSinceLastPromotion = 0, NumCompaniesWorked = 4, TrainingTimesLastYear = 1,
  WorkLifeBalance = 1, JobSatisfaction = 1, EnvironmentSatisfaction = 2, JobInvolvement = 2,
  PerformanceRating = 3, OverTime = "Yes", DistanceFromHome = 25, StockOptionLevel = 0
)

cat("\n Test Case 1: High-Risk Employee\n", rep("-", 60), "\n")
result <- predict_churn(test_case_1)
cat(sprintf(" Prediction: %s\n   Probability of Staying: %.2f%%\n   Probability of Leaving: %.2f%%\n   Risk Level: %s\n",
            result$status, result$prob_stay * 100, result$prob_leave * 100, result$risk_level))

