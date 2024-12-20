# Author: Ohm Kundurthy
# Date: Dec-7-2024
# Purpose: Case 3 - Bedding Bathing & Yonder (BBY) - model to predict - household revenue 
# Code Organized by SEMMA 
#--------------------------------------------------------------------------------------------
# Libs
library(vtreat)
library(dplyr)
library(ggplot2)
library(maps)
library(ggthemes)
library(scales)
library(tidyr)
library(reshape2)
library(plotly)
library(GGally)
library(MLmetrics)
library(Metrics)
library(pROC)
library(caret)
library(gbm) 
library(rpart.plot)
options(scipen = 999)

# Setwd
setwd("~/Git/GitHub_R/Cases/Fall/III Household Spend/studentTables")

# Read the training data for the case 
consumerData_training<- read.csv('consumerData_training15K_studentVersion.csv')
DonationsData_training<- read.csv('DonationsData_training15K_studentVersion.csv')
inHouseData_training<- read.csv('inHouseData_training15K_studentVersion.csv')
magazineData_training<- read.csv('magazineData_training15K_studentVersion.csv')
politicalData_training<- read.csv('politicalData_training15K_studentVersion.csv')


# Read the testing data for the case 
consumerData_testing<- read.csv('consumerData_testing5K_studentVersion.csv')
DonationsData_testing<- read.csv('DonationsData_testing5K_studentVersion.csv')
inHouseData_testing<- read.csv('inHouseData_testing5K_studentVersion.csv')
magazineData_testing<- read.csv('magazineData_testing5K_studentVersion.csv')
politicalData_testing<- read.csv('politicalData_testing5K_studentVersion.csv')

# Read the prospects data for the case 
consumerData_prospects<- read.csv('consumerData_prospects6K_studentVersion.csv')
DonationsData_prospects<- read.csv('DonationsData_prospects6K_studentVersion.csv')
inHouseData_prospects<- read.csv('inHouseData_prospects6K_studentVersion.csv')
magazineData_prospects<- read.csv('magazineData_prospects6K_studentVersion.csv')
politicalData_prospects<- read.csv('politicalData_prospects6K_studentVersion.csv')

########################################SAMPLE################################
#Note that data sets are already partitioned into training, testing and prospects

##### PREPROCESSING#####

#1.left Join training data sets using the key attribute
trainingData<- consumerData_training %>%
  left_join(DonationsData_training, by = "tmpID") %>%
  left_join(inHouseData_training, by = "tmpID") %>%
  left_join(magazineData_training, by = "tmpID") %>%
  left_join(politicalData_training, by = "tmpID")

# View the structure of the joined training data
str(trainingData)

#left Join testing data sets using the key attribute
testingData<- consumerData_testing %>%
  left_join(DonationsData_testing, by = "tmpID") %>%
  left_join(inHouseData_testing, by = "tmpID") %>%
  left_join(magazineData_testing, by = "tmpID") %>%
  left_join(politicalData_testing, by = "tmpID")

# View the structure of the joined testing data
str(testingData)


#left Join prospects data sets using the key attribute
prospectsData<- consumerData_prospects %>%
  left_join(DonationsData_prospects, by = "tmpID") %>%
  left_join(inHouseData_prospects, by = "tmpID") %>%
  left_join(magazineData_prospects, by = "tmpID") %>%
  left_join(politicalData_prospects, by = "tmpID")

# View the structure of the joined prospects data
str(prospectsData)
#note that prospects does not have the target variable yHat

#Partitioning not necessary as data is already partitioned

#############################Explore########################################################

# Examine the possible values for each attribute; 
summary(trainingData)

# 1.visualization#1 box plot for ResidenceHHGenderDescription vs. average yHat
ggplot(trainingData, aes(x = ResidenceHHGenderDescription, y = yHat)) +
  geom_bar(stat = "summary", fun = "mean", fill = "steelblue") +
  theme_minimal() +
  labs(title = "Average yHat by ResidenceHHGenderDescription",
       x = "ResidenceHHGenderDescription",
       y = "Average yHat") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#  Summarize average yHat by ResidenceHHGenderDescription
aggregate(yHat ~ ResidenceHHGenderDescription, data = trainingData, FUN = mean)
#Female Only, Male Only, and Mixed Gender Households have same distribution and same average
#"Cannot Determine" group has the highest prediction value (382.8046)

# 2. visualization#2 Box plot for PresenceOfChildrenCode vs. yHat
ggplot(trainingData, aes(x = PresenceOfChildrenCode, y = yHat)) +
  geom_boxplot(fill = "steelblue") +
  theme_minimal() +
  labs(title = "Distribution of yHat by PresenceOfChildrenCode", x = "Presence of Children", y = "yHat")
#evenly distributed regardless of the value, limited predictive power

# Summarize yHat statistics by PresenceOfChildrenCode
aggregate(yHat ~ PresenceOfChildrenCode, data = trainingData, summary)
#overlap in ranges suggests that PresenceOfChildrenCode has limited predictive power 


# 3.visualization#3 Density plot for ISPSA vs. yHat
ggplot(trainingData, aes(x = ISPSA, y = yHat)) +
  geom_bar(stat = "summary", fun = "mean", fill = "steelblue") +
  theme_minimal() +
  labs(title = "Average yHat by ISPSA",
       x = "ISPSA",
       y = "Average yHat") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Correlation between ISPSA and yHat
cor(trainingData$ISPSA, trainingData$yHat, use = "complete.obs")
#correlation coefficient of -0.0006 a very weak negative relationship


# 4. visualization#4  Box plot for HomeOwnerRenter vs yHat
ggplot(trainingData, aes(x = HomeOwnerRenter, y = yHat)) +
  geom_boxplot(fill = "steelblue") +
  theme_minimal() +
  labs(title = "yHat Distribution by HomeOwner vs Renter",
       x = "HomeOwner vs Renter",
       y = "yHat")

# Summary of yHat by HomeOwnerRenter
aggregate(yHat ~ HomeOwnerRenter, data = trainingData, FUN = summary)

# 5. visualization#5 Scatter plot for NetWorth vs yHat
ggplot(trainingData, aes(x = NetWorth, y = yHat)) +
  geom_point(color = "steelblue") +
  theme_minimal() +
  labs(title = "NetWorth vs yHat",
       x = "NetWorth",
       y = "yHat")

# Summary statistics for NetWorth vs yHat
aggregate(yHat ~ NetWorth, data = trainingData, FUN = mean)
# general trend is yHat decreases as net worth increases

#6. visualization#6 # Bar plot for Investor vs yHat
ggplot(trainingData, aes(x = Investor, y = yHat)) +
  geom_bar(stat = "summary", fun = "mean", fill = "steelblue") +
  theme_minimal() +
  labs(title = "Average yHat by Investor Status",
       x = "Investor",
       y = "Average yHat")

# Summary of yHat by Investor status
aggregate(yHat ~ Investor, data = trainingData, FUN = mean)
#investor status has no effect on predicting yHat

#7. Visualization 7  Box plot for BusinessOwner vs yHat
ggplot(trainingData, aes(x = BusinessOwner, y = yHat)) +
  geom_boxplot(fill = "steelblue") +
  theme_minimal() +
  labs(title = "yHat Distribution by Business Owner Status",
       x = "Business Owner Status",
       y = "yHat")

# Summary of yHat by BusinessOwner
aggregate(yHat ~ BusinessOwner, data = trainingData, FUN = summary)

#8. Visualization 8 Bar plot for Education vs yHat
ggplot(trainingData, aes(x = Education, y = yHat)) +
  geom_bar(stat = "summary", fun = "mean", fill = "steelblue") +
  theme_minimal() +
  labs(title = "Average yHat by Education",
       x = "Education",
       y = "Average yHat") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Summary of yHat by Education
aggregate(yHat ~ Education, data = trainingData, FUN = mean)
#couple of categories standout among rest for min and max mean values

#9. Visualization 9 Box plot for HomeOffice vs yHat
ggplot(trainingData, aes(x = HomeOffice, y = yHat)) +
  geom_boxplot(fill = "steelblue") +
  theme_minimal() +
  labs(title = "yHat Distribution by Home Office Status",
       x = "Home Office Status",
       y = "yHat")

# Summary of yHat by HomeOffice
 aggregate(yHat ~ HomeOffice, data = trainingData, FUN = summary)
#similar distributions across both groups

# 10. Bar plot for stateFips vs yHat
 ggplot(trainingData, aes(x = stateFips, y = yHat)) +
   geom_bar(stat = "summary", fun = "mean", fill = "steelblue") +
   theme_minimal() +
   labs(title = "Average yHat by stateFips",
        x = "stateFips",
        y = "Average yHat") +
   theme(axis.text.x = element_text(angle = 45, hjust = 1))
 
 # Summary of yHat by stateFips
 aggregate(yHat ~ stateFips, data = trainingData, FUN = mean)
 #distribution range is very small across statefips

#11. Use lat lon for map plots
 # Create a base world map
 map <- map_data("usa")
 
 
 # Plot using ggplot2
 ggplot() +
   # Plot the world map
   geom_polygon(data = map, aes(x = long, y = lat, group = group), 
                fill = "lightgray", color = "white") +
   # Overlay yHat points (colored by yHat)
   geom_point(data = trainingData, aes(x = lon, y = lat, color = yHat), size = 1) +
   # Add a color scale to show the range of yHat values
   scale_color_gradient(low = "blue", high = "red", name = "yHat") +
   # Customize the plot
   labs(title = "yHat vs Latitude and Longitude",
        x = "Longitude",
        y = "Latitude") +
   theme_minimal() +
   theme(legend.position = "bottom")

#OccupationIndustry: variable has a high cardinality (many unique industries)
table(trainingData$OccupationIndustry)
#Unknown for 9588 records 
#Other for 898 records 

table(trainingData$BusinessOwner)
#Populated for only 262 rows but could be relevant to the business

table(trainingData$HomeOffice)
#Populated for only 2657 rows  but could be relevant to the business

table(trainingData$Investor)
#Populated for 6299 fields

table(trainingData$stateFips)


#perform Chi Square Test
chisq.test(table(trainingData$OccupationIndustry, trainingData$yHat))
#A p-value 0.4899 means no statistically significant relationship with yHat
#Rebin to see if any specific occupations have correaltion to spending 

#PartiesDescription, ReligionsDescription, overallsocialviews - do these features don't have a clear relationship with household spending? 
chisq.test(table(trainingData$PartiesDescription, trainingData$yHat))
#A p-value 0.4899 means no statistically significant relationship with yHat

chisq.test(table(trainingData$ReligionsDescription, trainingData$yHat))
#A p-value 0.4916 means no statistically significant relationship with yHat

chisq.test(table(trainingData$overallsocialviews, trainingData$yHat))
#A p-value 0.4957 means no statistically significant relationship with yHat


# 12. Set a threshold for high cardinality 
threshold <- 6

# Identify categorical variables in the dataset
categoricalVars <- names(trainingData)[sapply(trainingData, is.factor) | sapply(trainingData, is.character)]

# Calculate the number of unique levels for each categorical variable
cardinality <- sapply(trainingData[categoricalVars], function(col) length(unique(col)))

# Find high-cardinality categorical variables
highCardinalityVars <- names(cardinality[cardinality > threshold])

# Print the high-cardinality variables and their cardinality
if (length(highCardinalityVars) > 0) {
  cat("High Cardinality Categorical Variables:\n")
  print(data.frame(Variable = highCardinalityVars, UniqueLevels = cardinality[highCardinalityVars]))
} else {
  cat("No high-cardinality categorical variables found.\n")
}

#Results
# EthnicDescription                     EthnicDescription           55. - Not Ethical to use in ML 
# BroadEthnicGroupings               BroadEthnicGroupings           52  - Not Ethical to use in ML 
# MosaicZ4                                       MosaicZ4           47  - Empty for many records, demographic and Economic indicators could be captured by StateFips
# NetWorth                                       NetWorth            9. 
# Education                                     Education           12
# OccupationIndustry                   OccupationIndustry           19 
# BookBuyerInHome                         BookBuyerInHome           10 
# ReligiousContributorInHome   ReligiousContributorInHome           10
# PoliticalContributerInHome   PoliticalContributerInHome           10
# FirstName                                     FirstName         4463 - Not suitable for ML 
# LastName                                       LastName          475 - Not suitable for ML 
# TelephonesFullPhone                 TelephonesFullPhone        15000 - Not suitable for ML 
# county                                           county         1570 - Duplicate to Lat/Lon 
# city                                               city         6617 - Duplicate to Lat/Lon
# state                                             state           58 - Duplicate to StateFips
# stateFips                                     stateFips           52 - standardized variable for State
# HomePurchasePrice                     HomePurchasePrice          505 - Conversion to bins needed
# LandValue                                     LandValue           88 - Conversion to bins needed
# DwellingUnitSize                       DwellingUnitSize           10
# PropertyType                               PropertyType            7
# EstHomeValue                               EstHomeValue         3061 - Conversion to bins needed
# FamilyMagazineInHome               FamilyMagazineInHome            8
# HealthFitnessMagazineInHome HealthFitnessMagazineInHome           10
# DoItYourselfMagazineInHome   DoItYourselfMagazineInHome           10
# FinancialMagazineInHome         FinancialMagazineInHome           10
# ReligionsDescription               ReligionsDescription           13  - Not Ethical to use in ML 

###############################MODIFY1########################################################

#apply vtreat to standardize the chosen features before modeling

# Specify the target variable
target <- "yHat"

# Create a treatment plan on the training data
treatPlan <- designTreatmentsN(trainingData, names(trainingData)[c(2:15, 17:23)], target)

# Apply the treatment to training, testing, and prospects data
trainingDataTreated <- prepare(treatPlan, trainingData)
testingDataTreated <- prepare(treatPlan, testingData)
prospectsDataTreated <- prepare(treatPlan, prospectsData)

########################################MODEL1########################################################
# Set the target variable and predictors
target <- "yHat"
predictors <- setdiff(names(trainingDataTreated), target)

# Prepare the training and testing datasets
trainData <- trainingDataTreated[, c(predictors, target)]
testData <- testingDataTreated[, c(predictors, target)]

# Define cross-validation method
trainControl <- trainControl(method = "cv", number = 3, savePredictions = "final")

# Function to calculate evaluation metrics
calculateMetrics <- function(pred, actual) {
  rmse <- rmse(actual, pred)
  r2 <- R2(pred, actual)
  mae <- mae(actual, pred)
  mape <- mean(abs((actual - pred) / actual)) * 100
  return(list(RMSE = rmse, R2 = r2, MAE = mae, MAPE = mape))
}

## 4.1 Random Forest Model
set.seed(123)
rfModel <- train(
  yHat ~ ., 
  data = trainData, 
  method = "rf",
  trControl = trainControl,
  tuneGrid = expand.grid(mtry = 3),
  ntree = 25
)

# Retrieve cross-validation predictions and calculate metrics
rfCvPred <- rfModel$pred$pred
rfActual <- rfModel$pred$obs
rfMetrics <- calculateMetrics(rfCvPred, rfActual)
print(rfMetrics)

## 4.2 Gradient Boosting Model
set.seed(123)
gbmModel <- train(
  yHat ~ ., 
  data = trainData, 
  method = "gbm",
  trControl = trainControl,
  tuneGrid = expand.grid(n.trees = 25, interaction.depth = 3, shrinkage = 0.1, n.minobsinnode = 10),
  verbose = FALSE
)

# Retrieve cross-validation predictions and calculate metrics
gbmCvPred <- gbmModel$pred$pred
gbmActual <- gbmModel$pred$obs
gbmMetrics <- calculateMetrics(gbmCvPred, gbmActual)
print(gbmMetrics)

## 4.3 Linear Regression Model
set.seed(123)
lmModel <- train(
  yHat ~ ., 
  data = trainData, 
  method = "lm",
  trControl = trainControl
)

# Retrieve cross-validation predictions and calculate metrics
lmCvPred <- lmModel$pred$pred
lmActual <- lmModel$pred$obs
lmMetrics <- calculateMetrics(lmCvPred, lmActual)
print(lmMetrics)

## 4.4 XGBoost Model
set.seed(123)
xgbModel <- train(
  yHat ~ .,
  data = trainData,
  method = "xgbTree",
  trControl = trainControl,
  tuneGrid = expand.grid(
    nrounds = 25,
    max_depth = 3,
    eta = 0.1,
    gamma = 0,
    colsample_bytree = 0.8,
    min_child_weight = 1,
    subsample = 0.8
  )
)

# Retrieve cross-validation predictions and calculate metrics
xgbCvPred <- xgbModel$pred$pred
xgbActual <- xgbModel$pred$obs
xgbMetrics <- calculateMetrics(xgbCvPred, xgbActual)
print(xgbMetrics)

###############################ASSESS1########################################################

# initial Model fit Comparison on training data 
metricsComparisonInitial <- data.frame(
  Model = c("Random Forest", "Gradient Boosting", "Linear Regression", "XGBoost"),
  RMSE = c(rfMetrics$RMSE, gbmMetrics$RMSE, lmMetrics$RMSE, xgbMetrics$RMSE),
  R2 = c(rfMetrics$R2, gbmMetrics$R2, lmMetrics$R2, xgbMetrics$R2),
  MAE = c(rfMetrics$MAE, gbmMetrics$MAE, lmMetrics$MAE, xgbMetrics$MAE),
  MAPE = c(rfMetrics$MAPE, gbmMetrics$MAPE, lmMetrics$MAPE, xgbMetrics$MAPE)
)
print(metricsComparisonInitial)

# Feature Importance 
lmImportance <- varImp(lmModel)
print(lmImportance)

## Random Forest Feature Importance
rfImportance <- varImp(rfModel)
print(rfImportance)

## XGBoost Feature Importance
xgbImportance <- varImp(xgbModel, scale = FALSE)
print(xgbImportance)

## Gradient Boosting Feature Importance
gbmImportance <- varImp(gbmModel, scale = FALSE)
print(gbmImportance)

#gather unique important features from all models 
uniqueFeatures<-c(
"Education_lev_x_Some_College_minus_Extremely_Likely", 
"DogOwner_lev_x_", 
"EthnicDescription_catN", 
"MedianEducationYears", 
"Education_lev_x_HS_Diploma_minus_Likely", 
"BookBuyerInHome_lev_x_2_book_purchases_in_home", 
"OccupationIndustry_lev_x_Clerical_slash_Office", 
"OccupationIndustry_lev_x_Manufacturing", 
"OccupationIndustry_lev_x_Financial_Services", 
"Education_lev_x_HS_Diploma_minus_Extremely_Likely", 
"BookBuyerInHome_catP", 
"BookBuyerInHome_lev_x_", 
"BookBuyerInHome_lev_x_1_book_purchase_in_home_", 
"OccupationIndustry_lev_x_Skilled_Trades", 
"OccupationIndustry_lev_x_Medical", 
"OccupationIndustry_lev_x_Management", 
"OccupationIndustry_lev_x_Unknown", 
"OccupationIndustry_catP", 
"OccupationIndustry_lev_x_Other", 
"OccupationIndustry_lev_x_Education", 
"BroadEthnicGroupings_catP", 
"ISPSA", 
"BroadEthnicGroupings_catD", 
"EthnicDescription_catP", 
"BroadEthnicGroupings_catN", 
"Education_catP", 
"EthnicDescription_catD", 
"Education_catD", 
"Education_catN", 
"NetWorth_catN", 
"OccupationIndustry_catD", 
"NetWorth_catP", 
"OccupationIndustry_catN", 
"NetWorth_catD", 
"BookBuyerInHome_catD", 
"BookBuyerInHome_catN", 
"PresenceOfChildrenCode_catN", 
"UpscaleBuyerInHome_catN", 
"UpscaleBuyerInHome_catD", 
"UpscaleBuyerInHome_catP", 
"MosaicZ4_catN", 
"Education_lev_x_Bach_Degree_minus_Extremely_Likely", 
"BroadEthnicGroupings_lev_x_Irish", 
"PresenceOfChildrenCode_catP", 
"DogOwner_lev_x_Yes", 
"MosaicZ4_catP")

###############################MODIFY2########################################################

# Keep only the columns in uniqueFeatures and target variable if applicable for each dataset
trainingDataTreated <- trainingDataTreated[, c(uniqueFeatures, "yHat")]
testingDataTreated <- testingDataTreated[, c(uniqueFeatures, "yHat")]
prospectsDataTreated <- prospectsDataTreated[, uniqueFeatures]

#Feature Engineering

# Check the new categories
table(trainingDataTreated$AgeCategory)

# Create a binary indicator for any donation (1 = any donation, 0 = no donation)
trainingDataTreated$AnyDonation <- ifelse(rowSums(trainingData[, c("DonatesEnvironmentCauseInHome",
                                                            "DonatesToCharityInHome",
                                                            "DonatestoAnimalWelfare",
                                                            "DonatestoArtsandCulture",
                                                            "DonatestoChildrensCauses",
                                                            "DonatestoHealthcare",
                                                            "DonatestoInternationalAidCauses",
                                                            "DonatestoVeteransCauses",
                                                            "DonatestoWildlifePreservation",
                                                            "DonatestoLocalCommunity")] == "Yes") > 0, 1, 0)

testingDataTreated$AnyDonation <- ifelse(rowSums(testingData[, c("DonatesEnvironmentCauseInHome",
                                                          "DonatesToCharityInHome",
                                                          "DonatestoAnimalWelfare",
                                                          "DonatestoArtsandCulture",
                                                          "DonatestoChildrensCauses",
                                                          "DonatestoHealthcare",
                                                          "DonatestoInternationalAidCauses",
                                                          "DonatestoVeteransCauses",
                                                          "DonatestoWildlifePreservation",
                                                          "DonatestoLocalCommunity")] == "Yes") > 0, 1, 0)

prospectsDataTreated$AnyDonation <- ifelse(rowSums(prospectsData[, c("DonatesEnvironmentCauseInHome",
                                                              "DonatesToCharityInHome",
                                                              "DonatestoAnimalWelfare",
                                                              "DonatestoArtsandCulture",
                                                              "DonatestoChildrensCauses",
                                                              "DonatestoHealthcare",
                                                              "DonatestoInternationalAidCauses",
                                                              "DonatestoVeteransCauses",
                                                              "DonatestoWildlifePreservation",
                                                              "DonatestoLocalCommunity")] == "Yes", na.rm = TRUE) > 0, 1, 0)
# Check the distribution of the new feature AnyDonation
table(trainingDataTreated$AnyDonation)


# Create a new binary variable HighSpenders
trainingDataTreated$HighSpenders <- ifelse(
  trainingData$UpscaleBuyerInHome %in% c("Yes", "1") | 
    trainingData$BuyerofAntiquesinHousehold %in% c("Yes", "1") | 
    trainingData$BuyerofArtinHousehold %in% c("Yes", "1") | 
    trainingData$GeneralCollectorinHousehold %in% c("Yes", "1") ,
  1,  # High spender if any condition is TRUE
  0   # Not a high spender otherwise
)

testingDataTreated$HighSpenders <- ifelse(
  testingData$UpscaleBuyerInHome %in% c("Yes", "1") | 
    testingData$BuyerofAntiquesinHousehold %in% c("Yes", "1") | 
    testingData$BuyerofArtinHousehold %in% c("Yes", "1") | 
    testingData$GeneralCollectorinHousehold %in% c("Yes", "1") ,
  1,  # High spender if any condition is TRUE
  0   # Not a high spender otherwise
)

prospectsDataTreated$HighSpenders <- ifelse(
  prospectsData$UpscaleBuyerInHome %in% c("Yes", "1") | 
    prospectsData$BuyerofAntiquesinHousehold %in% c("Yes", "1") | 
    prospectsData$BuyerofArtinHousehold %in% c("Yes", "1") | 
    prospectsData$GeneralCollectorinHousehold %in% c("Yes", "1") ,
  1,  # High spender if any condition is TRUE
  0   # Not a high spender otherwise
)

# Check the distribution of the new binary variable
table(trainingDataTreated$HighSpenders)

# Re-binning OccupationIndustry
trainingDataTreated$OccupationRebinned <- dplyr::case_when(
  trainingData$OccupationIndustry %in% c("Computer Professional", "Creative Arts", "Legal", "Scientific") ~ "Professional",
  trainingData$OccupationIndustry %in% c("Food Services", "Maintenance Services", "Skilled Trades") ~ "Service-Oriented",
  trainingData$OccupationIndustry %in% c("Management", "Civil Servant") ~ "Leadership/Management",
  trainingData$OccupationIndustry == "Medical" ~ "Healthcare",
  TRUE ~ "Other"
)

testingDataTreated$OccupationRebinned <- dplyr::case_when(
  testingData$OccupationIndustry %in% c("Computer Professional", "Creative Arts", "Legal", "Scientific") ~ "Professional",
  testingData$OccupationIndustry %in% c("Food Services", "Maintenance Services", "Skilled Trades") ~ "Service-Oriented",
  testingData$OccupationIndustry %in% c("Management", "Civil Servant") ~ "Leadership/Management",
  testingData$OccupationIndustry == "Medical" ~ "Healthcare",
  TRUE ~ "Other"
)
prospectsDataTreated$OccupationRebinned <- dplyr::case_when(
  prospectsData$OccupationIndustry %in% c("Computer Professional", "Creative Arts", "Legal", "Scientific") ~ "Professional",
  prospectsData$OccupationIndustry %in% c("Food Services", "Maintenance Services", "Skilled Trades") ~ "Service-Oriented",
  prospectsData$OccupationIndustry %in% c("Management", "Civil Servant") ~ "Leadership/Management",
  prospectsData$OccupationIndustry == "Medical" ~ "Healthcare",
  TRUE ~ "Other"
)


#HorseOwner, CatOwner, DogOwner, OtherPetOwner --> Create consolidated binary feature as individually dont add predictive power to BBY
trainingDataTreated$HasPet <- ifelse(
  trainingData$HorseOwner == "Yes" | 
    trainingData$CatOwner == "Yes" | 
    trainingData$DogOwner == "Yes" | 
    trainingData$OtherPetOwner == "Yes", 
  1, 0
)

testingDataTreated$HasPet <- ifelse(
  testingData$HorseOwner == "Yes" | 
    testingData$CatOwner == "Yes" | 
    testingData$DogOwner == "Yes" | 
    testingData$OtherPetOwner == "Yes", 
  1, 0
)

prospectsDataTreated$HasPet <- ifelse(
  prospectsData$HorseOwner == "Yes" | 
    prospectsData$CatOwner == "Yes" | 
    prospectsData$DogOwner == "Yes" | 
    prospectsData$OtherPetOwner == "Yes", 
  1, 0
)

#Consolidate FamilyMagazineInHome, FemaleOrientedMagazineInHome, ReligiousMagazineInHome, GardeningMagazineInHome, CulinaryInterestMagazineInHome, HealthFitnessMagazineInHome, DoItYourselfMagazineInHome, FinancialMagazineInHome to a single continuous variable
# Define the magazine-related columns
magazines <- c(
  "FamilyMagazineInHome", "FemaleOrientedMagazineInHome", 
  "ReligiousMagazineInHome", "GardeningMagazineInHome", 
  "CulinaryInterestMagazineInHome", "HealthFitnessMagazineInHome", 
  "DoItYourselfMagazineInHome", "FinancialMagazineInHome"
)

# Function to extract the number of magazine purchases from text
extractMagazines <- function(column) {
  # Replace NA or empty cells with "0"
  column[is.na(column) | column == ""] <- "0"
  # Extract numeric values and sum them, assuming the text follows a pattern like "1 magazine purchase"
  as.numeric(gsub(" .*", "", column))
}

# Apply the extraction function to each magazine column and sum across all columns
trainingDataTreated$MagazineSubscriptionCount <- rowSums(
  sapply(trainingData[magazines], extractMagazines), 
  na.rm = TRUE
)

testingDataTreated$MagazineSubscriptionCount <- rowSums(
  sapply(testingData[magazines], extractMagazines), 
  na.rm = TRUE
)

prospectsDataTreated$MagazineSubscriptionCount <- rowSums(
  sapply(prospectsData[magazines], extractMagazines), 
  na.rm = TRUE
)

trainData <- na.omit(trainData)

########################################MODEL2########################################################
#Refit Models including engineered features

# Set the target variable and predictors
target <- "yHat"
predictors <- setdiff(names(trainingDataTreated), target)

# Prepare the training and testing datasets
trainData <- trainingDataTreated[, c(predictors, target)]
testData <- testingDataTreated[, c(predictors, target)]

#Check for Missing Values
colSums(is.na(trainData))
colSums(is.na(testData))

# Impute missing values with median
preProcessValues <- preProcess(trainData, method = "medianImpute")
trainData <- predict(preProcessValues, trainData)

preProcessValues <- preProcess(testData, method = "medianImpute")
testData <- predict(preProcessValues, testData)

## 4.1 Random Forest Model
set.seed(123)
rfModelFinal <- train(
  yHat ~ ., 
  data = trainData, 
  method = "rf",
  trControl = trainControl,
  tuneGrid = expand.grid(mtry = 3),
  ntree = 25
)

# Predictions for training data
rfTrainPred <- predict(rfModelFinal, newdata = trainData)
rfTrainActual <- trainData$yHat
rfTrainMetrics <- calculateMetrics(rfTrainPred, rfTrainActual)
print("Random Forest - Train Metrics:")
print(rfTrainMetrics)

# Predictions for test data
rfTestPred <- predict(rfModelFinal, newdata = testData)
rfTestActual <- testData$yHat
rfTestMetrics <- calculateMetrics(rfTestPred, rfTestActual)
print("Random Forest - Test Metrics:")
print(rfTestMetrics)

## 4.2 Gradient Boosting Model
set.seed(123)
gbmModelFinal <- train(
  yHat ~ ., 
  data = trainData, 
  method = "gbm",
  trControl = trainControl, 
  tuneGrid = expand.grid(n.trees = 25, interaction.depth = 3, shrinkage = 0.1, n.minobsinnode = 10),
  verbose = FALSE
)

# Predictions for training data
gbmTrainPred <- predict(gbmModelFinal, newdata = trainData)
gbmTrainActual <- trainData$yHat
gbmTrainMetrics <- calculateMetrics(gbmTrainPred, gbmTrainActual)
print("Gradient Boosting - Train Metrics:")
print(gbmTrainMetrics)

# Predictions for test data
gbmTestPred <- predict(gbmModelFinal, newdata = testData)
gbmTestActual <- testData$yHat
gbmTestMetrics <- calculateMetrics(gbmTestPred, gbmTestActual)
print("Gradient Boosting - Test Metrics:")
print(gbmTestMetrics)

## 4.3 Linear Regression Model
set.seed(123)
lmModelFinal <- train(
  yHat ~ ., 
  data = trainData, 
  method = "lm",
  trControl = trainControl
)

# Predictions for training data
lmTrainPred <- predict(lmModelFinal, newdata = trainData)
lmTrainActual <- trainData$yHat
lmTrainMetrics <- calculateMetrics(lmTrainPred, lmTrainActual)
print("Linear Regression - Train Metrics:")
print(lmTrainMetrics)

# Predictions for test data
lmTestPred <- predict(lmModelFinal, newdata = testData)
lmTestActual <- testData$yHat
lmTestMetrics <- calculateMetrics(lmTestPred, lmTestActual)
print("Linear Regression - Test Metrics:")
print(lmTestMetrics)

## 4.4 XGBoost Model
set.seed(123)
xgbModelFinal <- train(
  yHat ~ ., 
  data = trainData, 
  method = "xgbTree", 
  trControl = trainControl, 
  tuneGrid = expand.grid(
    nrounds = 25, 
    max_depth = 3, 
    eta = 0.1, 
    gamma = 0, 
    colsample_bytree = 0.8, 
    min_child_weight = 1, 
    subsample = 0.8
  )
)

# Predictions for training data
xgbTrainPred <- predict(xgbModelFinal, newdata = trainData)
xgbTrainActual <- trainData$yHat
xgbTrainMetrics <- calculateMetrics(xgbTrainPred, xgbTrainActual)
print("XGBoost - Train Metrics:")
print(xgbTrainMetrics)

# Predictions for test data
xgbTestPred <- predict(xgbModelFinal, newdata = testData)
xgbTestActual <- testData$yHat
xgbTestMetrics <- calculateMetrics(xgbTestPred, xgbTestActual)
print("XGBoost - Test Metrics:")
print(xgbTestMetrics)


########################################Assess#2########################################################
# Create a dataframe to store comparison of metrics for both trainData and testData
metricsComparison <- data.frame(
  Model = c("Random Forest", "Gradient Boosting", "Linear Regression", "XGBoost"),
  Train_RMSE = c(rfTrainMetrics$RMSE, gbmTrainMetrics$RMSE, lmTrainMetrics$RMSE, xgbTrainMetrics$RMSE),
  Test_RMSE = c(rfTestMetrics$RMSE, gbmTestMetrics$RMSE, lmTestMetrics$RMSE, xgbTestMetrics$RMSE),
  Train_R2 = c(rfTrainMetrics$R2, gbmTrainMetrics$R2, lmTrainMetrics$R2, xgbTrainMetrics$R2),
  Test_R2 = c(rfTestMetrics$R2, gbmTestMetrics$R2, lmTestMetrics$R2, xgbTestMetrics$R2),
  Train_MAE = c(rfTrainMetrics$MAE, gbmTrainMetrics$MAE, lmTrainMetrics$MAE, xgbTrainMetrics$MAE),
  Test_MAE = c(rfTestMetrics$MAE, gbmTestMetrics$MAE, lmTestMetrics$MAE, xgbTestMetrics$MAE),
  Train_MAPE = c(rfTrainMetrics$MAPE, gbmTrainMetrics$MAPE, lmTrainMetrics$MAPE, xgbTrainMetrics$MAPE),
  Test_MAPE = c(rfTestMetrics$MAPE, gbmTestMetrics$MAPE, lmTestMetrics$MAPE, xgbTestMetrics$MAPE)
)

# Print the metricsComparison dataframe
print(metricsComparison)

#rf metrics not stable and not suitable for prediction

############################################PREDICT#####################################################

prosData <- prospectsDataTreated[, predictors]  # Ensure predictors are defined


# Gradient Boosting Model Predictions
prosData$gbmPred <- predict(gbmModelFinal, newdata = prosData)

# Linear Regression Model Predictions
prosData$lmPred <- predict(lmModelFinal, newdata = prosData)

# XGBoost Model Predictions
prosData$xgbPred <- predict(xgbModelFinal, newdata = prosData)

prosData$ensemblePred <- rowMeans(prosData[, c("gbmPred", "lmPred", "xgbPred")])

plot(density(prosData$ensemblePred, na.rm = TRUE),
     main = "Density Plot of ensemblePred",
     xlab = "ensemblePred",
     col = "blue",
     lwd = 2)

hist(prosData$ensemblePred,
     main = "Distribution of ensemblePred",
     xlab = "ensemblePred",
     col = "skyblue",
     border = "black",
     breaks = 25)

hist(trainingDataTreated$yHat,
     main = "Distribution of yHat in Training Data",
     xlab = "Actual yHat",
     col = "skyblue",
     border = "black",
     breaks = 25)

summary(prosData$ensemblePred)

# Order the data by ensemblePred in descending order and get the top 100 rows
top100 <- prosData[order(-prosData$ensemblePred), ][1:100, ]

# Write the result to a CSV file
write.csv(top100, "~/Git/GitHub_R/personalFiles/studentTablestop100_by_revenue.csv", row.names = FALSE)
