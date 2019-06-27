#####Loading libraries####
pacman::p_load(Amelia, C50, class, corrplot, FactoMineR, MASS, RMySQL, backports, caret, cellranger, corrplot, doParallel,
               dplyr, e1071, factoextra, foreach, forecast, GGally, ggfortify, ggplot2, gmodels, inum, kknn, padr, party, plotly, plyr, psych, 
               randomForest, readr, reshape, reshape2, rio, rmeta, rstudioapi, scatterplot3d, stats, stringr, tidyverse, tidyverse, utiml)

# Import datasets

current_path <- getActiveDocumentContext()$path
setwd(dirname(current_path))

train_WAP <- read.csv("UJIndoorLoc/trainingData.csv", stringsAsFactors = F)
val_WAP <- read.csv("UJIndoorLoc/validationData.csv", stringsAsFactors = F)

# Explore datasets

str(train_WAP) # all numerical variables

train_WAP$BUILDINGID <- factor(train_WAP$BUILDINGID, 
                               levels = c("0", "1", "2")) # factorize building
train_WAP$RELATIVEPOSITION <- factor(train_WAP$RELATIVEPOSITION,
                                     levels = c("1", "2"))
train_WAP$FLOOR <- factor(train_WAP$FLOOR,
                          levels = c("0", "1", "2", "3", "4"))

val_WAP$BUILDINGID <- factor(val_WAP$BUILDINGID, 
                             levels = c("0", "1", "2")) # replicate in validation dataset
val_WAP$RELATIVEPOSITION <- factor(val_WAP$RELATIVEPOSITION,
                                   levels = c("1", "2"))
val_WAP$FLOOR <- factor(val_WAP$FLOOR,
                        levels = c("0", "1", "2", "3", "4"))

# Change column names for training and models based on column variables
colnames(train_WAP)[names(train_WAP) == "BUILDINGID"] <-  "TRUE_BUILDINGID"
colnames(train_WAP)[names(train_WAP) == "LONGITUDE"] <- "TRUE_LONGITUDE"
colnames(train_WAP)[names(train_WAP) == "LATITUDE"] <- "TRUE_LATITUDE"
colnames(train_WAP)[names(train_WAP) == "FLOOR"] <- "TRUE_FLOOR"
colnames(train_WAP)[names(train_WAP) == "SPACEID"] <- "TRUE_SPACEID"
colnames(train_WAP)[names(train_WAP) == "RELATIVEPOSITION"] <- "TRUE_RELATIVEPOSITION"

colnames(val_WAP)[names(val_WAP) == "BUILDINGID"] <-  "TRUE_BUILDINGID"
colnames(val_WAP)[names(val_WAP) == "LONGITUDE"] <- "TRUE_LONGITUDE"
colnames(val_WAP)[names(val_WAP) == "LATITUDE"] <- "TRUE_LATITUDE"
colnames(val_WAP)[names(val_WAP) == "FLOOR"] <- "TRUE_FLOOR"
colnames(val_WAP)[names(val_WAP) == "SPACEID"] <- "TRUE_SPACEID"
colnames(val_WAP)[names(val_WAP) == "RELATIVEPOSITION"] <- "TRUE_RELATIVEPOSITION"

summary(train_WAP) # NAs being shown as "100" could be a  problem

train_WAP$USERID <- NULL # getting rid of identifier as it does not provide useful info and could lead to erroneous findings/overfitting
val_WAP$USERID <- NULL # same thing on validation dataset as above

train_WAP$PHONEID <- NULL # same as USERID, not useful
val_WAP$PHONEID <- NULL # same as USERID, not useful

train_WAP$TIMESTAMP <- NULL # TIMESTAMP seems to also be potentially confusing for our prediction so we will remove it, could be used to find duplicates so maybe later on we will keep it/ add again  
val_WAP$TIMESTAMP <- NULL # same thing on val dataset as above

# Dummify factor variables
#dummyVars() # SEE MULTIPLE REGRESSION IN R TASK


# Create WAP-only datasets

train_WAP1 <- train_WAP[,1:520]

val_WAP1 <- val_WAP[,1:520]

# NAs with "100" are out of scale and provide no predictive information, 
# Near Zero Variance gets rid of those non-informative predictors as well as 
# those with potentially the same value accross the dataset

nzv <-  nearZeroVar(train_WAP1, saveMetrics = T)

summary(nzv)

# After inspecting the nzv result, we can find out the cut-off value from which we will discriminate

x <- 0.0100316

dim(nzv[nzv$percentUnique > x,]) # size of the remaining (not discriminated) variables

colz <- c(rownames(nzv[nzv$percentUnique > x,])) # Creating string to apply to datasets

train_WAP1 <- cbind(train_WAP[,colz], train_WAP[,521:526]) # Only including features without near zero value AND labels
val_WAP1 <- cbind(val_WAP[,colz], val_WAP[,521:526]) # replicating in validation dataset

#new_train_WAP1 <- as.data.frame(train_WAP1[,colz]) # Only including features without near zero value
#new_val_WAP1 <- as.data.frame(val_WAP1[,colz]) # applying same rules to the validation set

# removing attributes that are no longer necessary
rm(nzv, colz, x)

# Normalizing data

# normalize <- function(x) { # Manual function to normalize the data since all values are numerical 
#   return((x - min(x))/(max(x) - min(x)))
# }
# 
# normalize(c(1, 2, 3, 4, 5)) # testing the normalize function
# 
# new_train_WAP_n <- as.data.frame(lapply(new_train_WAP1, normalize))
# 
# new_val_WAP_n <- as.data.frame(lapply(new_train_WAP1, normalize))

# # Reducing size of data frame to test initial model
# train_sample <- train_WAP1 %>% group_by(TRUE_LONGITUDE, TRUE_LATITUDE, TRUE_FLOOR, TRUE_BUILDINGID) %>% sample_frac(0.3)

# # Train KNN Model for BUILDINGID ####
# system.time(knn_BU <- train.kknn(formula = train_WAP1$TRUE_BUILDINGID~., 
#                                 data = train_WAP1[,1:428], kmax = 100, 
#                                 kernel = "optimal"))
# # saving the model KNN_BU
# saveRDS(knn_BU, file = "KNN_BU_Sample_Model.rds")

knn_BU <- readRDS("Models/KNN_BU_Sample_Model.rds")

# Test Prediction KNN_BU
val_WAP1$BUILDINGID <- predict(knn_BU, newdata = val_WAP1)

train_WAP1$BUILDINGID <- predict(knn_BU, newdata = train_WAP1) # Including prediction column in original train sample

CrossTable(x = train_WAP1$TRUE_BUILDINGID, y = train_WAP1$BUILDINGID,
           prop.chisq = T) # Confusion matrix to check errors

# Prediction Results KNN_BU
postResample(val_WAP1$BUILDINGID, val_WAP1$TRUE_BUILDINGID) # *Accuracy 0.978* *Kappa 0.965*

postResample(train_WAP1$BUILDINGID, train_WAP1$TRUE_BUILDINGID) # *Accuracy 0.998* *Kappa 0.997*

# # Train KNN Model for Altitude (FLOOR) ####
# 
# system.time(knn_FL <- train.kknn(formula = train_WAP1$TRUE_FLOOR~., 
#                                  data = train_WAP1[,c(1:428)], 
#                                  kmax = 100, kernel = "optimal",
#                                  preProcess = c("center", "scale")))
# # saving the model KNN_FL
# saveRDS(knn_FL, file = "KNN_FL_Sample_Model.rds")

knn_FL <- readRDS("Models/KNN_FL_Sample_Model.rds")

# Test Prediction KNN_FL
val_WAP1$FLOOR <- predict(knn_FL, newdata = val_WAP1)

train_WAP1$FLOOR <- predict(knn_FL, newdata = train_WAP1)

CrossTable(x = val_WAP1$TRUE_FLOOR, y = val_WAP1$FLOOR,
           prop.chisq = T) # Confusion matrix to check errors

CrossTable(x = train_WAP1$TRUE_FLOOR, y = train_WAP1$FLOOR,
           prop.chisq = T) # Confusion matrix to check errors 

# Prediction Results KNN_FL
postResample(val_WAP1$FLOOR, val_WAP1$TRUE_FLOOR) # *Accuracy 0.652* *Kappa 0.533*

postResample(train_WAP1$FLOOR, train_WAP1$TRUE_FLOOR) # *Accuracy 0.987* *Kappa 0.983*

# # Train KNN Model for LONGITUDE ####
# system.time(knn_LO <- train.kknn(formula = train_WAP1$TRUE_LONGITUDE~., 
#                                  data = train_WAP1[,c(1:428,435,436)], 
#                                  kmax = 100, kernel = "optimal", 
#                                  preProcess = c("center", "scale")))
# # saving the model KNN_LO
# saveRDS(knn_LO, file = "KNN_LO_Sample_Model.rds")

knn_LO <- readRDS("Models/KNN_LO_Sample_Model.rds")

# Test Prediction KNN_LO
val_WAP1$LONGITUDE <- predict(knn_LO, newdata = val_WAP1) 

train_WAP1$LONGITUDE <- predict(knn_LO, newdata = train_WAP1) 

# Prediction Results KNN_LO
postResample(val_WAP1$LONGITUDE, val_WAP1$TRUE_LONGITUDE) # *RMSE 27.528* *Rsquared 0.947* *MAE 12.051*

postResample(train_WAP1$LONGITUDE, train_WAP1$TRUE_LONGITUDE) # *RMSE 8.368* *Rsquared 0.995* *MAE 2.853*

# # Train KNN Model for LATITUDE ####
# system.time(knn_LA <- train.kknn(formula = train_WAP1$TRUE_LATITUDE~., 
#                                  data = train_WAP1[,c(1:428,435,436)],
#                                  kmax = 100, kernel = "optimal", 
#                                  preProcess = c("center", "scale")))
# # saving the model KNN_LA
# saveRDS(knn_LA, file = "KNN_LA_Sample_Model.rds")

knn_LA <- readRDS("Models/KNN_LA_Sample_Model.rds")

# Test Prediction KNN_LA
val_WAP1$LATITUDE <- predict(knn_LA, newdata = val_WAP1)

train_WAP1$LATITUDE <- predict(knn_LA, newdata = train_WAP1)

# Prediction Results KNN_LA
postResample(val_WAP1$LATITUDE, val_WAP1$TRUE_LATITUDE) # *RMSE 20.14* *Rsquared 0.92* *MAE 10.90*

postResample(train_WAP1$LATITUDE, train_WAP1$TRUE_LATITUDE) # *RMSE 5.929* *Rsquared 0.992* *MAE 2.784*

# # Train RF Model for BUILDINGID ####
# system.time(RF_BU <- randomForest(train_WAP1$TRUE_BUILDINGID~.,
#                                   data = train_WAP1[,c(1:428)],
#                                   ntree = 77, 
#                                   preProcess = c("center", "scale")))
# # saving the model RF_BU
# saveRDS(RF_BU, file = "RF_BU_Sample_Model.rds")

RF_BU <- readRDS("Models/RF_BU_Sample_Model.rds")

# Test Prediction RF_BU
val_WAP1$BUILDINGID <- predict(RF_BU, newdata = val_WAP1)

train_WAP1$BUILDINGID <- predict(RF_BU, newdata = train_WAP1)

# Prediction Results RF_BU 
postResample(val_WAP1$BUILDINGID, val_WAP1$TRUE_BUILDINGID) # *Accuracy 0.999* *Kappa 0.998*

postResample(train_WAP1$BUILDINGID, train_WAP1$TRUE_BUILDINGID) # *Accuracy 0.998* *Kappa 0.997*

# # Train RF Model for Altitude (FLOOR) ####
# system.time(RF_FL <- randomForest(train_WAP1$TRUE_FLOOR~.,
#                                   data = train_WAP1[,c(1:428,435)],
#                                   ntree = 77, 
#                                   preProcess = c("center", "scale")))
# # saving the model RF_FL
# saveRDS(RF_FL, file = "RF_FL_Sample_Model.rds")

RF_FL <- readRDS("Models/RF_FL_Sample_Model.rds")

# Test Prediction RF_FL
val_WAP1$FLOOR <- predict(RF_FL, newdata = val_WAP1)

train_WAP1$FLOOR <- predict(RF_FL, newdata = train_WAP1)

# Prediction Results RF_FL
postResample(val_WAP1$TRUE_FLOOR, val_WAP1$FLOOR) # *Accuracy 0.841* *Kappa 0.780*

postResample(train_WAP1$TRUE_FLOOR, train_WAP1$FLOOR) # *Accuracy 0.997* *Kappa 0.996*

CrossTable(x = val_WAP1$TRUE_FLOOR, y = val_WAP1$FLOOR, 
           prop.chisq = T) # Confusion matrix to check errors

CrossTable(x = train_WAP1$TRUE_FLOOR, y = train_WAP1$FLOOR,
           prop.chisq = T) # Confusion matrix to check errors 

# # Train RF Model for LONGITUDE ####
# system.time(RF_LO <- randomForest(train_WAP1$TRUE_LONGITUDE~., # Should model include previous KNN predictions
#                          data = train_WAP1[,c(1:428,435,436)],
#                          ntree = 77, preProcess = c("center", "scale")))
# # saving the model RF_LO
# saveRDS(RF_LO, file = "RF_LO_Sample_Model.rds")

RF_LO <- readRDS("Models/RF_LO_Sample_Model.rds")

#Test Prediction RF_LO
val_WAP1$LONGITUDE <- predict(RF_LO, newdata = val_WAP1)

train_WAP1$LONGITUDE <- predict(RF_LO, newdata = train_WAP1)

# Prediction Results RF_LO
postResample(val_WAP1$LONGITUDE, val_WAP1$TRUE_LONGITUDE) # *RMSE 12.784* *Rsquared 0.988* *MAE 8.580*

postResample(train_WAP1$LONGITUDE, train_WAP1$TRUE_LONGITUDE) # *RMSE 6.331* *Rsquared 0.997* *MAE 2.342*

# # Train RF Model for LATITUDE ####
# system.time(RF_LA <- randomForest(train_WAP1$TRUE_LATITUDE~., # Should model include previous KNN predictions
#                                   data = train_WAP1[,c(1:428,435,436,437)],
#                                   ntree = 77, 
#                                   preProcess = c("center", "scale")))
# # saving the model RF_LA
# saveRDS(RF_LA, file = "RF_LA_Sample_Model.rds")

RF_LA <- readRDS("Models/RF_LA_Sample_Model.rds")

# Test Prediction RF_LA
val_WAP1$LATITUDE <- predict(RF_LA, newdata = val_WAP1)

train_WAP1$LATITUDE <- predict(RF_LA, newdata = train_WAP1)

# Prediction Results RF_LA
postResample(val_WAP1$TRUE_LATITUDE, val_WAP1$LATITUDE) # *RMSE 11.217* *Rsquared 0.976* *MAE 7.472*

postResample(train_WAP1$TRUE_LATITUDE, train_WAP1$LATITUDE) # *RMSE 4.195* *Rsquared 0.996* *MAE 1.626*

# Train SVM Model for BUILDING ####
system.time(SVM_BU <- svm(train_sample$TRUE_BUILDINGID~., 
                          data = train_sample[,c(1:428)]))

# Test Prediction SVM_BU
val_WAP1$BUILDINGID <- predict(SVM_BU, newdata = val_WAP1)

train_sample$BUILDINGID <- predict(SVM_BU, newdata = train_sample)

# Splitting the dataset by Building ID#

Building_0 <- train_WAP1 %>% filter(BUILDINGID == 0)
Building_1 <- train_WAP1 %>% filter(BUILDINGID == 1)
Building_2 <- train_WAP1 %>% filter(BUILDINGID == 2)

val_Building_0 <- val_WAP1 %>% filter(BUILDINGID == 0)
val_Building_1 <- val_WAP1 %>% filter(BUILDINGID == 1)
val_Building_2 <- val_WAP1 %>% filter(BUILDINGID == 2)

# Drop unused levels for each subset

Building_0$TRUE_FLOOR <- droplevels(Building_0$TRUE_FLOOR)
Building_1$TRUE_FLOOR <- droplevels(Building_1$TRUE_FLOOR)
Building_2$TRUE_FLOOR <- droplevels(Building_2$TRUE_FLOOR)

val_Building_0$TRUE_FLOOR <- droplevels(val_Building_0$TRUE_FLOOR)
val_Building_1$TRUE_FLOOR <- droplevels(val_Building_1$TRUE_FLOOR)
val_Building_2$TRUE_FLOOR <- droplevels(val_Building_2$TRUE_FLOOR)

# Train Split Models using Random Forest ####

# # Floor RF predictions per Building
# system.time(BU0_RF_FL <- randomForest(Building_0$TRUE_FLOOR~.,
#                                       data = Building_0[,c(1:428,435)],
#                                       ntree = 72, preProcess = c("center", "scale")))
# 
# system.time(BU1_RF_FL <- randomForest(Building_1$TRUE_FLOOR~.,
#                                       data = Building_1[,c(1:428,435)],
#                                       ntree = 72, preProcess = c("center", "scale")))
# 
# system.time(BU2_RF_FL <- randomForest(Building_2$TRUE_FLOOR~.,
#                                       data = Building_2[,c(1:428,435)],
#                                       ntree = 72, preProcess = c("center", "scale")))
# # saving the models BUX_RF_FL
#saveRDS(BU0_RF_FL, file = "BU0_RF_FL_Sample_Model.rds")
#saveRDS(BU1_RF_FL, file = "BU1_RF_FL_Sample_Model.rds")
#saveRDS(BU2_RF_FL, file = "BU2_RF_FL_Sample_Model.rds")

# Read saved models
BU0_RF_FL <- readRDS("Models/BU0_RF_FL_Sample_Model.rds")
BU1_RF_FL <- readRDS("Models/BU1_RF_FL_Sample_Model.rds")
BU2_RF_FL <- readRDS("Models/BU2_RF_FL_Sample_Model.rds")

# Test Predictions for Building -> Floor
val_Building_0$FLOOR <- predict(BU0_RF_FL, newdata = val_Building_0)
val_Building_1$FLOOR <- predict(BU1_RF_FL, newdata = val_Building_1)
val_Building_2$FLOOR <- predict(BU2_RF_FL, newdata = val_Building_2)

Building_0$FLOOR <- predict(BU0_RF_FL, newdata = Building_0)
Building_1$FLOOR <- predict(BU1_RF_FL, newdata = Building_1)
Building_2$FLOOR <- predict(BU2_RF_FL, newdata = Building_2)

# Prediction Results RF_FL
postResample(val_Building_0$TRUE_FLOOR, val_Building_0$FLOOR) # *Accuracy 0,944* *Kappa 0,921*
postResample(val_Building_1$TRUE_FLOOR, val_Building_1$FLOOR) # *Accuracy 0,771* *Kappa 0,673*
postResample(val_Building_2$TRUE_FLOOR, val_Building_2$FLOOR) # *Accuracy 0,806* *Kappa 0,735*

postResample(Building_0$TRUE_FLOOR, Building_0$FLOOR) # *Accuracy 0,999* *Kappa 0,999*
postResample(Building_1$TRUE_FLOOR, Building_1$FLOOR) # *Accuracy 0,991* *Kappa 0,988*
postResample(Building_2$TRUE_FLOOR, Building_2$FLOOR) # *Accuracy 1* *Kappa 1*

# Validation Dataset
CrossTable(x = val_Building_0$TRUE_FLOOR, y = val_Building_0$FLOOR, 
           prop.chisq = T) # Confusion matrix to check errors

CrossTable(x = val_Building_1$TRUE_FLOOR, y = val_Building_1$FLOOR, # MANY ERRORS IN FLOOR 2!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
           prop.chisq = T) # Confusion matrix to check errors

CrossTable(x = val_Building_2$TRUE_FLOOR, y = val_Building_2$FLOOR, # MANY ERRORS IN FLOORS 2 & 4!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
           prop.chisq = T) # Confusion matrix to check errors
# Trainset Dataset 
CrossTable(x = Building_0$TRUE_FLOOR, y = Building_0$FLOOR,
           prop.chisq = T) # Confusion matrix to check errors 

CrossTable(x = Building_1$TRUE_FLOOR, y = Building_1$FLOOR,
           prop.chisq = T) # Confusion matrix to check errors 

CrossTable(x = Building_2$TRUE_FLOOR, y = Building_2$FLOOR,
           prop.chisq = T) # Confusion matrix to check errors 

# # Longitude RF Predictions per Building
# system.time(BU0_RF_LO <- randomForest(Building_0$TRUE_LONGITUDE~., # Should model include previous KNN predictions
#                                   data = Building_0[,c(1:428,435,436)],
#                                   ntree = 77, preProcess = c("center", "scale")))
# 
# system.time(BU1_RF_LO <- randomForest(Building_1$TRUE_LONGITUDE~., # Should model include previous KNN predictions
#                                   data = Building_1[,c(1:428,435,436)],
#                                   ntree = 77, preProcess = c("center", "scale")))
# 
# system.time(BU2_RF_LO <- randomForest(Building_2$TRUE_LONGITUDE~., # Should model include previous KNN predictions
#                                   data = Building_2[,c(1:428,435,436)],
#                                   ntree = 77, preProcess = c("center", "scale")))
# saving the models BUX_RF_LO
# saveRDS(BU0_RF_LO, file = "BU0_RF_LO_Sample_Model.rds")
# saveRDS(BU1_RF_LO, file = "BU1_RF_LO_Sample_Model.rds")
# saveRDS(BU2_RF_LO, file = "BU2_RF_LO_Sample_Model.rds")

# Loading saved models
BU0_RF_LO <- readRDS("Models/BU0_RF_LO_Sample_Model.rds")
BU1_RF_LO <- readRDS("Models/BU1_RF_LO_Sample_Model.rds")
BU2_RF_LO <- readRDS("Models/BU2_RF_LO_Sample_Model.rds")

# Test Predictions for Building -> Longitude
val_Building_0$LONGITUDE <- predict(BU0_RF_LO, newdata = val_Building_0)
val_Building_1$LONGITUDE <- predict(BU1_RF_LO, newdata = val_Building_1)
val_Building_2$LONGITUDE <- predict(BU2_RF_LO, newdata = val_Building_2)

Building_0$LONGITUDE <- predict(BU0_RF_LO, newdata = Building_0)
Building_1$LONGITUDE <- predict(BU1_RF_LO, newdata = Building_1)
Building_2$LONGITUDE <- predict(BU2_RF_LO, newdata = Building_2)

# Prediction Results BU_RF_LO
postResample(val_Building_0$TRUE_LONGITUDE, val_Building_0$LONGITUDE) # *RMSE 8,949* *Rsquared 0,896* *MAE 5,990*
postResample(val_Building_1$TRUE_LONGITUDE, val_Building_1$LONGITUDE) # *RMSE 12,423* *Rsquared 0,927* *MAE 8,111*
postResample(val_Building_2$TRUE_LONGITUDE, val_Building_2$LONGITUDE) # *RMSE 14,506* *Rsquared 0,800* *MAE 10,839*

postResample(Building_0$TRUE_LONGITUDE, Building_0$LONGITUDE) # *RMSE 1,430* *Rsquared 0,996* *MAE 0,847*
postResample(Building_1$TRUE_LONGITUDE, Building_1$LONGITUDE) # *RMSE 11,310* *Rsquared 0,950* *MAE 2,505*
postResample(Building_2$TRUE_LONGITUDE, Building_2$LONGITUDE) # *RMSE 3,377* *Rsquared 0,987* *MAE 1,612*

# Latitude RF Predictions per Building
# system.time(BU0_RF_LA <- randomForest(Building_0$TRUE_LATITUDE~., # Should model include previous KNN predictions
#                                   data = Building_0[,c(1:428,435,436,437)],
#                                   ntree = 77, 
#                                   preProcess = c("center", "scale")))
# 
# system.time(BU1_RF_LA <- randomForest(Building_1$TRUE_LATITUDE~., # Should model include previous KNN predictions
#                                   data = Building_1[,c(1:428,435,436,437)],
#                                   ntree = 77, 
#                                   preProcess = c("center", "scale")))
# 
# system.time(BU2_RF_LA <- randomForest(Building_2$TRUE_LATITUDE~., # Should model include previous KNN predictions
#                                   data = Building_2[,c(1:428,435,436,437)],
#                                   ntree = 77, 
#                                   preProcess = c("center", "scale")))
# saving the models BUX_RF_LA
# saveRDS(BU0_RF_LA, file = "BU0_RF_LA_Sample_Model.rds")
# saveRDS(BU1_RF_LA, file = "BU1_RF_LA_Sample_Model.rds")
# saveRDS(BU2_RF_LA, file = "BU2_RF_LA_Sample_Model.rds")

# Loading saved models
BU0_RF_LA <- readRDS("Models/BU0_RF_LA_Sample_Model.rds")
BU1_RF_LA <- readRDS("Models/BU1_RF_LA_Sample_Model.rds")
BU2_RF_LA <- readRDS("Models/BU2_RF_LA_Sample_Model.rds")

# Test Predictions for Building -> LATITUDE
val_Building_0$LATITUDE <- predict(BU0_RF_LA, newdata = val_Building_0)
val_Building_1$LATITUDE <- predict(BU1_RF_LA, newdata = val_Building_1)
val_Building_2$LATITUDE <- predict(BU2_RF_LA, newdata = val_Building_2)

Building_0$LATITUDE <- predict(BU0_RF_LA, newdata = Building_0)
Building_1$LATITUDE <- predict(BU1_RF_LA, newdata = Building_1)
Building_2$LATITUDE <- predict(BU2_RF_LA, newdata = Building_2)

# Prediction Results BU_RF_LA
postResample(val_Building_0$TRUE_LATITUDE, val_Building_0$LATITUDE) # *RMSE 8,097* *Rsquared 0,941* *MAE 5,108*
postResample(val_Building_1$TRUE_LATITUDE, val_Building_1$LATITUDE) # *RMSE 11,718* *Rsquared 0,891* *MAE 8,569*
postResample(val_Building_2$TRUE_LATITUDE, val_Building_2$LATITUDE) # *RMSE 12,405* *Rsquared 0,823* *MAE 8,447*

postResample(Building_0$TRUE_LATITUDE, Building_0$LATITUDE) # *RMSE 1,021* *Rsquared 0,999* *MAE 0,506*
postResample(Building_1$TRUE_LATITUDE, Building_1$LATITUDE) # *RMSE 7,569* *Rsquared 0,957* *MAE 1,805*
postResample(Building_2$TRUE_LATITUDE, Building_2$LATITUDE) # *RMSE 2,030* *Rsquared 0,994* *MAE 0,984*

# Calculate and add error columns####

# Combine subsets
train_WAP <- rbind(Building_0, Building_1, Building_2)
val_WAP <- rbind(val_Building_0, val_Building_1, val_Building_2)

# rm(Building_0, Building_1, Building_2,val_Building_0, val_Building_1, val_Building_2) # remove subsets
rm(BU0_RF_FL, BU1_RF_FL, BU2_RF_FL, BU0_RF_LA, BU1_RF_LA, BU2_RF_LA, BU0_RF_LO, BU1_RF_LO, BU2_RF_LO)

# Convert factors into numeric
val_WAP$FLOOR <- as.integer(val_WAP$FLOOR)
val_WAP$TRUE_FLOOR <- as.integer(val_WAP$TRUE_FLOOR)
val_WAP$BUILDINGID <- as.integer(val_WAP$BUILDINGID)
val_WAP$TRUE_BUILDINGID <- as.integer(val_WAP$TRUE_BUILDINGID)
train_WAP$FLOOR <- as.integer(train_WAP$FLOOR)
train_WAP$TRUE_FLOOR <- as.integer(train_WAP$TRUE_FLOOR)
train_WAP$BUILDINGID <- as.integer(train_WAP$BUILDINGID)
train_WAP$TRUE_BUILDINGID <- as.integer(train_WAP$TRUE_BUILDINGID)

# BUILDING ERROR
val_WAP$BU_ERROR <- (val_WAP$TRUE_BUILDINGID-val_WAP$BUILDINGID)
train_WAP$BU_ERROR <- abs(train_WAP$TRUE_BUILDINGID-train_WAP$BUILDINGID)

# Error column taking floor and building into account

val_WAP$ERROR <- abs(val_WAP$TRUE_LATITUDE-val_WAP$LATITUDE)+
  abs(val_WAP$TRUE_LONGITUDE-val_WAP$LONGITUDE)+
  abs(4*(val_WAP$TRUE_FLOOR-val_WAP$FLOOR))+(abs(val_WAP$BU_ERROR)*50)

train_WAP$ERROR <- abs(train_WAP$TRUE_LATITUDE-train_WAP$LATITUDE)+
  abs(train_WAP$TRUE_LONGITUDE-train_WAP$LONGITUDE)+
  abs(4*(train_WAP$TRUE_FLOOR-train_WAP$FLOOR))+(abs(train_WAP$BU_ERROR)*50)

val_WAP$BU_ERROR <- NULL
train_WAP$BU_ERROR <- NULL

summary(val_WAP$ERROR)
summary(train_WAP$ERROR)

val_WAP$ERROR_LABEL <- ifelse(abs(val_WAP$ERROR) <= 11.1881, "Small", 
                              ifelse(abs(val_WAP$ERROR) > 11.1881 & 
                                       abs(val_WAP$ERROR) <= 19.8934, "Medium",
                                     ifelse(abs(val_WAP$ERROR)> 19.8934 & 
                                              abs(val_WAP$ERROR) <= 50, "Big", "Huge")))

train_WAP$ERROR_LABEL <- ifelse(abs(train_WAP$ERROR) <= 1.1211, "Small", 
                              ifelse(abs(train_WAP$ERROR) > 1.1211 & 
                                       abs(train_WAP$ERROR) <= 2.3971, "Medium",
                                     ifelse(abs(train_WAP$ERROR)> 2.3971 & 
                                              abs(train_WAP$ERROR) <= 6, "Big", "Huge")))

hist(val_WAP$ERROR, main = "Histogram of Error", 
     col = "Gray",
     xlab = "Error value",
     breaks = 50)

plot(density(val_WAP$ERROR))

# Plotting Errors with Plotly####

# Validation ERROR plot:
p <- plot_ly(val_WAP, x = ~LONGITUDE, y = ~LATITUDE, z = ~FLOOR, 
             type = "scatter3d", mode = "markers",
             opacity = 6, color = ~ERROR_LABEL)
p

p3b <- plot_ly(val_WAP, x = ~TRUE_LONGITUDE, y = ~TRUE_LATITUDE, z = ~ TRUE_FLOOR,
               type = "scatter3d", mode = "markers",
               opacity = 6, color = ~ERROR_LABEL)
p3b

# train_WAP plot:
p2 <- plot_ly(train_WAP, x = ~LONGITUDE, y = ~LATITUDE, z = ~FLOOR, 
              type = "scatter3d", mode = "markers",
              opacity = 6, color = ~ERROR_LABEL)
p2

p2b <- plot_ly(train_WAP, x = ~TRUE_LONGITUDE, y = ~TRUE_LATITUDE, z = ~TRUE_FLOOR, 
             type = "scatter3d", mode = "markers",
             opacity = 6, color = ~ERROR_LABEL)
p2b

