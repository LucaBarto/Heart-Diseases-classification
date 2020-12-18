# Install all the necessary libraries
if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(ggplot2)) install.packages("ggplot2", repos = "http://cran.us.r-project.org")
if(!require(corrplot)) install.packages("recorrplotadr", repos = "http://cran.us.r-project.org")
if(!require(funModeling)) install.packages("funModeling", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(adabag)) install.packages("rstudioapi", repos = "http://cran.us.r-project.org")
if(!require(plyr)) install.packages("rstudioapi", repos = "http://cran.us.r-project.org")
if(!require(MASS)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(gbm)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(pROC)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(rpart.plot)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(knitr)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(nnet)) install.packages("nnet", repos = "http://cran.us.r-project.org")
if(!require(ggthemes)) install.packages("ggthemes", repos = "http://cran.us.r-project.org")
if(!require(rstudioapi)) install.packages("rstudioapi", repos = "http://cran.us.r-project.org")


library(tidyverse)
library(dplyr)
library(ggplot2)
library(corrplot)
library(funModeling)
library(caret)
library(adabag) 
library(plyr) 
library(MASS)
library(gbm)
library(pROC)
library(rpart.plot)
library(knitr)
library(nnet)
library(ggthemes)
library(scales)
library(rstudioapi)

# Read data from file and save in a data frame
data <- read.csv('https://raw.githubusercontent.com/LucaBarto/Heart-Diseases-classification/main/heart.csv')

# Explore structure
str(data)

# Explore dimension
dim(data)

# Summary of data
summary(data)

# Find correlation between predictors

# Change column name
colnames(data)[1] <- "age"

# Set all numeric outputs to 3 digits
options(digits = 3)

# Check for missing values
map_int(data, function(.x) sum(is.na(.x)))

# Correlation matrix
correlationMatrix <- cor(data[,1:ncol(data) - 1])

# The corrplot package is a graphical display of a correlation matrix, 
# confidence interval or general matrix
corrplot(correlationMatrix, order = "hclust", tl.cex = 1, addrect = 8)

# Find attributes that are highly corrected
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.6)

# Print indexes of highly correlated attributes
highlyCorrelated

# Change target variable to factor
data$target <- as.factor(ifelse(data$target == 1, "yes", "no"))

# Check proportion of data
prop.table(table(data$target))

# Plot distribution of target
ggplot(data, aes(x=target)) +
  geom_bar(fill="blue",alpha=0.5) +
  theme_economist() +
  labs(title="Distribution of target")

# Plotting Numerical Data
plot_num(data, bins=10) 


# Target correlation with predictors
# Plot and facet wrap density plots
data %>% 
  gather("feature", "value", -target) %>%
  ggplot(aes(value, fill = target)) +
  geom_density(alpha = 0.5) +
  xlab("Feature values") +
  ylab("Density") +
  theme(legend.position = "top",
        axis.text.x = element_blank(), axis.text.y = element_blank(),
        legend.title=element_blank()) +
  scale_fill_discrete(labels = c("No", "Yes")) +
  facet_wrap(~ feature, scales = "free", ncol = 3)

# Only few variables are normally distributed

# Principal Component Analysis (PCA)
pca <- prcomp(data[,1:ncol(data) - 1], center = TRUE, scale = TRUE)
plot(pca, type="l")

# Summary of data after PCA
summary(pca)

# We need 12 variables to reach 95% of the variance

pca_df <- as.data.frame(pca$x)
ggplot(pca_df, aes(x=PC1, y=PC2, col=data$target)) + geom_point(alpha=0.5)

# The data of the first 2 components cannot be easly separated into two classes.


# Start training data

# Creation of the partition 80% and 20%
set.seed(1, sample.kind="Rounding")
target_index <- createDataPartition(data$target, times=1, p=0.8, list = FALSE)
train_data <- data[target_index, ]
test_data <- data[-target_index, ]

# Define train control 
fitControl <- trainControl(method = "repeatedcv",
                           number = 10,
                           repeats = 10,
                           ## Estimate class probabilities
                           classProbs = TRUE,
                           ## Evaluate performance using 
                           ## the following function
                           summaryFunction = twoClassSummary)

# For each model we will set up a grid of adjustment parameters which, by 
# adapting to the model and calculating its performance, will allow us 
# to determine the values that provide optimal performance.


##############################################################
#####################Adaptive Boosting########################
##############################################################
# Set up tuning grid
am1_grid <- expand.grid(mfinal = (1:3)*3,         
                         maxdepth = c(1, 3),       
                         coeflearn = c("Zhu"))

# Train model
set.seed(1, sample.kind="Rounding")
am1_model <- train(target~., data=train_data,
                   method = "AdaBoost.M1", 
                   trControl = fitControl, 
                   verbose = FALSE, 
                   tuneGrid = am1_grid,
                   # center, scale - centering and scaling data
                   preProcess = c("center", "scale"), 
                   metric = "ROC")

am1_model

plot(am1_model)

# Predict data
set.seed(1, sample.kind="Rounding")
am1_pred <- predict(am1_model, newdata = test_data)

#Evaluate confusion matrix
am1_confusionMatrix <- confusionMatrix(am1_pred, test_data$target)

am1_confusionMatrix

# Plot 10 most important variables
plot(varImp(am1_model), top=10, main="Top variables Adaptive Boosting")

# Define ROC Curve
am1_rocCurve   <- roc(response = test_data$target,
                      predictor = as.numeric(am1_pred),
                      levels = rev(levels(test_data$target)),
                      plot = TRUE, col = "blue", auc = TRUE)

# Plot ROC curve
plot(am1_rocCurve, print.thres = "best")

##############################################################
############Stochastic Gradient Boosting######################
##############################################################

# Set up tuning grid
gbm_grid <-  expand.grid(interaction.depth = c(1, 5, 9), 
                         n.trees = (1:30)*50, 
                         shrinkage = 0.1,
                         n.minobsinnode = 20)

# Train model
set.seed(1, sample.kind="Rounding")
gbm_model <- train(target~., data=train_data,
                   method = "gbm", 
                   trControl = fitControl, 
                   verbose = FALSE, 
                   tuneGrid = gbm_grid,
                   # center, scale - centering and scaling data
                   preProcess = c("center", "scale"), 
                   metric = "ROC")

gbm_model

plot(gbm_model)

# Predict data
set.seed(1, sample.kind="Rounding")
gbm_pred <- predict(gbm_model, newdata = test_data)

#Evaluate confusion matrix
gbm_confusionMatrix <- confusionMatrix(gbm_pred, test_data$target)

gbm_confusionMatrix

# Plot 10 most important variables
plot(varImp(gbm_model), top=10, main="Top variables Stochastic Gradient Boosting")

# Define ROC Curve
gbm_rocCurve   <- roc(response = test_data$target,
                      predictor = as.numeric(gbm_pred),
                      levels = rev(levels(test_data$target)),
                      plot = TRUE, col = "blue", auc = TRUE)

# Plot ROC curve
plot(gbm_rocCurve, print.thres = "best")

############################################################
###############Classification Trees#########################
############################################################

# Set up tuning grid
ct_grid <-  data.frame(cp = seq(0.0, 0.1, len = 25))

# Train model
set.seed(1, sample.kind="Rounding")
ct_model <- train(target~., data=train_data,
                 method = "rpart",
                 metric="ROC",
                 # center, scale - centering and scaling data
                 preProcess = c("center", "scale"), 
                 tuneGrid = ct_grid,
                 trControl = fitControl)

ct_model

plot(ct_model)

# Predict data
set.seed(1, sample.kind="Rounding")
ct_pred <- predict(ct_model, newdata = test_data)

# Evaluate confusion matrix
ct_confusionMatrix <- confusionMatrix(ct_pred, test_data$target)

ct_confusionMatrix

# Plot 10 most important variables
plot(varImp(ct_model), top=10, main="Top variables Classification Tree")

# Define ROC Curve
ct_rocCurve   <- roc(response = test_data$target,
                    predictor = as.numeric(ct_pred),
                    levels = rev(levels(test_data$target)),
                    plot = TRUE, col = "blue", auc = TRUE)

# Plot ROC curve
plot(ct_rocCurve, print.thres = "best")

# The graph below shows the decision tree
rpart.plot(ct_model$finalModel)

###########################################################
#################Random Forest#############################
###########################################################

# Set up tuning grid
rf_grid <-  data.frame(mtry = seq(1, 10))

# Train model
set.seed(1, sample.kind="Rounding")
rf_model <- train(target~., data=train_data,
                  method = "rf",
                  metric = "ROC",
                  # center, scale - centering and scaling data
                  preProcess = c("center", "scale"), 
                  tuneGrid = rf_grid,
                  ntree = 100,
                  trControl = fitControl)

rf_model

plot(rf_model)

# Predict data
set.seed(1, sample.kind="Rounding")
rf_pred <- predict(rf_model, newdata = test_data)

# Evaluate confusion matrix
rf_confusionMatrix <- confusionMatrix(rf_pred, test_data$target)

rf_confusionMatrix

# Plot 10 most important variables
plot(varImp(rf_model), top=10, main="Top variables Random Forest")

# Define ROC Curve
rf_rocCurve   <- roc(response = test_data$target,
                  predictor = as.numeric(rf_pred),
                  levels = rev(levels(test_data$target)),
                  plot = TRUE, col = "blue", auc = TRUE)

# Plot ROC curve
plot(rf_rocCurve, print.thres = "best")


###############################################################
#############K Nearest Neighbor (KNN) Model####################
###############################################################

# Set up tuning grid
knn_grid <-  data.frame(k = seq(1, 50, 1))

# Train model
set.seed(1, sample.kind="Rounding")
knn_model <- train(target~., data=train_data,
                   method="knn",
                   metric="ROC",
                   # center, scale - centering and scaling data
                   preProcess = c("center", "scale"), 
                   tuneGrid = knn_grid,
                   trControl=fitControl)

knn_model

plot(knn_model)

# Predict data
set.seed(1, sample.kind="Rounding")
knn_pred <- predict(knn_model, newdata = test_data)

# Evaluate confusion matrix
knn_confusionMatrix <- confusionMatrix(knn_pred, test_data$target)

knn_confusionMatrix

# Plot 10 most important variables
plot(varImp(knn_model), top=10, main="Top variables K Nearest Neighbor (KNN) Model")

# Define ROC Curve
knn_rocCurve   <- roc(response = test_data$target,
                      predictor = as.numeric(knn_pred),
                      levels = rev(levels(test_data$target)),
                      plot = TRUE, col = "blue", auc = TRUE)

# Plot ROC curve
plot(knn_rocCurve, print.thres = "best")

###############################################################
#####################Neural Network############################
###############################################################

# Set up tuning grid
nn_grid <- expand.grid(size = c(1:5, 10),
                       decay = c(0, 0.05, 0.1, 1, 2))

# Train model
set.seed(1, sample.kind="Rounding")
nn_model <- train(target~., data=train_data,
                  method="nnet",
                  metric="ROC",
                  # center, scale - centering and scaling data
                  preProcess = c("center", "scale"), 
                  tuneGrid = nn_grid,
                  trace=FALSE,
                  trControl=fitControl)

nn_model

plot(nn_model)

# Predict data
set.seed(1, sample.kind="Rounding")
nn_pred <- predict(nn_model, newdata = test_data)

# Evaluate confusion matrix
nn_confusionMatrix <- confusionMatrix(nn_pred, test_data$target)

nn_confusionMatrix

# Plot 10 most important variables
plot(varImp(nn_model), top=10, main="Top variables Neural Network Model")

# Define ROC Curve
nn_rocCurve   <- roc(response = test_data$target,
                      predictor = as.numeric(nn_pred),
                      levels = rev(levels(test_data$target)),
                      plot = TRUE, col = "blue", auc = TRUE)

# Plot ROC curve
plot(nn_rocCurve, print.thres = "best")


##################################################
#############Compare algorithms###################
##################################################

#List of all algorithms
models_list <- list(Adapt_Boost = am1_model,
                    Gradient_Bost=gbm_model,
                    Class_Tree = ct_model,
                    Random_Forest=rf_model,
                    KNN=knn_model,
                    Neural_Network=nn_model) 

models_results <- resamples(models_list)

# Summary of algorithms
summary(models_results)

# Confusion matrix of the algorithms
confusion_matrix_list <- list(
  Adapt_Boost=am1_confusionMatrix, 
  Gradient_Bost=gbm_confusionMatrix,
  Class_Tree = ct_confusionMatrix,
  Random_Forest=rf_confusionMatrix,
  KNN=knn_confusionMatrix,
  Neural_Network=nn_confusionMatrix) 

confusion_matrix_list_results <- sapply(confusion_matrix_list, function(x) x$byClass)
confusion_matrix_list_results %>% kable()


