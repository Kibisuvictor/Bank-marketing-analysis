## load the data set and working libraries
library(tidyverse)
library(rpart)
library(rpart.plot)

## load the data
library(data.table)
bank <- fread("bank-additional.csv")
bank %>% view()

## structure of the dataset
bank %>% str()

## change all the characters to factors because the levels.
bank <- bank %>% mutate_if(is.character, as.factor)

## looking for missing values 
library(naniar)
bank %>% miss_var_summary()

## exploratory data analysis
## the outcome variable
bank %>% 
  ggplot(aes(y))+
  geom_bar(fill = "blue")

## using data explorer
library(DataExplorer)
bank %>% DataExplorer::plot_histogram()

bank %>% plot_bar()

bank %>% ggplot(aes(y, fill = marital))+
  geom_bar(position = "dodge")
## library caret for splitting and fitting the model
library(rsample) ## does well in splitting the data
splits <- bank %>% initial_split(prop = .7, strata = y)
bank_train <- training(splits) ## training set
table(bank_train$y)
bank_test <- testing(splits) ## testing set

## fitting the decision tree with the default parameters
mod1 <-rpart(y~., data = bank_train)
summary(mod1)
## trying to predict on the training set
p1 <- mod1 %>% predict(bank_train, type= "class")
## a matrix to investigate the false positives and false negatives
confusionMatrix(bank_train$y, p1)

## predicting on the test set
pt1 <- mod1 %>% predict(bank_test, type ="class")
confusionMatrix(bank_test$y, pt1)

## a tree showing the important variables
prp(mod1, extra = 4, faclen = 0, varlen = 0, cex = .75)
library(vip)
vip(mod1)


library(caret)
## using the caret package
## one hot encoding with the caret
dummies <- dummyVars(y~., data = bank_train)
## create the variables using the predict function
train_data <- predict(dummies, newdata = bank_train)

## convert to data frame
train_data<- data.frame(train_data)
str(train_data)

## append the y variable
train_data$y <- bank_train$y

## feature importance
featurePlot(x = train_data[, 1:63],
            y = train_data$y,
            plot = "box",
            strip = strip.custom(par.strip.text = list(cex =.7)),
            scales = list(x = list(relation = "free"),
                          y = list(relation = "free"))
set.seed(100)
subsets <- c(1:10, 20,30,40,50,60, 63)
ctrl <- rfeControl(functions = rfFuncs,
                   method = "repeatedcv",
                   repeats = 5,
                   verbose = FALSE)
prfile <- rfe(x = train_data[, 1:63], y = train_data$y,
              sizes = subsets,
              rfeControl = ctrl)

## train the decision  tree on the train data to get some of the important features
mod2 <- rpart(y~., data = train_data)
vip(mod2)
mod2_preds <- mod2 %>% predict(train_data, type = "class")
confusionMatrix(train_data$y, mod2_preds)
printcp(mod2)

## testing on the test set
## first prepare the data set
test_data <- predict(dummies, bank_test)
test_data <- data.frame(test_data)
test_data$y <- bank_test$y


tst_pred <- mod2 %>% predict(test_data, type = "class")
confusionMatrix(test_data$y, tst_pred)


## tuning the hyperparameters
printcp(mod2)
## getting an index with lowest error
opt <- which.min(mod2$cptable[,"xerror"])

## get its value
cp <- mod2$cptable[opt, "CP"]
cp

## now pruning the tree
pruned_model <- prune(mod2, cp)
r_pruned_pred <- pruned_model %>% predict(test_data, type = "class")
confusionMatrix(test_data$y, r_pruned_pred)

## trying to tune the max depth
mod3 <- rpart(y~., data = train_data, control = rpart.control(maxdepth = 7))
mod3_pred <- mod3 %>% predict(test_data, type="class")
confusionMatrix(test_data$y, mod3_pred)

## using caret to tune the parameters and train a classification model
## tunable parameters
modelLookup("rpart2")

dec_control <- trainControl(method = "repeatedcv",
                            number = 10, # 10-fold cv,
                            classProbs = TRUE)
## train with caret
mod4 <- train(y~., data = train_data,
              method = "rpart2",
              tuneLength = 6,
              trControl = dec_control,
              metric = "Accuracy")
mod4
mod4_preds <- mod4 %>% predict(test_data)
confusionMatrix(test_data$y, mod4_preds)

## using the grid search
tune_g <- expand.grid(maxdepth = 2:10)
mod5 <- train(y~., data = train_data,
              method = "rpart2",
              tuneGrid = tune_g,
              trControl = dec_control,
              metric = "Accuracy")
mod5
mod5_preds <- mod5 %>% predict(test_data)
confusionMatrix(test_data$y, mod5_preds)
