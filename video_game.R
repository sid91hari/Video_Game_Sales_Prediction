setwd("E:\\Machine Hack\\Video Game Sales Prediction")

# Load libraries
install.packages("dplyr")
library(dplyr)
library(caret)
library(mice)
library(Metrics)
library(MLmetrics)
library(MASS)
library(leaps)

# Load dataset

tr <- read.csv("Train.csv",stringsAsFactors = F)
te <- read.csv("Test.csv",stringsAsFactors = F)

# Checking for blanks

colSums(tr=='')
colSums(te=='')

# Checking NAs
round(colSums(is.na(tr))*100/nrow(tr),2)
round(colSums(is.na(te))*100/nrow(te),2)

# No blanks or NAs present

# Data statistics
str(tr)
summary(tr)

# Checking unique elements across every column
sapply(tr,function(x){length(unique(x))})

#################### EDA ####################

hist(tr$SalesInMillions)

library(rcompanion)

plotNormalHistogram(tr$SalesInMillions)

qqnorm(tr$SalesInMillions,
       ylab="Sales")

qqline(tr$SalesInMillions,
       col="red")

# Sales is right skewed and has outliers

pairs(tr[c('CRITICS_POINTS','USER_POINTS','SalesInMillions')])

boxplot(tr$SalesInMillions)

# Average sales for categorical variables
layout(matrix(c(1,1,1,2,2,2,3,3,3,4,4,4), 4, 3, byrow = TRUE))

fact_cols <- c('CONSOLE','CATEGORY','PUBLISHER','RATING','YEAR')

for (i in fact_cols){
  plot_table <- aggregate(SalesInMillions ~ get(i), data=tr, FUN=mean, na.rm=T)
  plot_table <- plot_table %>% arrange(desc(SalesInMillions))
  names(plot_table)[1] <- i
  barplot(plot_table$SalesInMillions,names = plot_table[,i],
          xlab = i, ylab = "Sales",
          main = paste("Average Sales for",i))
  
}

### By year

table(tr$YEAR)

avg_sales_by_year <- aggregate(SalesInMillions ~ YEAR, data=tr, FUN=mean, na.rm=T)

plot(SalesInMillions ~ YEAR,avg_sales_by_year, type = 'l')
abline(lm(SalesInMillions ~ YEAR,tr),col = "red")

summary(lm(SalesInMillions ~ YEAR,tr))

summary(aov(SalesInMillions~factor(YEAR),tr))


################ Feature Engineering ################

table(tr$CONSOLE)

# 3ds   dc   ds  gba   gc   pc   ps  ps2  ps3  ps4  psp  psv  wii wiiu    x x360 xone 
# 84    4  246  129  164  315   83  585  438  121  180   59  253   45  266  448   85 

library(sqldf)

query1 <- "select *,
case 
when CONSOLE in ('3ds','ds','gba','gc','wii','wiiu') then 'Nintendo'
when CONSOLE like ('ps%') then 'PlayStation'
when CONSOLE like ('x%') then 'Microsoft_Xbox'
when CONSOLE = 'ds' then 'Sega'
else 'Windows' end as PLATFORM
from tr
"

tr <- sqldf(query1)

query2 <- "select *,
case 
when CONSOLE in ('3ds','ds','gba','gc','wii','wiiu') then 'Nintendo'
when CONSOLE like ('ps%') then 'PlayStation'
when CONSOLE like ('x%') then 'Microsoft_Xbox'
when CONSOLE = 'ds' then 'Sega'
else 'Windows' end as PLATFORM
from te
"
te <- sqldf(query2)

summary(tr$CRITICS_POINTS)

summary(tr$USER_POINTS)

tr$TOTAL_POINTS <- tr$CRITICS_POINTS + tr$USER_POINTS
te$TOTAL_POINTS <- te$CRITICS_POINTS + te$USER_POINTS


plotNormalHistogram(tr$CRITICS_POINTS)

plotNormalHistogram(tr$USER_POINTS)

plotNormalHistogram(tr$TOTAL_POINTS)

#######################################################

# Pre Processing

tr_id <- tr$ID
te_id <- te$ID

tr$ID <- NULL
te$ID <- NULL


tr$YEAR <- as.character(tr$YEAR)
te$YEAR <- as.character(te$YEAR)

y <- tr$SalesInMillions

tr$SalesInMillions <- NULL

### Standardizing using Z-score and Box-Cox

preProcValues <- preProcess(tr, method = c("center","scale","BoxCox"))

tr <- predict(preProcValues, tr)

preProcValues <- preProcess(te, method = c("center","scale","BoxCox"))

te <- predict(preProcValues, te)

tr$y <- y

#### Output

par(mfrow=c(2,3))
plotNormalHistogram(tr$CRITICS_POINTS,main = "Histogram of Critics Points - Train")

plotNormalHistogram(tr$USER_POINTS,main = "Histogram of User Points - Train")

plotNormalHistogram(tr$TOTAL_POINTS,,main = "Histogram of Total Points - Train")

plotNormalHistogram(te$CRITICS_POINTS,main = "Histogram of Critics Points - Test")

plotNormalHistogram(te$USER_POINTS,main = "Histogram of User Points - Test")

plotNormalHistogram(te$TOTAL_POINTS,,main = "Histogram of Total Points - Test")

## Processing Categorical Variables

fact_cols <- c('CONSOLE','CATEGORY','PUBLISHER','RATING','YEAR','PLATFORM')


for (i in fact_cols){
  tr[[i]] <- as.factor(tr[[i]])
  te[[i]] <- as.factor(te[[i]])
}



####### Modeling


d <- tr[tr$y < 30,]


# Train control

ctrl <- trainControl(method="cv",     # cross-validation set approach to use
                     number=10,        # k number of times to do k-fold
                     classProbs = FALSE,  # if you want probabilities
                     summaryFunction = defaultSummary,  # for regression
                     allowParallel=T,
                     verboseIter = T
)


# GBM

gbm_grid <- expand.grid(n.trees = seq(from = 500, to = 1000, by = 100), 
                        interaction.depth = c(1:10), 
                        shrinkage = c(0.01,0.05,0.1),
                        n.minobsinnode=10)


set.seed(42)
gbm_model <- train(y~.,
                   data = d,        # train set used to build model
                   method = "gbm",      # type of model you want to build
                   distribution = "gaussian",
                   trControl = ctrl,    # how you want to learn
                   metric = "RMSE",       # performance measure
                   tuneGrid = gbm_grid)


set.seed(42)
gbm_model2 <- train(y~.,
                   data = d,        # train set used to build model
                   method = "gbm",      # type of model you want to build
                   distribution = "gaussian",
                   trControl = ctrl,    # how you want to learn
                   metric = "RMSE",       # performance measure
                   tuneGrid = gbm_grid)


y_pred1_tr <- predict(gbm_model,d)

print(RMSE(y_pred1_tr,d$y))

# 1.805506

y_pred1_tr <- predict(gbm_model2,d)

print(RMSE(y_pred1_tr,d$y))

# 1.094209
# 1.383725

y_pred1 <- predict(gbm_model2,te)

results <- data.frame(y_pred1)

names(results) <- c("SalesInMillions")	

write.table(results, "Submissions\\gbm_1.csv", row.names = F, sep = ",")

# 1.7853683940353038 GBM 1 - Pre processed Label Encoded

# 1.768692634496393 GBM 1 - Pre processed Label Encoded (Outlier 80M removed)

##### XGBoost

ctrl <- trainControl(method="cv",     # cross-validation set approach to use
                     number=10,        # k number of times to do k-fold
                     classProbs = FALSE,  # if you want probabilities
                     summaryFunction = defaultSummary,  # for regression
                     allowParallel=T,
                     verboseIter = T
)

xgb_grid <- expand.grid(nrounds = seq(from = 200, to = 1000, by = 100),
                        max_depth = c(3:8),
                        colsample_bytree = 0.8,
                        eta = c(0.01,0.05),
                        gamma = 0,
                        min_child_weight = c(1,3,5),
                        subsample = 0.8)

set.seed(42)
xgb_model  <- train(y ~ .,               # model specification
                    data = d,        # train set used to build model
                    method = "xgbTree",      # type of model you want to build
                    trControl = ctrl,    # how you want to learn
                    metric = "RMSE",       # performance measure
                    tuneGrid = xgb_grid,
                    seed = 42)



y_pred3_tr <- predict(xgb_model,d)

print(RMSE(y_pred3_tr,d$y))

# 1.015127

y_pred3 <- predict(xgb_model,te)

results <- data.frame(y_pred3)

names(results) <- c("SalesInMillions")	

write.table(results, "Submissions\\xgb_1.csv", row.names = F, sep = ",")

# 1.7563849594088408 XGB 1 - Pre processed Label Encoded (Outlier 80M removed)

########## LightGBM

#Data partition
set.seed(42) # set a seed so you can replicate your results

inTrain <- createDataPartition(y = d$y,   # outcome variable
                               p = .8,   # % of training data you want
                               list = F)

# create your partitions
train <- d[inTrain,]  # training data set
val <- d[-inTrain,]  # test data set

library(lightgbm)

dtrain <- lgb.Dataset(as.matrix(train[1:ncol(train)-1]), label=train$y)
dval <- lgb.Dataset.create.valid(dtrain, data = as.matrix(val[1:ncol(val)-1]), label = val$y)

valids <- list(train = dtrain,test = dval)

lgb.grid = list(objective = "regression",
                metric = "rmse",
                lambda_l1 = 0,
                lambda_l2 = 0)

set.seed(42)
lgb.model.cv = lgb.cv(params = lgb.grid, data = dtrain, learning_rate = 0.005,num_leaves = 50,
                      nrounds = 5000, early_stopping_rounds = 1000,num_threads = 4,
                      nfold = 10,verbose = 1)


best.iter = lgb.model.cv$best_iter

set.seed(42)
lgb.model = lgb.train(params = lgb.grid, data = dtrain, learning_rate = 0.005, num_leaves = 50,
                      nrounds = best.iter,early_stopping_rounds = 1000,num_threads = 4,
                      valids = valids, verbose = 1)

# train's rmse:1.83865	test's rmse:1.84645

y_pred_train <- predict(lgb.model, as.matrix(train[1:ncol(train)-1]))


print(RMSE(y_pred_train,train$y))


y_pred4 <- predict(lgb.model, as.matrix(te))

results <- data.frame(y_pred4)

names(results) <- c("SalesInMillions")	

write.table(results, "Submissions\\lgb_1.csv", row.names = F, sep = ",")

# 1.8148908773963213 Light GBM 1 - Pre processed Label Encoded

# 1.7784682929584807 Light GBM 1 - Pre processed Label encoded Categorical features (Outlier 80M removed)

#### Catboost

library(catboost)

target <- "y"

x <- d[ , -which(names(d) %in% target)]

y <- d[ , which(names(d) %in% target)]

grid <- expand.grid(depth = c(2,4,6,8),
                    learning_rate = c(0.001,0.005,0.01),
                    iterations = seq(from = 100, to = 1000, by = 100),
                    l2_leaf_reg = 0.001,
                    rsm = 1,
                    border_count = 64)


# Label encoding
set.seed(42)
cat_model <- train(x, y,
                   method = catboost.caret,
                   logging_level = 'Verbose',
                   tuneGrid = grid, trControl = ctrl)

y_pred5_tr <- predict(cat_model,d)

print(RMSE(y_pred5_tr,d$y))

# 1.214059 Label Encoded

y_pred5_old <- predict(cat_model,te)

# Still factored
set.seed(42)
cat_model2 <- train(x, y,
                   method = catboost.caret,
                   logging_level = 'Verbose',
                   tuneGrid = grid, trControl = ctrl)


y_pred5_tr <- predict(cat_model2,d)

print(RMSE(y_pred5_tr,d$y))

# 1.376822 Still factored with new features

y_pred5 <- predict(cat_model2,te)

results <- data.frame(y_pred5)

names(results) <- c("SalesInMillions")

write.table(results, "Submissions\\cat_1.csv", row.names = F, sep = ",")


grid2 <- expand.grid(depth = c(6,8),
                     learning_rate = 0.01,
                     iterations = seq(from = 100, to = 1000, by = 100),
                     l2_leaf_reg = 0.001,
                     rsm = 1,
                     border_count = 64)


# Outliers 30M removed
set.seed(42)
cat_model3 <- train(x, y,
                    method = catboost.caret,
                    logging_level = 'Verbose',
                    tuneGrid = grid2, trControl = ctrl)

y_pred5_tr <- predict(cat_model3,d)

print(RMSE(y_pred5_tr,d$y))

# 1.363701

y_pred6 <- predict(cat_model3,te)

results <- data.frame(y_pred6)

names(results) <- c("SalesInMillions")

write.table(results, "Submissions\\cat_2.csv", row.names = F, sep = ",")

# 1.7531312744920167 Catboost 1 - Pre processed Label Encoded (Outlier 80M removed)	

# 1.6968450413943739 Catboost 1 - Pre processed Still Factored (Outlier 80M removed)

# 1.6858222339272744 Catboost 1 - Pre processed Still Factored with new features (Outlier 80M removed)

# 1.6801099187176487 Catboost 2 - Pre processed Still Factored with new features (Outlier 30M removed)



#########################
