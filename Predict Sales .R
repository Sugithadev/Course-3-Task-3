#Regression Model - Predicting Sales Volume 

#R-squared (R2), representing the squared correlation between the observed outcome values and the predicted values by the model. 
#The higher the adjusted R2, the better the model.
#Root Mean Squared Error (RMSE), which measures the average prediction error made by the model in predicting the outcome for an observation. 
#That is, the average difference between the observed known outcome values and the values predicted by the model. 
#The lower the RMSE, the better the model.
#Mean Absolute Error (MAE), an alternative to the RMSE that is less sensitive to outliers. 
#It corresponds to the average absolute difference between observed and predicted outcomes. The lower the MAE, the better the model


#=================================================================
#load libraries 
#=================================================================
library(readr) 
library(caret)
library(ggplot2)
library(fastDummies)
library(corrplot)
library(GGally)
library(tidyverse)
library(psych)
library(reshape)
library(e1071)
library(rminer)


#=================================================================
#load data 
#=================================================================
#existing product attributes 2017.csv
p <- file.choose()
df_ps <-read.csv(p)
View(df_ps)

#=================================================================
#Pre-processing Data 
#=================================================================
is.na(df_ps)
attributes(df_ps)
summary(df_ps) 
str(df_ps)
names(df_ps)
sum(is.na(df_ps)) # we have na 
# Is each row a repeat?
duplicated(df_ps)
# Show the repeat entries
df_ps[duplicated(df_ps),]


# Create dummy variables:
#Way 1 - noticed that this method converts the columns to int 
#df_ps1 <- dummy_cols(df_ps, select_columns = 'ProductType')
#df_ps1 <- dummy_cols(df_ps, select_columns = c('ProductType'),remove_selected_columns = TRUE)

#way 2 - noticed that this method converts the columns to num 
newDataFrame <- dummyVars(" ~ .", data = df_ps)
readyData <- data.frame(predict(newDataFrame, newdata = df_ps))

View(readyData)
str(readyData)

# remove productnum because this is a unique identifier and will not help in machine learning. 
readyData$ProductNum <- NULL 
# remove BestSellersRank due to domain and also it has na. 
readyData$BestSellersRank <- NULL 
# remove ProfitMargin since after the correlation plot we can confirm it has nothing to do with our volume prediction
readyData$ProfitMargin <- NULL 
readyData$ProductTypeSoftware <- NULL #same reason
#ProductTypeNetbook, ProductTypeSmartphone ,ProductTypePC , and ProductTypeLaptop - needed product types 
readyData$ProductTypeAccessories <- NULL
readyData$ProductTypeDisplay <- NULL
readyData$ProductTypeExtendedWarranty <- NULL
readyData$ProductTypeGameConsole <- NULL
readyData$ProductTypePrinter <- NULL
readyData$ProductTypePrinterSupplies <- NULL
readyData$ProductTypeTablet <- NULL

summary(readyData)


# 5, 4, and 3 star review seems to have strong relation to volume and positive service review 

#=================================================================
#Correlation 
#=================================================================
corrData <- cor(readyData)
corrData
#plot
corrplot(corrData)

#=================================================================
#EDA
#=================================================================
ggplot(data = readyData) + 
  geom_bar(mapping = aes(x = Volume))

qqnorm(readyData$Volume)
qqnorm(readyData$Price)

ggplot(data = readyData, mapping = aes(x = Volume, y = Price)) + 
  geom_boxplot()

plot(readyData)
scatter.smooth(x=readyData$Volume, y=readyData$Price, main="Volume~ Price")

#plots 

ggplot(readyData) +
  geom_point(aes(x = Volume, y = Price, 
                 color = x5StarReviews), size = 4) +
  labs(x = 'Volume', y = "Price") +
  ggtitle("Volume ~ Price, x5StarReviews") +
  theme_bw() +
  theme(axis.text.x = element_text(face = 'bold', size = 10),
        axis.text.y = element_text(face = 'bold', size = 10))


ggplot(readyData) +
  geom_point(aes(x = Volume, y = Price, 
                 color = ShippingWeight), size = 4) +
  labs(x = 'Volume', y = "Price") +
  ggtitle("Volume ~ Price, ShippingWeight ") +
  theme_bw() +
  theme(axis.text.x = element_text(face = 'bold', size = 10),
        axis.text.y = element_text(face = 'bold', size = 10))
#ggpairs
ggpairs(readyData) + theme_bw()
 
pairs(readyData)

pairs(readyData[ , 1:4],
      col = "blue",                                         # Change color
      labels = c("ProductTypeLaptop", "ProductTypeNetbook", "ProductTypePC","ProductTypeSmartphone"),                  # Change labels of diagonal
      main = "This is a nice pairs plot in R")   

upper.panel<-function(x, y){
  points(x,y, pch=19, col=c("red", "green3", "blue"))
  r <- round(cor(x, y), digits=2)
  txt <- paste0("R = ", r)
  usr <- par("usr"); on.exit(par(usr))
  par(usr = c(0, 1, 0, 1))
  text(0.5, 0.9, txt)
}

pairs(readyData[,5:12], lower.panel = NULL, 
      upper.panel = upper.panel)



pairs.panels(readyData[,], 
             method = "pearson", # correlation method
             hist.col = "#00AFBB",
             density = TRUE,  # show density plots
             ellipses = TRUE # show correlation ellipses
)

#Checking distribution of target variable 
ggplot(data=readyData, aes(Volume)) +
  geom_histogram(aes(y =..density..), fill = "orange") +
  geom_density()

#Analyzing Summary Statistics

psych::describe(readyData)

#Checking Outliers Using Boxplots
meltData <- melt(readyData)
p <- ggplot(meltData, aes(factor(variable), value))
p + geom_boxplot() + facet_wrap(~variable, scale="free")

#=================================================================
#Linear Model
#=================================================================

set.seed(100)
index_ps <- createDataPartition(readyData$Volume, p = .80, list = FALSE)
train_ps <- readyData[index_ps, ]
test_ps <- readyData[-index_ps, ]
View(train_ps)
View(test_ps)

# Checking the dim of train
dim(train_ps)

# Build the linear model on training data
lmModel_psd <- lm(Volume ~ . , data = train_ps)
# Printing the model object
print(lmModel_psd)
#Intercept represents that the minimum value of Volume that will be attained, if all the variables are constant or absent.
# Checking model statistics
summary(lmModel_psd)

Pred_ps <- predict(lmModel_psd, test_ps)  # predict volume
View(Pred_ps)

data.frame( R2 = R2(Pred_ps, test_ps$Volume),
            RMSE = RMSE(Pred_ps, test_ps$Volume),
            MAE = MAE(Pred_ps, test_ps$Volume))


# Define training control
train.control.lm <- trainControl(method = "LOOCV")
# Train the model
model.lm <- train(Volume ~., data = readyData, method = "lm",
               trControl = train.control.lm)
# Summarize the results
print(model.lm)

Pred_lm <- predict(model.lm, test_ps)  # predict volume
View(Pred_lm)
summary(Pred_lm)
data.frame( R2 = R2(Pred_lm, test_ps$Volume),
            RMSE = RMSE(Pred_lm, test_ps$Volume),
            MAE = MAE(Pred_lm, test_ps$Volume))
summary(model.lm)$adj.r.squared 

set.seed(123) 
train.control.lm.ps <- trainControl(method = "cv", number = 3)
# Train the model
model.lm.ps <- train(Volume ~., data = readyData, method = "lm",
               trControl = train.control.lm.ps)
# Summarize the results
print(model.lm.ps)

Pred_ps_lm <- predict(model.lm.ps, test_ps)  # predict volume
View(Pred_ps_lm)

data.frame( R2 = R2(Pred_ps_lm, test_ps$Volume),
            RMSE = RMSE(Pred_ps_lm, test_ps$Volume),
            MAE = MAE(Pred_ps_lm, test_ps$Volume))

#Residual plots - https://www.statology.org/residual-plot-r/ 

residuals_ps <- residuals(lmModel_psd)
residuals_ps
res <- resid(lmModel_psd)
plot(fitted(lmModel_psd), res)
#add a horizontal line at 0 
abline(0,0)
#The x-axis displays the fitted values and the y-axis displays the residuals

qqnorm(res)
#add a straight diagonal line to the plot
qqline(res) 

# since the linear model works for parametric data type this existing product attribute data frame will not work in linear model

#=================================================================
#Split the data - 80% and 20%
#=================================================================

set.seed(100)

ps_intraining <- createDataPartition(readyData$Volume, p = .80, list = FALSE)
ps_training <- readyData[ps_intraining,]
ps_testing <- readyData[-ps_intraining,]
View(ps_training)
View(ps_testing)

#=================================================================
#Examine the proportions of the brand
#=================================================================
prop.table(table(readyData$Volume))
prop.table(table(ps_training$Volume))
prop.table(table(ps_testing$Volume))


#=================================================================
#SVM - EPS  Regressesion #https://www.youtube.com/watch?v=8qsFI22c5Lk
#=================================================================
set.seed(123)

ps_svmlinear<- svm(Volume~.,data=ps_training,type="eps-regression",kernel="linear",cross=65)
ps_svmpoly<- svm(Volume~.,data=ps_training,type="eps-regression",kernel="polynomial",cross=65)
ps_svmrad<- svm(Volume~.,data=ps_training,type="eps-regression",kernel="radial",cross=65)
ps_svmsig<- svm(Volume~.,data=ps_training,type="eps-regression",kernel="sigmoid",cross=65)
summary(ps_svmlinear)
summary(ps_svmpoly)
summary(ps_svmrad)
summary(ps_svmsig)

res1 <-resid(ps_svmlinear)
plot(fitted(ps_svmlinear), res1)
#add a horizontal line at 0 
abline(0,0)

#=================================================================
#Prediction - SVM eps regression 
#=================================================================

predictlinear_ps<- predict(ps_svmlinear,ps_testing)
predictpoly_ps<- predict(ps_svmpoly,ps_testing)
predictrad_ps<- predict(ps_svmrad,ps_testing)
predictsig_ps<- predict(ps_svmsig,ps_testing)

predictlinear_ps
predictpoly_ps
predictrad_ps
predictsig_ps

plot(ps_testing$Volume,predictlinear_ps)

cor(ps_testing$Volume,predictlinear_ps)^2
cor(ps_testing$Volume,predictpoly_ps)^2
cor(ps_testing$Volume,predictrad_ps)^2
cor(ps_testing$Volume,predictsig_ps)^2

#=================================================================
# SVM - NU Regression 
#=================================================================
set.seed(123)
ps_svmlinearnu<- svm(Volume~.,data=ps_training,type="nu-regression",kernel="linear",cross=65)
ps_svmpolynu<- svm(Volume~.,data=ps_training,type="nu-regression",kernel="polynomial",cross=65)
ps_svmradnu<- svm(Volume~.,data=ps_training,type="nu-regression",kernel="radial",cross=65)
ps_svmsignu<- svm(Volume~.,data=ps_training,type="nu-regression",kernel="sigmoid",cross=65)
summary(ps_svmlinearnu)
summary(ps_svmpolynu)
summary(ps_svmradnu)
summary(ps_svmsignu)


#=================================================================
#Prediction - SVM - nu-regression
#=================================================================

predictlinearnu_ps<- predict(ps_svmlinearnu,ps_testing)
predictpolynu_ps<- predict(ps_svmpolynu,ps_testing)
predictradnu_ps<- predict(ps_svmradnu,ps_testing)
predictsignu_ps<- predict(ps_svmsignu,ps_testing)

predictlinearnu_ps
plot(ps_testing$Volume,predictlinearnu_ps)

cor(ps_testing$Volume,predictlinearnu_ps)^2
cor(ps_testing$Volume,predictpolynu_ps)^2
cor(ps_testing$Volume,predictradnu_ps)^2
cor(ps_testing$Volume,predictsignu_ps)^2

predictlinearnu_ps
predictpolynu_ps
predictradnu_ps
predictsignu_ps


dfsl<-data.frame( method="svm Linear NU",R2 = R2(predictlinearnu_ps, ps_testing$Volume),
                 RMSE = RMSE(predictlinearnu_ps, ps_testing$Volume),
                 MAE = MAE(predictlinearnu_ps, ps_testing$Volume))
dfsp<-data.frame( method="svm Ploy NU",R2 = R2(predictpolynu_ps, ps_testing$Volume),
                 RMSE = RMSE(predictpolynu_ps, ps_testing$Volume),
                 MAE = MAE(predictpolynu_ps, ps_testing$Volume))
dfsr<-data.frame( method="svm radial NU",R2 = R2(predictradnu_ps, ps_testing$Volume),
                 RMSE = RMSE(predictradnu_ps, ps_testing$Volume),
                 MAE = MAE(predictradnu_ps, ps_testing$Volume))
dfss<-data.frame( method="svm sigmoid NU",R2 = R2(predictsignu_ps, ps_testing$Volume),
                 RMSE = RMSE(predictsignu_ps, ps_testing$Volume),
                 MAE = MAE(predictsignu_ps, ps_testing$Volume))


dfsl1<-data.frame( method="svm Linear EPS",R2 = R2(predictlinear_ps, ps_testing$Volume),
                  RMSE = RMSE(predictlinear_ps, ps_testing$Volume),
                  MAE = MAE(predictlinear_ps, ps_testing$Volume))
dfsp2<-data.frame( method="svm Ploy EPS",R2 = R2(predictpoly_ps, ps_testing$Volume),
                  RMSE = RMSE(predictpoly_ps, ps_testing$Volume),
                  MAE = MAE(predictpoly_ps, ps_testing$Volume))
dfsr3<-data.frame( method="svm radial EPS",R2 = R2(predictrad_ps, ps_testing$Volume),
                  RMSE = RMSE(predictrad_ps, ps_testing$Volume),
                  MAE = MAE(predictrad_ps, ps_testing$Volume))
dfss4<-data.frame( method="svm sigmoid EPS",R2 = R2(predictsig_ps, ps_testing$Volume),
                  RMSE = RMSE(predictsig_ps, ps_testing$Volume),
                  MAE = MAE(predictsig_ps, ps_testing$Volume))

dfsl

res1nu <-resid(ps_svmlinearnu)
plot(fitted(ps_svmlinearnu), res1nu)
#add a horizontal line at 0 
abline(0,0)






#variable importance
M <- fit(Volume~., data=ps_training, model="svm",  C=3)
svm.imp <- Importance(M, data=ps_training)

list(runs=1,sen=t(svm.imp),sresponses=svm.imp$sresponses)

#=================================================================
#RF - Regressesion
#=================================================================
#cross validation
train.control.ps <- trainControl(method = "repeatedcv",
                              number = 3,
                              repeats = 1,
                              search = "grid")


rfgrid_ps<-expand.grid(mtry=c(1,2,3,4,5))
set.seed(123)

caret.rf.ps <- train(Volume ~ ., 
                  data = ps_training,
                  method = "rf",
                  tuneGrid = rfgrid_ps,
                  trControl = train.control.ps)
caret.rf.ps

plot(caret.rf.ps)

#variable importance
rfImp_ps <- varImp(caret.rf.ps, scale = FALSE)
rfImp_ps
plot(rfImp_ps)

#=================================================================
#Prediction  - rf 
#=================================================================

preds.rf.ps <- predict(caret.rf.ps, ps_testing)
#confusionMatrix(as.factor(preds.rf.ps), as.factor(ps_testing$Volume))

preds.rf.ps


dfr<-data.frame( method="rf",R2 = R2(preds.rf.ps, ps_testing$Volume),
            RMSE = RMSE(preds.rf.ps, ps_testing$Volume),
            MAE = MAE(preds.rf.ps, ps_testing$Volume))
#         R2     RMSE      MAE
#1 0.8645149 1052.192 329.7855

dfr


residuals_rf <- residuals(caret.rf.ps)
residuals_rf
res1 <- resid(caret.rf.ps)
plot(fitted(caret.rf.ps), res1)
#add a horizontal line at 0 
abline(0,0)
#The x-axis displays the fitted values and the y-axis displays the residuals


#=================================================================
#ParF - Type 2 RF method
#=================================================================
set.seed(123)
caret.parRF.ps <- train(Volume ~ ., 
                     data = ps_training,
                     method = "parRF",
                     tuneGrid = rfgrid_ps,
                     trControl = train.control.ps)
caret.parRF.ps
plot(caret.parRF.ps)


#=================================================================
#Prediction  - parrf 
#=================================================================

preds.parRF.ps <- predict(caret.parRF.ps, ps_testing)

preds.parRF.ps

dfp<- data.frame( method="parf",R2 = R2(preds.parRF.ps, ps_testing$Volume),
            RMSE = RMSE(preds.parRF.ps, ps_testing$Volume),
            MAE = MAE(preds.parRF.ps, ps_testing$Volume))


dfp

#         R2     RMSE      MAE
#1 0.7293347 1366.312 572.9089

#=================================================================
#GB - Regression
#=================================================================

set.seed(123)
gbm.ps <- train(Volume ~ ., 
                data = ps_training,
                method = "gbm",
                trControl = train.control.ps)

gbm.ps
plot(gbm.ps)

#=================================================================
#Prediction  - gbm 
#=================================================================
preds.gbm.ps <- predict(gbm.ps, ps_testing)

preds.gbm.ps


dfg<- data.frame( method="gbm",R2 = R2(preds.gbm.ps, ps_testing$Volume),
            RMSE = RMSE(preds.gbm.ps, ps_testing$Volume),
            MAE = MAE(preds.gbm.ps, ps_testing$Volume))
dfg

#         R2     RMSE      MAE
#1 0.3718745 1382.971 599.6776

#calculate resamples
resample_results_ps <- resamples(list(GBM = gbm.ps,RF = caret.rf.ps))
summary(resample_results_ps)

#
newdf <- rbind(dfsl,dfsp,dfsr,dfss,dfsl1,dfsp2,dfsr3,dfss4,dfr,dfp,dfg)

newdf


#=================================================================
#=================================================================

#Predict the new product type using SVM Linear NU Regression 
#=================================================================
#=================================================================
#load new product attributes 2017.csv
q <- file.choose()
df_np <-read.csv(q)
View(df_np)
head(df_np)

#preprocess
is.na(df_np)
attributes(df_np)
summary(df_np) 
str(df_np)
names(df_np)
sum(is.na(df_np)) # we have no na 
# Is each row a repeat?
duplicated(df_np)
# Show the repeat entries
df_np[duplicated(df_np),]


#way 2 - noticed that this method converts the columns to num 
newDataFrame_np <- dummyVars(" ~ .", data = df_np)
readyData_np <- data.frame(predict(newDataFrame_np, newdata = df_np))

View(readyData_np)

# remove productnum because this is a unique identifier and will not help in machine learning. 
readyData_np$ProductNum <- NULL 
# remove BestSellersRank due to domain and also it has na. 
readyData_np$BestSellersRank <- NULL 
# remove ProfitMargin since after the correlation plot we can confirm it has nothing to do with our volume prediction
readyData_np$ProfitMargin <- NULL 
readyData_np$ProductTypeSoftware <- NULL #same reason
#ProductTypeNetbook, ProductTypeSmartphone ,ProductTypePC , and ProductTypeLaptop - needed product types 
readyData_np$ProductTypeAccessories <- NULL
readyData_np$ProductTypeDisplay <- NULL
readyData_np$ProductTypeExtendedWarranty <- NULL
readyData_np$ProductTypeGameConsole <- NULL
readyData_np$ProductTypePrinter <- NULL
readyData_np$ProductTypePrinterSupplies <- NULL
readyData_np$ProductTypeTablet <- NULL

summary(readyData_np)

corrData1 <- cor(readyData_np)
corrData1
#plot
corrplot(corrData1)


#Predict using SVM linear - NU Regression  

predictlinearnu_ps_final<- predict(ps_svmlinearnu,readyData_np)
predictlinearnu_ps_final
data.frame( method="svm Linear NU",R2 = R2(predictlinearnu_ps_final, readyData_np$Volume),
                  RMSE = RMSE(predictlinearnu_ps_final, readyData_np$Volume),
                  MAE = MAE(predictlinearnu_ps_final, readyData_np$Volume))


plot(predictlinearnu_ps_final)

# Overfit 
cor(readyData_np$Volume,predictlinearnu_ps_final)^2

#Predict using RF
preds.rf.np <- predict(caret.rf.ps, readyData_np)
preds.rf.np
data.frame( method="RF",R2 = R2(preds.rf.np, readyData_np$Volume),
            RMSE = RMSE(preds.rf.np, readyData_np$Volume),
            MAE = MAE(preds.rf.np, readyData_np$Volume))
cor(readyData_np$Volume,preds.rf.np)^2

#Third, as the warning messages plainly tell you, some of the vectors you are passing to cor() have zero variance. 
#They have nothing to do with the NaNs: as the following shows, R doesn't complain about standard deviations of 0 when NaN are involved. 
#(Quite sensibly too, since you can't calculate standard deviations for undefined numbers):

#write to csv
output <- df_np 
output$predictions <- predictlinearnu_ps_final
write.csv(output, file="C3.T3output.csv", row.names = TRUE)






