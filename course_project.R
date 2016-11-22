rm(list = ls())
cat('\014')
#load libraries
library(caret)
library(ggplot2)
library(randomForest)
library(knitr)
library(parallel)
library(doParallel)

#Make parallel processing nebale

cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

#step1 : Download data and put it in folder if it is not there
if(!file.exists("./projectdata")){
        dir.create("./projectdata")
}
downloadUrlTrain <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
if(!file.exists("./projectdata/pml-training.csv")){
        download.file(downloadUrlTrain,destfile="./projectdata/pml-training.csv",method="curl")
}
downloadUrlTest <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
if(!file.exists("./projectdata/pml-testing.csv")){
        download.file(downloadUrlTest,destfile="./projectdata/pml-testing.csv",method="curl")
}
#step2 : Read data file
trainData <- read.csv("./projectdata/pml-training.csv", 
                      na.strings = c("NA","#DIV/0!", ""))
testData <- read.csv("./projectdata/pml-testing.csv",
                     na.strings = c("NA","#DIV/0!", ""))

dim(trainData)
dim(testData)

#delete colums with all missing values
trainData <- trainData[,colSums(is.na(trainData))==0]
testData <- testData[,colSums(is.na(testData))==0]

#Remove unnecesary data as it is not relavant
trainData <- trainData[,-c(1:7)]
testData <- testData[,-c(1:7)]

#find correlation
highCor <- findCorrelation(cor(trainData[,-53]), cutoff = 0.8)
names(trainData)[highCor]

#final predictors
names(trainData)

#generate training, testing & validation data
set.seed(123)
inBuild <- createDataPartition(y = trainData$classe, p = 0.8, list = F)
buildData <- trainData[inBuild,]
validation <- trainData[-inBuild,]
predDf <- data.frame(run = 0, time = 0, gbm = 0, rf = 0, svmr = 0, 
                     svml = 0, nn = 0, lb = 0)
start.time.all = Sys.time()
for (i in 1:10){
        inTrain <- createDataPartition(y = buildData$classe, p = 0.75, list = F)
        training <- buildData[inTrain,]
        testing <- buildData[-inTrain,]
        dim(validation)
        dim(training)
        dim(testing)
        #Start building model
        # train control parameter
        fitCtrl <- trainControl(method = "cv",number = 7, verboseIter = F, 
                                preProcOptions = c("pca"),
                                allowParallel = T)
        start.time = Sys.time()
        mod.gbm <- train(classe ~ . , data= training , method = "gbm", 
                         trControl = fitCtrl, verbose = F)
        mod.rf <- train(classe ~ . , data= training , method = "rf", 
                        trControl = fitCtrl, verbose = F)
        mod.svmr <- train(classe ~ . , data= training , method = "svmRadial", 
                          trControl = fitCtrl, verbose = F)
        mod.svml <- train(classe ~ . , data= training , method = "svmLinear", 
                         trControl = fitCtrl, verbose = F)
        mod.nn <- train(classe ~ . , data= training , method = "nnet", 
                        trControl = fitCtrl, verbose = F)
        mod.lb <- train(classe ~ . , data= training , method = "LogitBoost", 
                        trControl = fitCtrl, verbose = F)
        stop.time = Sys.time()
        
        #Predictions
        pred_val <- c( i, (stop.time - start.time),
                        unname(confusionMatrix(predict(mod.gbm, testing), 
                                               testing$classe)$overall[1]),
                        unname(confusionMatrix(predict(mod.rf, testing), 
                                               testing$classe)$overall[1]),
                        unname(confusionMatrix(predict(mod.svmr, testing), 
                                               testing$classe)$overall[1]),
                        unname(confusionMatrix(predict(mod.svml, testing), 
                                               testing$classe)$overall[1]),
                        unname(confusionMatrix(predict(mod.nn, testing), 
                                               testing$classe)$overall[1]),
                        unname(confusionMatrix(predict(mod.lb, testing), 
                                               testing$classe)$overall[1]))
        predDf <- rbind(predDf, pred_val)
}
stop.time.all = Sys.time()
print(stop.time.all - start.time.all)
predDf <- predDf[-1,]
confusionMatrix(predict(mod.rf, validation), validation$classe)$overall[1]
confusionMatrix(predict(mod.gbm, validation), validation$classe)$overall[1]
confusionMatrix(predict(mod.svmr, validation), validation$classe)$overall[1]
# Rf is best 
finMod.rf <- train(classe ~ . , data= trainData , method = "rf", 
                trControl = fitCtrl, verbose = F)
# gbm for agreement accuracy
finMod.gbm <- train(classe ~ . , data= trainData , method = "gbm", 
                   trControl = fitCtrl, verbose = F)

#svmr for agreement accuracy
finMod.svmr <- train(classe ~ . , data= trainData , method = "svmRadial", 
                        trControl = fitCtrl, verbose = F)
#predict from 3 different best model
predFin.rf <- predict(finMod.rf,testData)
predFin.gbm <- predict(finMod.gbm,testData)
predFin.svmr <- predict(finMod.svmr, testData)

#check for agreement accuracy
confusionMatrix(predFin.gbm,predFin.rf)
confusionMatrix(predFin.svmr,predFin.rf)
