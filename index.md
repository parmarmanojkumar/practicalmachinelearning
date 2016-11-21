# Practical Machine Learning: Project
Manojkumar Parmar  
11/21/2016  


# About Data
## Background
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Data

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. If you use the document you create for this class for any purpose please cite them as they have been very generous in allowing their data to be used for this kind of assignment.

##Goal of the Analysis
The goal of your project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

### Peer Review Portion

Your submission for the Peer Review portion should consist of a link to a Github repo with your R markdown and compiled HTML file describing your analysis. Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page can be viewed online (and you always want to make it easy on graders :-).

# Data Cleaning & Basic analysis

## Getting Data


```r
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
```


## cleaning Data


```r
trainData <- read.csv("./projectdata/pml-training.csv", 
                      na.strings = c("NA","#DIV/0!", ""))
testData <- read.csv("./projectdata/pml-testing.csv",
                     na.strings = c("NA","#DIV/0!", ""))
# delete colums with all missing values
trainData <- trainData[,colSums(is.na(trainData))==0]
testData <- testData[,colSums(is.na(testData))==0]

# remove unnecesary data as it is not relavant
trainData <- trainData[,-c(1:7)]
testData <- testData[,-c(1:7)]
```


## Generating training, testing & validation data


```r
set.seed(123)
inBuild <- createDataPartition(y = trainData$classe, p = 0.8, list = F)
buildData <- trainData[inBuild,]
validation <- trainData[-inBuild,]
inTrain <- createDataPartition(y = buildData$classe, p = 0.75, list = F)
training <- buildData[inTrain,]
testing <- buildData[-inTrain,]
dim(validation)
```

```
## [1] 3923   53
```

```r
dim(training)
```

```
## [1] 11776    53
```

```r
dim(testing)
```

```
## [1] 3923   53
```


# Model Building

## trainControl Parameter

```r
# train control parameter
fitCtrl <- trainControl(method = "cv",number = 7, verboseIter = F, 
                        preProcOptions = c("pca"),
                        allowParallel = T)
```


## To be added


