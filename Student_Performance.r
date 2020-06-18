### Dataset - Students' Academic Performance Dataset - (480 Rows & 17 Columns)

# Class Attribute - Class (Column 17) has 3 levels: Low (0 to 69), Medium (70 to 89) & High(90 to 100)
## Aim of Dataset - To monitor learning progress, learner’s actions and determine grade levels.
# Helps the learning activity providers to determine the learner, activity and objects that describe a learning experience.
## Has Features (Independent Attributes) - 3 Types: 1. Demographic (Gender, Nationality..), 
# 2. Academic Background (Education Stage, Grade level, Section....), 
# 3. Behavioral Features (Raising hands, visited resources, answering survey by parents, parent school satisfaction)
# 4. School Attendance (Absence days >7 OR <7)
# Can be used by educational institutions to improve student performance.

###########################################################################
acad=read.csv((file.choose()))
View(acad)
head(acad) # Returns first 6 tuples and all columns
a1 = acad

# Converting all attributes to numeric
a1[sapply(a1, is.factor)] <- lapply(a1[sapply(a1, is.factor)], as.numeric)
str(a1)
View(a1)

# Checking correlation of attributes by using simple correlation table
a1.cor = cor(a1)
print(a1.cor[,17])

# This shows us that the attributes with strongest correlation to our class attribute 'Class'
# are 'raisedhands' (Negative correlation) and 'VisiTedResources' (Negative correlation) 

# Using these two attributes to calculate the following:

# Minimum, Maximum, Mean, Median, Q1 and Q3
summary(a1$raisedhands)
summary(a1$VisITedResources)

# Mode
# For raisedhands
mode<-(a1$raisedhands)
temp<-table(as.vector((mode)))
names(temp)[temp==max(temp)]
# For visitedresources
mode<-(a1$VisITedResources)
temp<-table(as.vector((mode)))
names(temp)[temp==max(temp)]

# Standard Deviation
sd(a1$raisedhands)
sd(a1$VisITedResources)

# Here we see that the attribute 'raisedhands' has the smaller standard deviation of the two chosen attributes.

###SCATTERPLOT###
plot (a1$Class,a1$raisedhands, main = "Class by Raised Hands" , xlab = "Class (High - Low - Medium)", ylab = "raisedHands", pch = 20, col=" purple")
plot (a1$Class,a1$VisITedResources, main = "Class by Visited Resources" , xlab = "Class (High - Low - Medium)", ylab = "visitedResources", pch = 20, col=" blue")

# Basic Scatterplots show that for attributes:
# Raised Hands: As student raises hands more, the chance of their grades also increases slightly from (Low-Medium) to (Medium-High)
# Visited Resources: As student visits more course content, the likelihood of their grades also improves strongly 
# from (Low-Medium) to (Medium-High)

###########################################################################
### CLASSIFIERS ###

### k NEAREST NEIGHBOR ### -- ISSUES
install.packages("car")
install.packages("mlbench")
install.packages("mboost")
install.packages("textir")
install.packages("class")
install.packages("e1071")

library(textir) ## For standardizing data
library(MASS)  
ak = acad

ind <- sample(2, nrow(ak), replace=TRUE, prob=c(0.25, 0.75))
tdata <- ak[ind==1,]
testData <- ak[ind==2,]
par(mfrow=c(3,3), mai=c(.3,.6,.1,.1))
plot(hlth ~ medicaid, data=ak, col=c(blue(.2),2:6))
plot( adldiff ~ medicaid, data=sample1, col=c(grey(.2),2:6))
n=length(sample1$medicaid)
n
nt=180
set.seed(1) ## to make the calculations reproducible in repeated runs
train <- sample(1:n,nt)##dont change sample here its formula
###Normalize the dataset

x=sample1[,c(2,8)]
x[,1]=(x[,1]-mean(x[,1]))/sd(x[,1])
x[,2]=(x[,2]-mean(x[,2]))/sd(x[,2])

x[1:3,]

library(class)  
nearest1 <- knn(train=x[train,],test=x[-train,],cl=sample1$medicaid[train],k=1)
nearest5 <- knn(train=x[train,],test=x[-train,],cl=sample1$medicaid[train],k=5)
data.frame(sample1[-train],nearest1,nearest5)

## plotting to see the result
par(mfrow=c(1,2))
## plot for k=1 (single) nearest neighbor
plot(x[train,],col=sample1$medicaid[train],cex=.8,main="1-nearest neighbor")
points(x[-train,],bg=nearest1,pch=21,col=grey(.9),cex=1.25)
## plot for k=5 nearest neighbors
plot(x[train,],col=sample1$medicaid[train],cex=.8,main="5-nearest neighbors")
points(x[-train,],bg=nearest5,pch=21,col=grey(.9),cex=1.25)
legend("topright",legend=levels(sample1$medicaid),fill=1:6,bty="n",cex=.75)
## calculate the proportion of correct classifications on this one 
## training set

pcorrn1=100*sum(sample1$medicaid[-train]==nearest1)/(n-nt)
pcorrn5=100*sum(sample1$medicaid[-train]==nearest5)/(n-nt)
pcorrn1
pcorrn5
## cross-validation (leave one out)

pcorr=dim(10)
for (k in 1:10) {
  pred=knn.cv(x,sample1$medicaid,k)
  pcorr[k]=100*sum(sample1$medicaid==pred)/n
}
pcorr

#################################
### NAIVE BAYES ###
library(mlbench)
library(e1071)

# Splitting the dataset into training(25%) and test(75%) subsets
a2 = acad
Y <- sample(2, nrow(a2), replace=TRUE, prob=c(0.25, 0.75))
traina2 <- a2[Y==1,]
testa2 <- a2[Y==2,]

# Removing class attribute from the test data set
testa2$Class <- NULL 

## Using naiveBayes classifier
nB <- naiveBayes(Class~.,data = traina2)
nB
str(nB)
summary(testa2)
nb_test_predict <-predict(nB, testa2[,-10])

## Building confusion matrix
table(pred=nb_test_predict,true=testa2$raisedhands)

### NAIVE BAYES OL###
df<-read.csv("../input/xAPI-Edu-Data.csv")
str(df)
library(caret)
set.seed(1)
trainIndex<-createDataPartition(df$gender, p=.8, list=FALSE)
Train<-df[trainIndex,]
Test<-df[-trainIndex,]
library(e1071)
model<-naiveBayes(Train$gender~., data=Train, laplace=1)
#conditional probabilities
model$tables
#interpretation of absence days' conditional probability
model$table$StudentAbsenceDays
#we can see that the probability that the number of absence days is more than seven given that the student
#is female is 0.2535211 and the probability that the number of absence days is more than seven given that the student
#is male is 0.4756098,so its more for males.The probability that the number of absence days is less than seven given that the student
#is female is 0.7464789 and the probability that the number of absence days is less than seven given that the student
#is male is 0.5243902,so its more for females.
#So,we can understand that females are being absent less then males.

pred<-predict(model, newdata=Test)
confusionMatrix(data=pred, reference=Test$gender, positive="F")
#Accuracy - 0.5729 
#Sensitivity - 0.5429 , so if the person was actually female, 
#the model predictes that she is a female with probability 0.5429   
#Specificity - 0.5902, so if the person was actually male, 
#the model predicted that he is a male with probability 0.5902 

#The ROC curve
library(ROCR)
pr<-predict(model, newdata = Test, type="raw")
P_test<-prediction(pr[,2], Test$gender)
perf<-performance(P_test,"tpr","fpr")
plot(perf)

#The AUC (area under the curve)
performance(P_test,"auc")@y.values
#The AUC is 0.645901639344262 so our model is not so bad

### RANDOM FOREST ###
library(ggplot2)
library(dplyr)
library(gridExtra)
library(stringr)
library(caTools)
library(tidyr)
library(randomForest)
library(class)
library(rpart)
library(rpart.plot)
library(e1071)
library(caret)
library(caTools)
library(party)

acadRF=read.csv((file.choose()))
acadRF$Class <- as.numeric(acadRF$Class)
aRF <- acadRF
str(aRF)

ind <- sample(2, nrow(aRF), replace=TRUE, prob=c(0.25, 0.75))
traindata <- aRF[ind==1,]
testdata <- aRF[ind==2,]

rf <- randomForest(Class ~ ., data=traindata, ntree=50, proximity=TRUE)

table(predict(rf), traindata$Class)
print(rf) # We get a accuracy of 0.63, which means our model is reliable
attributes(rf)

## Plotting the error rates with various number of trees.
plot(rf)

## Importance of the variables can be found using functions importance() and varImpPlot()
importance(rf)
varImpPlot(rf)

# Finally, we test the random forest model on test data, and the result is correlated with the functions table() and margin(). 
# The margin of a data point is as the proportion of votes for the correct class minus maximum proportion of votes for other classes. 
# Here, positive margin implies correct classification.
Pred <- predict(rf, newdata=testdata)
table(Pred, testdata$Class)
plot(margin(rf, testdata$Class))

##################################################################################################
##### HIERARCHICAL CLUSTERING ######
# Def: A set of nested clusters organized as a hierarchical tree

install.packages("cluster")  ## Should be installed by default
install.packages("fpc")  ## For density-based clustering

## Performing hierarchical clustering with hclust()
## We first draw a sample of 40 records from the iris data, so that the clustering plot will 
## not be overcrowded. Same as before, variable Species is removed from the data. After that, 
## we apply hierarchical clustering to the data.

ah = acad
Y <- sample(2, nrow(ah), replace=TRUE, prob=c(0.25, 0.75))
trainah <- ah[Y==1,]
testah <- ah[Y==2,]
testah$Class <- NULL

idx <- sample(1:dim(testah)[1], 40)
ahs <- ah[idx,]
ahs$Class <- NULL

hc <- hclust(dist(ahs), method="ave")

plot(hc, hang = -1, labels=testah$Class[idx])

## Let's cut the tree into 3 clusters
rect.hclust(hc, k=3)
groups <- cutree(hc, k=3)


#################################
###### DBSCAN ######
### DBScan analysis
library(fpc)
county <- read.csv("./AcadPerfKaggle.csv")

county1 <- county[,c("raisedhands", "VisITedResources", "AnnouncementsView")]
ds <- dbscan(county1, eps=0.42, MinPts = 5)
ds
plotcluster(county1, ds$cluster)


# The business decisions that can be taken based on this analysis are education based. For an educational institution, the performance of a student can be measured by estimating their participation in class via 'Raised Hands', usage of resources via 'Visited Resources' and thus being able to predict the Grade level of the student.
# This way the engagement of the student in class and the effect it has on their performance can be estimated and decisions based on this insight can be taken appropriately.
