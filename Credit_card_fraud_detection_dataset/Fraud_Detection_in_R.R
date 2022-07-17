# Importing dataset
credit_card <- read.csv("creditcard.csv")

# understanding the Structure of our Dataset
str(credit_card)

# as our 'Class' column is categorical,
# so lets first convert it into factor
credit_card$Class <- factor(credit_card$Class, levels = c(0, 1))

# now lets check if their is any missing values in our dataset
sum(is.na(credit_card)) # so we don't have any missing value


# Now, lets see the distribution of fraud & legit transaction in dataset
table(credit_card$Class)
# so we only have 492 fraud transaction,
# & 284315 legitimate transaction

# so lets create a bar plot for seeing the fraud & legitimate transactions
barplot(table(credit_card$Class), col = c("#0d6eb4","red"))




# Now lets create a small dataset for faster calculation
library(dplyr)
library(ggplot2)

set.seed(1)
credit_card <- credit_card %>% sample_frac(0.1)

table(credit_card$Class)

ggplot(data = credit_card, aes(x=V1, y=V2, col=Class)) +
    geom_point()+
    scale_color_manual(values = c("#2653c5","red"))



# Creating traning & test set
library(caTools)

set.seed(100)

data_sample <- sample.split(credit_card$Class, SplitRatio = 0.8)

train_data <- subset(credit_card, data_sample==TRUE)

test_data <- subset(credit_card, data_sample==FALSE)





# Lets build model using Synthetic Minority Oversampling Technique (SMOTE)
library(smotefamily)

table(train_data$Class)

n0 <- 22750 # legit
n1 <- 35 # fraud
r0 <- 0.6 # ratio we want after smote (i.e. 60% legit and 40% fraud)

ntimes <- ((1 - r0)/r0)*(n0/n1) - 1

smote_output = SMOTE(X = train_data[ , -c(1,31)],
                    target = train_data$Class,
                    K = 5,
                    dup_size = ntimes)

credit_smote <- smote_output$data

colnames(credit_smote)[30] <- "Class"

prop.table(table(credit_smote$Class))


# scatter plot for original train data
ggplot(data = train_data, aes(x=V1, y=V2, col=Class)) +
    geom_point()+
    scale_color_manual(values = c("#2653c5","red"))

# scatter plot for sampled data using smote
ggplot(data = credit_smote, aes(x=V1, y=V2, col=Class)) +
    geom_point()+
    scale_color_manual(values = c("#2653c5","red"))


# now lets draw the decision tree,
# for classifying our transaction into fraud or legitimate
library(rpart)
library(rpart.plot)

model <- rpart(Class ~ . , credit_smote)

rpart.plot(model, extra = 0, type = 5, tweak = 1.2)


# Predict fraud classes in test data
predicted_val <- predict(model, test_data, type="class")

library(caret)
confusionMatrix(predicted_val, test_data$Class)



# Lets Predict for the original dataset
predicted_val <- predict(model, credit_card[,-1], type="class")
confusionMatrix(predicted_val, credit_card$Class)

# Hence using smote, we can actually predict fraud transaction with 98% accuracy