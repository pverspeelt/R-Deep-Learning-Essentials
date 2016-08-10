

digits_train <- readr::read_csv("Data/train.csv")

dim(digits_train)
head(colnames(digits_train), 4)
tail(colnames(digits_train), 4)
head(digits_train[, 1:4])

## convert to factor
digits_train$label <- factor(digits_train$label, levels = 0:9)

# reduce the number of observations for the sake of speed.
i <- 1:5000
digits_x <- digits_train[i, -1]
#define digits_y with the dollar sign not with [i, 1]. Tibbles keep their structure unlike data.frame
digits_y <- digits_train$label[i]

# check the distribution
barplot(table(digits_y))


library(caret)
library(nnet)

# start out with 5 hidden neurons
# maximum number of weights at 10000

set.seed(1234)
digits_model1 <- train(x = digits_x, 
                       y = digits_y,
                       method = "nnet",
                       tuneGrid = expand.grid(
                         .size = c(5),
                         .decay = 0.1),
                       trControl = trainControl(method = "none"),
                       MaxNWts = 10000,
                       maxit = 100)

pred_model1 <- predict(digits_model1)
barplot(table(pred_model1))

confusionMatrix(xtabs(~pred_model1 + digits_y))
# accuracy of 44.32%


# increasing from 5 to 10 hidden neurons
# increasing maximum number of weights from 10000 to 50000
set.seed(1234)
starttime <- Sys.time()
digits_model2 <- train(x = digits_x, 
                       y = digits_y,
                       method = "nnet",
                       tuneGrid = expand.grid(
                         .size = c(10),
                         .decay = 0.1),
                       trControl = trainControl(method = "none"),
                       MaxNWts = 50000,
                       maxit = 100)
endtime <- Sys.time()
endtime - starttime
# 2.261349 mins


pred_model2 <- predict(digits_model2)
barplot(table(pred_model2))

confusionMatrix(xtabs(~pred_model2 + digits_y))
# accuracy of 57.64%

