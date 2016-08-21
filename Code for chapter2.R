

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


# nnet -------------------------------------------------

library(caret)
library(nnet)

###########################
# Model 1
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

# predict on training data
pred_model1 <- predict(digits_model1)
barplot(table(pred_model1))

#confusionMatrix on training data
confusionMatrix(xtabs(~pred_model1 + digits_y))
# accuracy of 44.32%


###########################
# Model 2
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

###########################
# Model 3
# increasing from 10 to 40 hidden neurons
# maximum number of weights stays the same
set.seed(1234)
starttime <- Sys.time()
digits_model3 <- train(x = digits_x, 
                       y = digits_y,
                       method = "nnet",
                       tuneGrid = expand.grid(
                         .size = c(40),
                         .decay = 0.1),
                       trControl = trainControl(method = "none"),
                       MaxNWts = 50000,
                       maxit = 100)
endtime <- Sys.time()
endtime - starttime
#  34.63921 mins


pred_model3 <- predict(digits_model3)
barplot(table(pred_model3))

confusionMatrix(xtabs(~pred_model3 + digits_y))
# accuracy of 81.1%

# Model performance 3 5 8 9 still not great

# RSNNS ---------------------------------------------------
# Stuttgart Neural Network Simulator

library(RSNNS)

set.seed(1234)
starttime <- Sys.time()
digits_model4 <- mlp(as.matrix(digits_x),
                 decodeClassLabels(digits_y),
                 size = 40,
                 learnFunc = "Rprop",
                 shufflePatterns = FALSE,
                 maxit = 60)

endtime <- Sys.time()
endtime - starttime
# 2.930157 mins

#  As before, we can get in-sample predictions, but here we have to use another function, fitted.values(). 
# Because this again returns a matrix where each column represents a single digit, 
# we use the encodeClassLabels() function to convert back into a single vector of digit labels to plot 
# and evaluate model performance


fitted_model4 <- fitted.values(digits_model4)
fitted_model4 <- encodeClassLabels(fitted_model4)
barplot(table(pred_model4))

# The only catch is that, when the output is encoded back into a single vector, 
# by default the digits are labeled 1 to k, where k is the number of classes. 
# Because the digits are 0 to 9, to make them match the original digit vector, we subtract 1
caret::confusionMatrix(xtabs(~ I(pred_model4 - 1) + digits_y))
# accuracy of 87.64%



# previous part but done in caret (not in book)
library(caret)
set.seed(1234)
starttime <- Sys.time()
digits_model4a <- train(x = digits_x, 
                        y = digits_y,
                        method = "mlp",
                        tuneGrid = expand.grid(
                           .size = c(40)),
                        trControl = trainControl(method = "none"),
                        learnFunc = "Rprop",
                        shufflePatterns = FALSE,
                        maxit = 60)
endtime <- Sys.time()
endtime - starttime
# 2.936794 mins

pred_model4a <- predict(digits_model4a)
barplot(table(pred_model4a))

confusionMatrix(xtabs(~pred_model4a + digits_y))
# accuracy of 86.26% but t

# Predictions --------------------------

insample_digits_model4 <- fitted.values(digits_model4)
head(round(insample_digits_model4, 2))

# WTA = winner takes all
table(encodeClassLabels(insample_digits_model4,
                        method = "WTA", l = 0, h = 0))

table(encodeClassLabels(insample_digits_model4,
                        method = "WTA", l = 0, h = .5))

table(encodeClassLabels(insample_digits_model4,
                        method = "WTA", l = .2, h = .5))

# 402040 classifies if only one value is above a user-defined threshold, 
# and all other values are below another user-defined threshold
# if multiple values are above the first threshold, 
# or any value is not below the second threshold, 
# it treats the observation as unknown.
table(encodeClassLabels(insample_digits_model4,
                        method = "402040", l = .4, h = .6))


i2 <- 5001:10000
pred_digits_model4 <- predict(digits_model4, 
                             as.matrix(digits_train[i2, -1]))

table(encodeClassLabels(pred_digits_model4,
                        method = "WTA", l = 0, h = 0))

# Over Fitting ---------------
caret::confusionMatrix(xtabs(~ digits_train$label[i2] +
                               I(encodeClassLabels(pred_digits_model4) - 1)))
# accuracy of 83.6%.

# this is 4 percent lower. Model4 is overestimating by 4%.


pred_yhat_model1 <- predict(digits_model1, digits_train$label[i2])
pred_yhat_model2 <- predict(digits_model2, digits_train$label[i2])
pred_yhat_model3 <- predict(digits_model3, digits_train$label[i2])

measures <- c("AccuracyNull", "Accuracy", "AccuracyLower", "AccuracyUpper")

nn5_insample <- caret::confusionMatrix(xtabs(~digits_y + pred_model1))
nn5_outsample <- caret::confusionMatrix(xtabs(~digits_train$label[i2] + pred_yhat_model1))

nn10_insample <- caret::confusionMatrix(xtabs(~digits_y + pred_model2))
nn10_outsample <- caret::confusionMatrix(xtabs(~digits_train$label[i2] + pred_yhat_model2))

nn40_insample <- caret::confusionMatrix(xtabs(~digits_y + pred_model3))
nn40_outsample <- caret::confusionMatrix(xtabs(~digits_train$label[i2] + pred_yhat_model3))

rssn40_insample <- caret::confusionMatrix(xtabs(~digits_y + I(fitted_model4-1)))
rssn40_outsample <- caret::confusionMatrix(xtabs(~digits_train$label[i2] + I(encodeClassLabels(pred_digits_model4) - 1)))


## results
shrinkage <- rbind(
  cbind(Size = 5, Sample = "In", as.data.frame(t(nn5_insample$overall[measures]))),
  cbind(Size = 5, Sample = "Out", as.data.frame(t(nn5_outsample$overall[measures]))),
  cbind(Size = 10, Sample = "In", as.data.frame(t(nn10_insample$overall[measures]))),
  cbind(Size = 10, Sample = "Out", as.data.frame(t(nn10_outsample$overall[measures]))),
  cbind(Size = 40, Sample = "In", as.data.frame(t(nn40_insample$overall[measures]))),
  cbind(Size = 40, Sample = "Out", as.data.frame(t(nn40_outsample$overall[measures]))),
  cbind(Size = 40, Sample = "In", as.data.frame(t(rssn40_insample$overall[measures]))),
  cbind(Size = 40, Sample = "Out", as.data.frame(t(rssn40_outsample$overall[measures])))
)
shrinkage$Pkg <- rep(c("nnet", "RSNNS"), c(6, 2))

dodge <- position_dodge(width=0.4)

p_shrinkage <- ggplot(shrinkage, aes(interaction(Size, Pkg, sep = " : "), Accuracy,
                                     ymin = AccuracyLower, ymax = AccuracyUpper,
                                     shape = Sample, linetype = Sample)) +
  geom_point(size = 2.5, position = dodge) +
  geom_errorbar(width = .25, position = dodge) +
  xlab("") + ylab("Accuracy + 95% CI") +
  theme_classic() +
  theme(legend.key.size = unit(1, "cm"), legend.position = c(.8, .2))


print(p_shrinkage)


