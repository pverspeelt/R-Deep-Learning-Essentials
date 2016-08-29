library(glmnet)
library(MASS)

# L1 penalty ------

set.seed(1234)

x <- mvrnorm(n = 200, mu = c(0, 0, 0, 0, 0),
             Sigma = matrix(c(
               1, .9999, .99, .99, .10,
               .9999, 1, .99, .99, .10,
               .99, .99, 1, .99, .10,
               .99, .99, .99, 1, .10,
               .10, .10, .10, .10, 1
             ), ncol = 5))

y <- rnorm(200, 3 + x %*% matrix(c(1, 1, 1, 1, 0)), .5)

m_ols <- lm(y[1:100] ~ x[1:100, ])

m_lasso_cv <- cv.glmnet(x[1:100, ], y[1:100], alpha = 1)

plot(m_lasso_cv)

cbind(
  OLS = coef(m_ols),
  Lasso = coef(m_lasso_cv)[,1])

# L2 penalty ------

m_ridge_cv <- cv.glmnet(x[1:100, ], y[1:100], alpha = 0)

plot(m_ridge_cv)

cbind(
  OLS = coef(m_ols),
  Lasso = coef(m_lasso_cv)[,1],
  Ridge = coef(m_ridge_cv)[,1])


# wight decay => L2 in neural networks ------

# load mnist data
digits_train <- readr::read_csv("Data/train.csv")

## convert to factor
digits_train$label <- factor(digits_train$label, levels = 0:9)

# reduce the number of observations for the sake of speed.
i <- 1:5000
digits_x <- digits_train[i, -1]
digits_y <- digits_train$label[i]



library(caret)
library(nnet)

# vary the weight decay penalty at 0 (no penalty) and 0.10.
# two sets of the number of iterations allowed: 100 or 150
# Use doParallel: caret cross validation can run in parallel. 

library(doParallel)
cl <- makeCluster(4)
registerDoParallel(cl)

set.seed(1234)
starttime <- Sys.time()
decay_models <- lapply(c(100, 150), function(its) {
  train(digits_x, digits_y,
        method = "nnet",
        tuneGrid = expand.grid(
          .size = c(10),
          .decay = c(0, .1)),
        trControl = trainControl(method = "cv", number = 5, repeats = 1),
        MaxNWts = 10000,
        maxit = its)
})
endtime <- Sys.time()
endtime - starttime
# 20.0276 mins


decay_models[[1]]
# decay  Accuracy   Kappa    
# 0.0    0.6281947  0.5866181
# 0.1    0.5914099  0.5451618


decay_models[[2]]
# decay  Accuracy   Kappa    
# 0.0    0.5445942  0.4925507
# 0.1    0.5634120  0.5143569

# interestingly the results are not the same as in the example of the book
# I expected model 2 to have a higher accuracy, like in this chapter, but I 
# actually end up with a lower accuracy. 

# These results highlight the point that regularization is often most helpful with more complex models 
# that have greater flexibility to fit (and overfit) the data, and that 
# (in models that are appropriate or overly simplistic for the data) regularization may actually decrease performance.

# stop cluster for now
stopCluster(cl)


# Model averaging ------

# simulated data
set.seed(1234)
d <- data.frame(
  x = rnorm(400))
d$y <- with(d, rnorm(400, 2 + ifelse(x < 0, x + x^2, x + x^2.5), 1))
d_train <- d[1:200, ]
d_test <- d[201:400, ]

# three different models
m1 <- lm(y ~ x, data = d_train)
m2 <- lm(y ~ I(x^2), data = d_train)
m3 <- lm(y ~ pmax(x, 0) + pmin(x, 0), data = d_train)

# In sample R2
cbind(
  M1 = summary(m1)$r.squared,
  M2 = summary(m2)$r.squared,
  M3 = summary(m3)$r.squared)

# correlations in the training data
cor(cbind(
  M1 = fitted(m1),
  M2 = fitted(m2),
  M3 = fitted(m3)))


# generate predictions and the average prediction
d_test$yhat1 <- predict(m1, newdata = d_test)
d_test$yhat2 <- predict(m2, newdata = d_test)
d_test$yhat3 <- predict(m3, newdata = d_test)
d_test$yhatavg <- rowMeans(d_test[, paste0("yhat", 1:3)])

# correlation in the testing data
cor(d_test)
# From the results we can see that indeed the average of the three 
# models' predictions performs better than any of the models individually

# Bagging and model averaging is not used as frequently in deep neural networks 
# because the computational cost of training each model can be quite high 
# and thus repeating the process many times becomes prohibitively expensive 
# in terms of time and compute resources.

# deepnet: improving out-of-sample model performance using dropout ------

# start cluster again but this time with deepnet available for each worker
cl <- makeCluster(4)
clusterEvalQ(cl, {
  library(deepnet)
  
})
registerDoParallel(cl)

## Fit Models

starttime <- Sys.time()
nn_models <- foreach(i = 1:4, .combine = 'c') %dopar% {
  set.seed(1234)
  list(nn.train(
    x = as.matrix(digits_x),
    y = model.matrix(~ 0 + digits_y),
    hidden = c(40, 80, 40, 80)[i],
    activationfun = "tanh",
    learningrate = 0.8,
    momentum = 0.5,
    numepochs = 150,
    output = "softmax",
    hidden_dropout = c(0, 0, .5, .5)[i],
    visible_dropout = c(0, 0, .2, .2)[i]))
}
endtime <- Sys.time()
endtime - starttime
# 1.285196 mins

# RSNNS and deepnet are not loaded if the code is started from scratch.
# Added package names in front of functions to avoid loading everything

nn_yhat <- lapply(nn_models, function(obj) {
  RSNNS::encodeClassLabels(deepnet::nn.predict(obj, as.matrix(digits_x)))
})

train_performance <- do.call(cbind, lapply(nn_yhat, function(yhat) {
  caret::confusionMatrix(xtabs(~ I(yhat - 1) + digits_y))$overall
}))
colnames(train_performance) <- c("N40", "N80", "N40_Reg", "N80_Reg")

options(digits = 4)
train_performance
#                   N40    N80 N40_Reg N80_Reg
# Accuracy       0.9050 0.9546  0.9212  0.9396
# Kappa          0.8944 0.9495  0.9124  0.9329
# AccuracyLower  0.8965 0.9485  0.9134  0.9326
# AccuracyUpper  0.9130 0.9602  0.9285  0.9460
# AccuracyNull   0.1116 0.1116  0.1116  0.1116
# AccuracyPValue 0.0000 0.0000  0.0000  0.0000
# McnemarPValue     NaN    NaN     NaN     NaN

# When evaluating the models in the in-sample training data, 
# it seems that the 40-neuron model performs better with regularization than without it, 
# but that the 80-neuron model performs better without regularization 
# than with regularization

i2 <- 5001:10000
test_x <- digits_train[i2, -1]
test_y <- digits_train$label[i2]

nn_yhat_test <- lapply(nn_models, function(obj) {
  RSNNS::encodeClassLabels(deepnet::nn.predict(obj, as.matrix(test_x)))
})

test_performance <- do.call(cbind, lapply(nn_yhat_test, function(yhat) {
  caret::confusionMatrix(xtabs(~ I(yhat - 1) + test_y))$overall
}))
colnames(test_performance) <- c("N40", "N80", "N40_Reg", "N80_Reg")

test_performance
#                   N40    N80 N40_Reg N80_Reg
# Accuracy       0.8652 0.8684  0.8868  0.9014
# Kappa          0.8502 0.8537  0.8742  0.8904
# AccuracyLower  0.8554 0.8587  0.8777  0.8928
# AccuracyUpper  0.8746 0.8777  0.8955  0.9095
# AccuracyNull   0.1074 0.1074  0.1074  0.1074
# AccuracyPValue 0.0000 0.0000  0.0000  0.0000
# McnemarPValue     NaN    NaN     NaN     NaN

# The testing data highlights the fact that, in the non-regularized model, 
# the additional neurons do not meaningfully improve the performance 
# of the model on the testing data.

# However, here we see the advantage of the regularized models 
# for both the 40- and the 80-neuron models. 



# stop cluster
stopCluster(cl)
