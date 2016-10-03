# Use case – training a deep neural network for automatic classification

# data from a subset of the Million Song Dataset
# download.file("http://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip", 
# destfile = "YearPredictionMSD.txt.zip")
# unzip("YearPredictionMSD.txt.zip")

library(ggplot2)

d <- readr::read_csv("Data/YearPredictionMSD.txt", col_names = FALSE)


p.hist <- ggplot(d, aes(X1)) +
  geom_histogram(binwidth = 1) +
  theme_classic() +
  xlab("Year of Release") 

p.hist


# One possible concern is that the relatively extreme values may 
# exert an undue influence on the model. We can reduce this 
# by reflecting the distribution and taking the square root.
# We could also exclude a small amount of the more extreme cases, 
# such as by excluding the bottom and top 0.5%

quantile(d$X1, probs = c(.005, .995))

# trim the data
d_train <- subset(d[1:463715, ], X1 >= 1957 & X1 <= 2010)
d_test <-  subset(d[463716:515345, ], X1 >= 1957 & X1 <= 2010)

# start h2o cluster
library(h2o)
local_h2o <- h2o.init(max_mem_size = "12G", nthreads = 3)

h2omsd_train <- as.h2o(
  d_train,
  destination_frame = "h2omsd_train")

h2omsd_test <- as.h2o(
  d_test,
  destination_frame = "h2omsd_test")


# baseline performance levels
summary(m0 <- lm(X1 ~ ., data = d_train))$r.squared
# 0.24

cor(
  d_test$X1,
  predict(m0, newdata = d_test))^2
# 0.23

# shallow network with single layer
# performance scoring occur on the full dataset 
# 0, passed to the score_training_samples and score_validation_samples arguments.

starttime <- Sys.time()
m1 <- h2o.deeplearning(
  x = colnames(d)[-1],
  y = "X1",
  model_id = "m1",
  training_frame= h2omsd_train,
  validation_frame = h2omsd_test,
  activation = "RectifierWithDropout",
  hidden = c(50),
  epochs = 100,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(0),
  score_training_samples = 0,
  score_validation_samples = 0,
  diagnostics = TRUE,
  export_weights_and_biases = TRUE,
  variable_importances = TRUE
)
endtime <- Sys.time()
endtime - starttime
# 2.519929 mins

h2o.r2(m1)
# 0.37


# Try a larger, deep feedforward neural network.
# three layers of hidden neurons 200 200 400
# hidden_dropout_ratios 0.2
starttime <- Sys.time()
m2 <- h2o.deeplearning(
  x = colnames(d)[-1],
  y = "X1",
  model_id = "m2",
  training_frame= h2omsd_train,
  validation_frame = h2omsd_test,
  activation = "RectifierWithDropout",
  hidden = c(200, 200, 400),
  epochs = 100,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(.2, .2, .2),
  score_training_samples = 0,
  score_validation_samples = 0,
  diagnostics = TRUE,
  export_weights_and_biases = TRUE,
  variable_importances = TRUE
)
endtime <- Sys.time()
endtime - starttime
# 49.312 mins
h2o.r2(m2)
# 0.4606324


## Not run. This runs over 10 hrs on a 10 core machine
# 
# m3 <- h2o.deeplearning(
#   x = colnames(d)[-1],
#   y = "V1",
#   training_frame= h2omsd.train,
#   validation_frame = h2omsd.test,
#   activation = "RectifierWithDropout",
#   hidden = c(500, 500, 1000),
#   epochs = 500,
#   input_dropout_ratio = 0,
#   hidden_dropout_ratios = c(.5, .5, .5),
#   score_training_samples = 0,
#   score_validation_samples = 0,
#   diagnostics = TRUE,
#   export_weights_and_biases = TRUE
# )


# As long as the general architecture—the number of hidden neurons, 
# layers, and connections—remains the same, 
# using the checkpoint can be a great time saver.
# This is because the previous training iterations can be re-used, 
# but also because it tends to take longer for earlier than later iterations.

# increase epochs to 200 
starttime <- Sys.time()
m2b <- h2o.deeplearning(
  x = colnames(d)[-1],
  y = "X1",
  model_id = "m2b",
  training_frame= h2omsd_train,
  validation_frame = h2omsd_test,
  activation = "RectifierWithDropout",
  hidden = c(200, 200, 400),
  checkpoint = "m2",
  epochs = 200,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(.2, .2, .2),
  score_training_samples = 0,
  score_validation_samples = 0,
  diagnostics = TRUE,
  export_weights_and_biases = TRUE,
  variable_importances = TRUE
)
endtime <- Sys.time()
endtime - starttime
# 20.14798 secs

h2o.r2(m2b)
# 0.43
# additional epochs did not improve the model performance

# save model
h2o.saveModel(
  object = m2,
  path = "Data\\Models",
  force = TRUE)

# score history shows the performance of the model across iterations
# as well as a time stamp and the duration for each epoch
h2o.scoreHistory(m2)


yhat <- as.data.frame(h2o.predict(m1, h2omsd_train))
yhat <- cbind(as.data.frame(h2omsd_train[["X1"]]), yhat)

p_resid <- ggplot(yhat, aes(factor(X1), predict - X1)) +
  geom_boxplot() +
  geom_hline(yintercept = 0) +
  theme_classic() +
  theme(axis.text.x = element_text(
    angle = 90, vjust = 0.5, hjust = 0)) +
  xlab("Year of Release") +
  ylab("Predicted Year of Release")

print(p_resid)

# the results show a marked pattern of decreasing residuals in later years or, 
# conversely, show extremely aberrant model predictions for the earlier years. 
# In part, this may be due to the distribution of the data. 
# With most cases coming from the mid 1990s to 2000s the model will be 
# most sensitive to accurately predicting these values, 
# and the comparatively fewer cases before 1990 or 1980 will have less influence

imp <- as.data.frame(h2o.varimp(m2))
imp[1:10, ]


p_imp <- ggplot(imp, aes(factor(variable, levels = variable), percentage)) +
  geom_point() +
  theme_classic() +
  theme(axis.text.x = element_blank()) +
  xlab("Variable Number") +
  ylab("Percentage of Total Importance")

print(p_imp)



# From the description of the dataset, the first 12 variables represented 
# various timbres of the music, with the next 78 being the unique elements 
# of a covariance matrix from the first 12


# simple model with only first 12 variables.
starttime <- Sys.time()
mtest <- h2o.deeplearning(
  x = colnames(d)[2:13],
  y = "X1",
  model_id = "mtest",
  training_frame= h2omsd_train,
  validation_frame = h2omsd_test,
  activation = "RectifierWithDropout",
  hidden = c(50),
  epochs = 100,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(0),
  score_training_samples = 0,
  score_validation_samples = 0,
  diagnostics = TRUE,
  export_weights_and_biases = TRUE,
  variable_importances = TRUE
)
endtime <- Sys.time()
endtime - starttime
# 1.621172 mins

h2o.r2(mtest)
# 0.24

# lower R2 versus model m1 and m2.
# many variables have a small importance but they add up.




# stop h2o cluster
h2o.shutdown(prompt = FALSE)