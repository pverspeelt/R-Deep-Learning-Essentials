# picking hyperparameters

library(ggplot2)
library(h2o)
local_h2o <- h2o.init(max_mem_size = "12G", nthreads = 3)

# load mnist data directly via h2o. h2o.importFile imports in parallel ----
h2o_digits <- h2o.importFile("Data/train.csv", destination_frame = "h2odigits")
h2o_digits$label <- as.factor(h2o_digits$label)

i <- 1:32000
h2o_digits_train <- h2o_digits[i, ]

itest <- 32001:42000
h2o_digits_test <- h2o_digits[itest, ]
xnames <- colnames(h2o_digits_train)[-1]

# run models ----
starttime <- Sys.time()
ex1 <- h2o.deeplearning(
  x = xnames,
  y = "label",
  training_frame = h2o_digits_train,
  validation_frame = h2o_digits_test,
  model_id = "ex1",
  activation = "RectifierWithDropout",
  hidden = c(100),
  epochs = 10,
  adaptive_rate = FALSE,
  rate = 0.001,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(0.2)
)
endtime <- Sys.time()
endtime - starttime
#  27.90629 secs

starttime <- Sys.time()
ex2 <- h2o.deeplearning(
  x = xnames,
  y = "label",
  training_frame = h2o_digits_train,
  validation_frame = h2o_digits_test,
  model_id = "ex2",
  activation = "RectifierWithDropout",
  hidden = c(100),
  epochs = 10,
  adaptive_rate = FALSE,
  rate = 0.01,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(0.2)
)
endtime <- Sys.time()
endtime - starttime
# 18.69458 secs

# conclusions picking hyperparameters ------
# difference ex1 takes longer to train than ex2
ex1
ex2

# other differenct ex1 performs better on validation set.

## Training and predicting new data from a deep neural network ------
# load HAR dataset ------
har_train_x <- readr::read_table("Data/UCI HAR Dataset/train/X_train.txt", col_names = FALSE)
har_train_y <- readr::read_table("Data/UCI HAR Dataset/train/y_train.txt", col_names = FALSE)
# unlist readr reads in _y as a list and doesn't work with factor
har_train <- cbind(har_train_x, outcome = factor(unlist(har_train_y)))

har_test_x <- readr::read_table("Data/UCI HAR Dataset/test/X_test.txt", col_names = FALSE)
har_test_y <- readr::read_table("Data/UCI HAR Dataset/test/y_test.txt", col_names = FALSE)
# unlist readr reads in _y as a list and doesn't work with factor
har_test <- cbind(har_test_x, outcome = factor(unlist(har_test_y)))

har_labels <- readr::read_table("Data/UCI HAR Dataset/activity_labels.txt", col_names = FALSE)

h2oactivity_train <- as.h2o(
  har_train,
  destination_frame = "h2oactivitytrain")

h2oactivity_test <- as.h2o(
  har_test,
  destination_frame = "h2oactivitytest")

# build model ------

# activation function is a linear rectifier with dropout
# input dropout ratio = 20%
# hidden dropout ratio = 50%
# 50 hidden neurons
# 10 training iterations

# 561 columns * 50 neurons = 28050 weigths between input and layer 1
# 50 * 6 = 300 weight between layer 1 and output
starttime <- Sys.time()
mt1 <- h2o.deeplearning(
  x = colnames(har_train_x),
  y = "outcome",
  model_id = "mt1",
  training_frame= h2oactivity_train,
  activation = "RectifierWithDropout",
  hidden = c(50),
  epochs = 10,
  loss = "CrossEntropy",
  input_dropout_ratio = .2,
  hidden_dropout_ratios = c(.5), 
  export_weights_and_biases = TRUE
)
endtime <- Sys.time()
endtime - starttime
# 5.266154 secs

# look at the features of the model using the h2o.deepfeatures() function
# model, data, and layer

features1 <- as.data.frame(h2o.deepfeatures(mt1, h2oactivity_train, 1))
features1[1:10, 1:5]


# extract weights from each layer
weights1 <- as.matrix(h2o.weights(mt1, 1))
## plot heatmap of the weights
tmp <- as.data.frame(t(weights1))
tmp$Row <- 1:nrow(tmp)
tmp <- reshape2::melt(tmp, id.vars = c("Row"))


p.heat <- ggplot(tmp, aes(variable, Row, fill = value)) +
  geom_tile() +
  scale_fill_gradientn(colours = c("black", "white", "blue")) +
  theme_classic() +
  theme(axis.text = element_blank()) +
  xlab("Hidden Neuron") +
  ylab("Input Variable") +
  ggtitle("Heatmap of Weights for Layer 1")

print(p.heat)


# Steps for calculating the first layer manually
# input data
d <- as.matrix(har_train[, -562])

# biases for hidden layer 1 neurons
b1 <- as.matrix(h2o.biases(mt1, 1))
b12 <- do.call(rbind, rep(list(t(b1)), nrow(d)))

# construct the features for layer 1
# step 1: standardize each column of input data
d_scaled <- apply(d, 2, scale)

# step 2: multiply scaled data by weights
# step 3: add the bias matrix
d_weighted <- d_scaled %*% t(weights1) + b12

# step 4: adjust for drop out ratio 
d_weighted <- d_weighted * (1 - .5)

# step 5: linear rectifier => take values 0 or higher
d_weighted_rectifier <- apply(d_weighted, 2, pmax, 0)


all.equal(
  as.numeric(features1[, 1]),
  d_weighted_rectifier[, 1],
  check.attributes = FALSE,
  use.names = FALSE,
  tolerance = 1e-04)


# steps for the second layer
weights2 <- as.matrix(h2o.weights(mt1, 2))

b2 <- as.matrix(h2o.biases(mt1, 2))
b22 <- do.call(rbind, rep(list(t(b2)), nrow(d)))

yhat <- d_weighted_rectifier %*% t(weights2) + b22

# steps for softmax function.
yhat <- exp(yhat)
normalizer <- do.call(cbind, rep(list(rowSums(yhat)), ncol(yhat)))

yhat <- yhat / normalizer

# final step: derive predicted classifaction
yhat <- cbind(Outcome = apply(yhat, 1, which.max), yhat)


h2o_yhat <- as.data.frame(h2o.predict(mt1, newdata = h2oactivity_train))

# manual process matches that of H2O exactly.
xtabs(~ yhat[, 1] + h2o_yhat[, 1])


h2o.shutdown(prompt = FALSE)
