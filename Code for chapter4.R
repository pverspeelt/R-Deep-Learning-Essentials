# Auto-encoders

# more info on deeplearning wiht h2o: https://github.com/h2oai/h2o-tutorials/tree/master/tutorials/deeplearning

# load mnist data 
# digits_train <- readr::read_csv("Data/train.csv")
# digits_train$label <- factor(digits_train$label, levels = 0:9)

library(h2o)
local_h2o <- h2o.init(max_mem_size = "12G", nthreads = 3)

# load mnist data directly via h2o. h2o.importFile imports in parallel
h2o_digits <- h2o.importFile("Data/train.csv", destination_frame = "h2odigits")
h2o_digits$label <- as.factor(h2o_digits$label)

i <- 1:20000
h2o_digits_train <- h2o_digits[i, -1]

itest <- 20001:30000
h2o_digits_test <- h2o_digits[itest, -1]
xnames <- colnames(h2o_digits_train)

# H2O uses a parallelization approach known as Hogwild!, 
# that parallelizes stochastic gradient descent optimization, 
# how the weights for the model are optimized/determined
# see for more information on Hogwild!:
# https://www.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf

# Because of the way that Hogwild! works, 
# it is not possible to make the results exactly replicable

# start with a single layer
# 50 hidden neurons
# 20 training iterations => epochs
# no regularization

# if dropout is specified, use activation "WithDropout"
# This is different from book


starttime <- Sys.time()
m1 <- h2o.deeplearning(
  x = xnames,
  training_frame= h2o_digits_train,
  validation_frame = h2o_digits_test,
  model_id = "m1",
  activation = "TanhWithDropout",
  autoencoder = TRUE,
  hidden = c(50),
  epochs = 20,
  sparsity_beta = 0,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = 0,
  l1 = 0,
  l2 = 0
)
endtime <- Sys.time()
endtime - starttime
#  4.541138 mins


# start with a single layer
# 100 hidden neurons
# 20 training iterations => epochs
# no regularization

starttime <- Sys.time()
m2a <- h2o.deeplearning(
  x = xnames,
  training_frame= h2o_digits_train,
  validation_frame = h2o_digits_test,
  model_id = "m2a",
  activation = "TanhWithDropout",
  autoencoder = TRUE,
  hidden = c(100),
  epochs = 20,
  sparsity_beta = 0,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(0),
  l1 = 0,
  l2 = 0
)
endtime <- Sys.time()
endtime - starttime
#  4.069617 mins

# start with a single layer
# 100 hidden neurons
# 20 training iterations => epochs
# sparsity_beta = 0.5

starttime <- Sys.time()
m2b <- h2o.deeplearning(
  x = xnames,
  training_frame= h2o_digits_train,
  validation_frame = h2o_digits_test,
  model_id = "m2b",
  activation = "TanhWithDropout",
  autoencoder = TRUE,
  hidden = c(100),
  epochs = 20,
  sparsity_beta = 0.5,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(0),
  l1 = 0,
  l2 = 0
)
endtime <- Sys.time()
endtime - starttime
#  4.741505 mins


# start with a single layer
# 50 hidden neurons
# 20 training iterations => epochs
# input_dropout_ratio = 0.2

starttime <- Sys.time()
m2c <- h2o.deeplearning(
  x = xnames,
  training_frame= h2o_digits_train,
  validation_frame = h2o_digits_test,
  model_id = "m2c",
  activation = "TanhWithDropout",
  autoencoder = TRUE,
  hidden = c(100),
  epochs = 20,
  sparsity_beta = 0,
  input_dropout_ratio = 0.2,
  hidden_dropout_ratios = c(0),
  l1 = 0,
  l2 = 0
)
endtime <- Sys.time()
endtime - starttime
#  2.042921 mins

m1
m2a
m2b
m2c

h2o.shutdown(prompt = FALSE)
