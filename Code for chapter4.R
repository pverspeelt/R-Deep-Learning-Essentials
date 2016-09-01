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

# look at the model metrics
# In model m1, the MSE is fairly low and identical in the training and validation data. 
# This may be in part due to how relatively simple the model is
# (50 hidden neurons and 20 epochs, when there are hundreds of input variables). 
# In model m2a, there is about a 45% reduction in the MSE, although both are low. 
# However, with the greater model complexity, a slight difference between the training
# and validation metrics is observed. 
# Similar results are noted in model m2b. 
# Despite the fact that the validation metrics did not improve with regularization, 
# the training metrics were closer to the validation metrics, 
# suggesting the performance of the regularized training data generalizes better. 
# In model m2c, the 20% input dropout without additional model complexity results 
# in poorer performance in both the training and validation data. 

m1
m2a
m2b
m2c


# Another way we can look at the model results is to calculate how anomalous each case is. 
# This can be done using the h2o.anomaly() function

error1 <- as.data.frame(h2o.anomaly(m1, h2o_digits_train))
error2a <- as.data.frame(h2o.anomaly(m2a, h2o_digits_train))
error2b <- as.data.frame(h2o.anomaly(m2b, h2o_digits_train))
error2c <- as.data.frame(h2o.anomaly(m2c, h2o_digits_train))

error <- tibble::as_tibble(rbind(
  cbind.data.frame(Model = "1", error1),
  cbind.data.frame(Model = "2a", error2a),
  cbind.data.frame(Model = "2b", error2b),
  cbind.data.frame(Model = "2c", error2c)))

library(dplyr)
library(ggplot2)

percentile <- error %>% group_by(Model) %>% mutate(Percentile = quantile(Reconstruction.MSE, probs = .99)) 

# The histograms show the error rates for each case and the dashed line is the 99th percentile. 
# Any value beyond the 99th percentile may be considered fairly extreme or anomalous

p <- ggplot(error, aes(Reconstruction.MSE)) +
  geom_histogram(binwidth = .001, fill = "grey50") +
  geom_vline(aes(xintercept = Percentile), data = percentile, linetype = 2) +
  theme_bw() +
  facet_wrap(~Model)

# Models 2a and 2b have the lowest error rates, and you can see the small tails
print(p)


error.tmp <- cbind(error1, error2a, error2b, error2c)
colnames(error.tmp) <- c("M1", "M2a", "M2b", "M2c")
plot(error.tmp)


# extract the deep features from the model
# The deep features are the values for the hidden neurons in the model
# One way to explore these features is to correlate them and examine the distribution of correlations
features1 <- as.data.frame(h2o.deepfeatures(m1, h2o_digits_train))
cor_features1 <- cor(features1)
cor_features1 <- data.frame(r = cor_features1[upper.tri(cor_features1)])

p_hist <- ggplot(cor_features1, aes(r)) +
  geom_histogram(binwidth = .02) +
  theme_classic()
print(p_hist)


# Given that we know the MNIST dataset consists of 10 different 
# handwritten digits, perhaps we might try adding a second layer 
# of hidden neurons with only 10 neurons, supposing that, 
# when the model learns the features of the data, 
# 10 prominent features may correspond to the 10 digits.

# start with 2 layers
# 100 and 10 hidden neurons
# 30 training iterations => epochs
# make sure hidden dropout ratios reflect the hidden layers

starttime <- Sys.time()
m3 <- h2o.deeplearning(
  x = xnames,
  training_frame= h2o_digits_train,
  validation_frame = h2o_digits_test,
  model_id = "m3",
  activation = "TanhWithDropout",
  autoencoder = TRUE,
  hidden = c(100, 10),
  epochs = 30,
  sparsity_beta = 0,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(0, 0),
  l1 = 0,
  l2 = 0
)
endtime <- Sys.time()
endtime - starttime
# 3.594916 mins


# specify layer 2 of the deepfeatures
features3 <- as.data.frame(h2o.deepfeatures(m3, h2o_digits_train, 2))
head(features3)


features3$label <- as.vector(h2o_digits$label[i])
features3 <- tidyr::gather(features3, feature, value, -label)

ft <- data.table::melt(features3t, id.vars = "label")


p_line <- ggplot(features3, aes(as.numeric(factor(feature)), value,
                                colour = label, linetype = label)) +
  stat_summary(fun.y = mean, geom = "line") +
  scale_x_continuous("Deep Features", breaks = 1:10) +
  theme_classic() +
  theme(legend.position = "bottom", legend.key.width = unit(1, "cm"))

print(p_line)


m3
# With an MSE of about 0.039, the model fits substantially worse 
# than did the shallow model, probably because having only 10 hidden
# neurons for the second layer is too simplistic to capture all the 
# different features of the data needed to reproduce the original inputs


h2o.shutdown(prompt = FALSE)
