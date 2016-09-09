# load data ------
har_train_x <- readr::read_table("Data/UCI HAR Dataset/train/X_train.txt", col_names = FALSE)
har_train_y <- readr::read_table("Data/UCI HAR Dataset/train/y_train.txt", col_names = FALSE)

har_test_x <- readr::read_table("Data/UCI HAR Dataset/test/X_test.txt", col_names = FALSE)
har_test_y <- readr::read_table("Data/UCI HAR Dataset/test/y_test.txt", col_names = FALSE)

har_labels <- readr::read_table("Data/UCI HAR Dataset/activity_labels.txt", col_names = FALSE)


# load libraries and start H2O ------
# Our goal is to identify any anomalous values or values that are aberrant or otherwise unusual.
library(ggplot2)
library(h2o)
local_h2o <- h2o.init(max_mem_size = "12G", nthreads = 3)


h2oactivity_train <- as.h2o(
  har_train_x,
  destination_frame = "h2oactivitytrain")

h2oactivity_test <- as.h2o(
  har_test_x,
  destination_frame = "h2oactivitytest")


# 2 layers
# 2 times 100 hidden neurons
# no regularization

starttime <- Sys.time()
mu1 <- h2o.deeplearning(
  x = colnames(h2oactivity_train),
  training_frame= h2oactivity_train,
  validation_frame = h2oactivity_test,
  model_id = "mu1",
  activation = "TanhWithDropout",
  autoencoder = TRUE,
  hidden = c(100, 100),
  epochs = 30,
  sparsity_beta = 0,
  input_dropout_ratio = 0,
  hidden_dropout_ratios = c(0, 0),
  l1 = 0,
  l2 = 0
)
endtime <- Sys.time()
endtime - starttime
#  2.015389 mins

mu1

erroru1 <- as.data.frame(h2o.anomaly(mu1, h2oactivity_train))

pue1 <- ggplot(erroru1, aes(Reconstruction.MSE)) +
  geom_histogram(binwidth = .001, fill = "grey50") +
  geom_vline(xintercept = quantile(erroru1[[1]], probs = .99), linetype = 2) +
  theme_bw()
print(pue1)
# There are a few cases that are more anomolous than the rest

# One way to try to explore these anomalous cases further is 
# to examine whether any of the activities tend to have more 
# or less anomalous values. We can do this by finding which 
# cases are anomalous, here defined as the top 1% of error 
# rates, and then extracting the activities of those cases 
# and plotting them

i_anomolous <- erroru1$Reconstruction.MSE >= quantile(erroru1[[1]], probs = .99)

pu_anomolous <- ggplot(as.data.frame(table(har_labels$X2[har_train_y$X1[i_anomolous]])),
                       aes(Var1, Freq)) +
  geom_bar(stat = "identity") +
  xlab("") + ylab("Frequency") +
  theme_classic() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1))
print(pu_anomolous)


# These deep auto-encoders are also useful in other contexts 
# where identifying anomalies is important, such as with financial data or 
# credit card usage patterns. 
# one could train an auto-encoder model and use it to identify anomalies for further investigation.

h2o.shutdown(prompt = FALSE)
