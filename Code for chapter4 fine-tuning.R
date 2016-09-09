# Fine-tuning auto-encoder models
# cross-validation is now supported by deep learning in h2o. 
# hyper parameter tuning for deep learning can be found on
# https://github.com/h2oai/h2o-tutorials/tree/master/tutorials/deeplearning


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


## create 5 folds
folds <- caret::createFolds(1:20000, k = 5)

## create parameters to try
hyper_params <- list(
  list(
    hidden = c(50),
    input_dr = c(0),
    hidden_dr = c(0)),
  list(
    hidden = c(200),
    input_dr = c(.2),
    hidden_dr = c(0)),
  list(
    hidden = c(400),
    input_dr = c(.2),
    hidden_dr = c(0)),
  list(
    hidden = c(400),
    input_dr = c(.2),
    hidden_dr = c(.5)),
  list(
    hidden = c(400, 200),
    input_dr = c(.2),
    hidden_dr = c(.25, .25)),
  list(
    hidden = c(400, 200),
    input_dr = c(.2),
    hidden_dr = c(.5, .25)))

# loop through the hyperparameters and 5-fold cross-validation
# training 6 * 5 models

starttime <- Sys.time()
fm <- lapply(hyper_params, function(v) {
  lapply(folds, function(i) {
    h2o.deeplearning(
      x = xnames,
      training_frame= h2o_digits_train[-i, ], # train on not current folds
      validation_frame = h2o_digits_train[i, ], # validate on current fold
      # model_id = "m1",
      activation = "TanhWithDropout",
      autoencoder = TRUE,
      hidden = v$hidden,
      epochs = 30,
      sparsity_beta = 0,
      input_dropout_ratio = v$input_dr,
      hidden_dropout_ratios = v$hidden_dr,
      l1 = 0,
      l2 = 0
    )
  })
})
endtime <- Sys.time()
endtime - starttime
# 2.714138 hours

fm_results <- lapply(fm, function(m) {
  sapply(m, h2o.mse, valid = TRUE)
})

fm_results <- tibble::data_frame(
  Model = rep(paste0("M", 1:6), each = 5),
  MSE = unlist(fm_results))


head(fm_results)

library(ggplot2)
p_error_rate <- ggplot(fm_results, aes(Model, MSE)) +
  geom_boxplot() +
  stat_summary(fun.y = mean, geom = "point", colour = "red") +
  theme_classic()

print(p_error_rate)

library(dplyr)
fm_results %>% group_by(Model) %>% summarise(Mean_MSE = mean(MSE)) %>% arrange(Mean_MSE)


# It appears that the fourth set of hyperparameters provided
# the lowest cross-validated MSE. The fourth set of hyperparameters 
# was a fairly complex model, with 400 hidden neurons, but also had regularization
# with 20% of the input variables dropped and 50% of the hidden neurons dropped 
# at each iteration, and this actually outperforms (albeit only slightly) 
# the third set of hyperparameters where the same model complexity was 
# used but without any dropout on the hidden layer. 
# Although not much worse, the deep models here with a second layer 
# of 200 hidden neurons perform worse than the shallow model.




starttime <- Sys.time()
fm_final <- h2o.deeplearning(
  x = xnames,
  training_frame= h2o_digits_train,
  validation_frame = h2o_digits_test,
  model_id = "fm_final",
  activation = "TanhWithDropout",
  autoencoder = TRUE,
  hidden = hyper_params[[4]]$hidden,
  epochs = 30,
  sparsity_beta = 0,
  input_dropout_ratio = hyper_params[[4]]$input_dr,
  hidden_dropout_ratios = hyper_params[[4]]$hidden_dr,
  l1 = 0,
  l2 = 0
)
endtime <- Sys.time()
endtime - starttime
# 5.753717 mins

fm_final 
