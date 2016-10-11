# Dealing with missing data

## setup iris data with some missing
library(ggplot2)
library(gridExtra)
library(dplyr)
d <- iris %>% mutate(Petal.Width = ifelse(Species == "setosa", NA, Petal.Width),
                  Petal.Length = ifelse(Species == "setosa", NA, Petal.Length))


# start h2o cluster
library(h2o)
local_h2o <- h2o.init(max_mem_size = "12G", nthreads = 3)

h2o_dmiss <- as.h2o(d, destination_frame="iris_missing")
h2o_dmeanimp <- as.h2o(d, destination_frame="iris_missing_imp")

## mean imputation
missing_cols <- colnames(h2o_dmiss)[apply(d, 2, anyNA)]


#mistake in book or change in workings of h2o?


for (v in missing_cols) {
  h2o_dmeanimp[, v][is.na(h2o_dmeanimp[, v])] <- h2o.impute(h2o_dmeanimp[v], v)
  # h2o.dmeanimp <- h2o.impute(h2o.dmeanimp, column = v) # Code from book
}


## random forest imputation
d_imputed <- d

## prediction model
for (v in missing_cols) {
  tmp_m <- h2o.randomForest(
    x = setdiff(colnames(h2o_dmiss), v),
    y = v,
    training_frame = h2o_dmiss)
  yhat <- as.data.frame(h2o.predict(tmp_m, newdata = h2o_dmiss))
  d_imputed[[v]] <- ifelse(is.na(d_imputed[[v]]), yhat$predict, d_imputed[[v]])
}

# compare the different methods
grid.arrange(
  ggplot(iris, aes(Petal.Length, Petal.Width,
                   color = Species, shape = Species)) +
    geom_point() +
    theme_classic() +
    ggtitle("Original Data"),
  ggplot(as.data.frame(h2o_dmeanimp), aes(Petal.Length, Petal.Width,
                                          color = Species, shape = Species)) +
    geom_point() +
    theme_classic() +
    ggtitle("Mean Imputed Data"),
  ggplot(d_imputed, aes(Petal.Length, Petal.Width,
                        color = Species, shape = Species)) +
    geom_point() +
    theme_classic() +
    ggtitle("Random Forest Imputed Data"),
  ncol = 1)

# In this case, the mean imputation creates aberrant values quite removed from reality
# If needed, more advanced prediction models could be generated. 
# In statistical inferences, multiple imputation is preferred over single imputation 
# (regardless of the method) as the latter fails to account for uncertainty; 
# that is, when imputing the missing values there is some degree of uncertainty 
# as to exactly what those values are. However, in most use cases for deep learning, 
# the datasets are far too large and the computational time too demanding to create 
# multiple datasets with different imputed values, train models on each, and pool the results; 
# thus, these simpler methods (such as mean imputation or using some other prediction model) are common.



# Solutions for models with low accuracy
# For more information on tuning hyperparameters, see Bengio, Y. (2012), 
# particularly Section 3, Hyper-Parameters, which discusses the selection 
# and characteristics of various hyperparameters.

# Grid searching is excellent when there are only a few values for a few parameters.
# An alternative approach is searching through random sampling.

# To do random sampling: 
# take a seed and then randomly sample a number of hyperparameters, 
# store the sampled parameters, run the model, and return the results.

# when using dropout for regularization, it is common to have a relatively 
# smaller amount of dropout for the input variables (around 20% commonly) 
# and a higher amount for hidden neurons (around 50% commonly).


par(mfrow = c(2, 1))
plot(
  seq(0, .5, by = .001),
  dbeta(seq(0, .5, by = .001), 1, 12),
  type = "l", xlab = "x", ylab = "Density",
  main = "Density of a beta(1, 12)")

plot(
  seq(0, 1, by = .001)/2,
  dbeta(seq(0, 1, by = .001), 1.5, 1),
  type = "l", xlab = "x", ylab = "Density",
  main = "Density of a beta(1.5, 1) / 2")


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


# If rho = 1, then the current gradient is not used and it is completely based
# on the previous gradients. If rho = 0, the previous gradients are 
# not used and it is completely based on the current gradient. 
# Typically, values between .9 and .999 are used.

# The epsilon parameter is a small constant that is added when taking 
# the root mean square of previous squared gradients to improve conditioning 
# (it is ideal to avoid this becoming actually zero) 
# and is typically a very small number

# H2O uses ADADELTA (Zeiler, M. D. (2012)) algorithm to automatically tune the learning rate

# sample the depth or number of layers from 1 to 5 
# sample the number of neurons in each layer from 20 to 600
# by default each will have an equal probability.

run <- function(seed, name = paste0("m_", seed), run = TRUE) {
  set.seed(seed)
  
  p <- list(
    name = name,
    seed = seed,
    depth = sample(1:5, 1),
    l1 = runif(1, 0, .01),
    l2 = runif(1, 0, .01),
    input_dropout = rbeta(1, 1, 12),
    rho = runif(1, .9, .999),
    epsilon = runif(1, 1e-10, 1e-4))
  
  p$neurons <- sample(20:600, p$depth, TRUE)
  p$hidden_dropout <- rbeta(p$depth, 1.5, 1)/2
  
  if (run) {
    model <- h2o.deeplearning(
      x = colnames(har_train_x),
      y = "outcome",
      training_frame = h2oactivity_train,
      activation = "RectifierWithDropout",
      hidden = p$neurons,
      epochs = 100,
      loss = "CrossEntropy",
      input_dropout_ratio = p$input_dropout,
      hidden_dropout_ratios = p$hidden_dropout,
      l1 = p$l1,
      l2 = p$l2,
      rho = p$rho,
      epsilon = p$epsilon,
      export_weights_and_biases = TRUE,
      model_id = p$Name
    )
    
    ## performance on training data
    p$MSE <- h2o.mse(model)
    p$R2 <- h2o.r2(model)
    p$Logloss <- h2o.logloss(model)
    p$CM <- h2o.confusionMatrix(model)
    
    ## performance on testing data
    perf <- h2o.performance(model, h2oactivity_test)
    p$T.MSE <- h2o.mse(perf)
    p$T.R2 <- h2o.r2(perf)
    p$T.Logloss <- h2o.logloss(perf)
    p$T.CM <- h2o.confusionMatrix(perf)
    
  } else {
    model <- NULL
  }
  
  return(list(
    Params = p,
    Model = model))
}

# to make the parameters reproducible
use_seeds <- c(403L, 10L, 329737957L, -753102721L, 1148078598L, -1945176688L,
               -1395587021L, -1662228527L, 367521152L, 217718878L, 1370247081L,
               571790939L, -2065569174L, 1584125708L, 1987682639L, 818264581L,
               1748945084L, 264331666L, 1408989837L, 2010310855L, 1080941998L,
               1107560456L, -1697965045L, 1540094185L, 1807685560L, 2015326310L,
               -1685044991L, 1348376467L, -1013192638L, -757809164L, 1815878135L,
               -1183855123L, -91578748L, -1942404950L, -846262763L, -497569105L,
               -1489909578L, 1992656608L, -778110429L, -313088703L, -758818768L,
               -696909234L, 673359545L, 1084007115L, -1140731014L, -877493636L,
               -1319881025L, 3030933L, -154241108L, -1831664254L)

starttime <- Sys.time()
model_results <- lapply(use_seeds, run)
endtime <- Sys.time()
endtime - starttime
# 1.542883 hours

model_results_dat <- do.call(rbind, 
                             lapply(model_results, 
                                    function(x) with(x$Params,
                                                     data.frame(l1 = l1, 
                                                                l2 = l2,
                                                                depth = depth, 
                                                                input_dropout = input_dropout,
                                                                SumNeurons = sum(neurons),
                                                                MeanHiddenDropout = mean(hidden_dropout),
                                                                rho = rho, 
                                                                epsilon = epsilon, 
                                                                MSE = T.MSE))))


plot_performance <- ggplot(data.table::melt(model_results_dat, id.vars = c("MSE")), aes(value, MSE)) +
  geom_point() +
  stat_smooth(color = "black") +
  facet_wrap(~ variable, scales = "free_x", ncol = 2) +
  theme_classic()

plot_performance


# In addition to viewing the univariate relations between parameters and
# the model error, it can be helpful to use a multivariate model to 
# simultaneously take different parameters into account.

# To fit this (and allow some non-linearity), you can use a generalized 
# additive model, using the gam() function from the mgcv package. 
# We specifically hypothesize an interaction between the model depth 
# and total number of hidden neurons, which we capture by including both
# of those terms in a tensor expansion using the te() function, 
# with the remaining terms given univariate smooths, using the s() function. 
library(mgcv)
m_gam <- gam(MSE ~ s(l1, k = 4) +
               s(l2, k = 4) +
               s(input_dropout) +
               s(rho, k = 4) +
               s(epsilon, k = 4) +
               s(MeanHiddenDropout, k = 4) +
               te(depth, SumNeurons, k = 4),
             data = model_results_dat)

summary(m_gam)

par(mfrow = c(3, 2))
for (i in 1:6) {
  plot(m_gam, select = i)
}

par(mfrow = c(1, 1))
plot(m_gam, select = 7)
# In general, it seems that, the more layers there are, 
# the more neurons are required to achieve a comparable performance:


# run optimized model ------
model_optimized <- h2o.deeplearning(
  x = colnames(har_train_x),
  y = "outcome",
  training_frame = h2oactivity_train,
  activation = "RectifierWithDropout",
  hidden = c(300, 300, 300),
  epochs = 100,
  loss = "CrossEntropy",
  input_dropout_ratio = .08,
  hidden_dropout_ratios = c(.50, .50, .50),
  l1 = .002,
  l2 = 0,
  rho = .95,
  epsilon = 1e-10,
  export_weights_and_biases = TRUE,
  model_id = "optimized_model"
)


h2o.performance(model_optimized, h2oactivity_test)
# H2OMultinomialMetrics: deeplearning
# 
# Test Set Metrics: 
#   =====================
#   
#   MSE: (Extract with `h2o.mse`) 0.04552533
# RMSE: (Extract with `h2o.rmse`) 0.2133667
# Logloss: (Extract with `h2o.logloss`) 0.1566842
# Mean Per-Class Error: 0.06015901
# Confusion Matrix: Extract with `h2o.confusionMatrix(<model>, <data>)`)
# =========================================================================
#   Confusion Matrix: vertical: actual; across: predicted
# 1   2   3   4   5   6  Error          Rate
# 1      482   1  13   0   0   0 0.0282 =    14 / 496
# 2       12 445  13   0   1   0 0.0552 =    26 / 471
# 3        4  22 394   0   0   0 0.0619 =    26 / 420
# 4        0   2   0 445  43   1 0.0937 =    46 / 491
# 5        0   0   0  51 481   0 0.0959 =    51 / 532
# 6        0   0   0   0  14 523 0.0261 =    14 / 537
# Totals 498 470 420 496 539 524 0.0601 = 177 / 2.947
# 
# Hit Ratio Table: Extract with `h2o.hit_ratio_table(<model>, <data>)`
# =======================================================================
#   Top-6 Hit Ratios: 
#   k hit_ratio
# 1 1  0.939939
# 2 2  0.994571
# 3 3  0.999661
# 4 4  0.999661
# 5 5  1.000000
# 6 6  1.000000


model_results_dat[which.min(model_results_dat$MSE), ]
# l1           l2 depth input_dropout SumNeurons MeanHiddenDropout       rho      epsilon        MSE
# 18 0.002379311 0.0001135312     5  9.989629e-05       2186         0.3903965 0.9648398 2.993998e-06 0.06072076

# it is also possible to optimize hyperparameters more formally, 
# such as using Bayesian optimazation of hyperparameters


# stop h2o cluster
h2o.shutdown(prompt = FALSE)



