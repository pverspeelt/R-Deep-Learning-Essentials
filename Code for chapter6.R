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

# TODO

# stop h2o cluster
h2o.shutdown(prompt = FALSE)


h2o_dmeanimp[, v][is.na(h2o_dmeanimp[, v])]
