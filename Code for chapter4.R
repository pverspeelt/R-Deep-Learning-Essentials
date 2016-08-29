# Auto-encoders

# load mnist data 
digits_train <- readr::read_csv("Data/train.csv")
digits_train$label <- factor(digits_train$label, levels = 0:9)

library(h2o)
local_h2o <- h2o.init(max_mem_size = "12G", nthreads = 4)


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

