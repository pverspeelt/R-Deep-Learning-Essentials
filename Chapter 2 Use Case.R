# Chapter 2 Use Case ------

# load data ------
har_train_x <- readr::read_table("Data/UCI HAR Dataset/train/X_train.txt", col_names = FALSE)
har_train_y <- readr::read_table("Data/UCI HAR Dataset/train/y_train.txt", col_names = FALSE)

har_test_x <- readr::read_table("Data/UCI HAR Dataset/test/X_test.txt", col_names = FALSE)
har_test_y <- readr::read_table("Data/UCI HAR Dataset/test/y_test.txt", col_names = FALSE)

har_labels <- readr::read_table("Data/UCI HAR Dataset/activity_labels.txt", col_names = FALSE)


barplot(table(har_train_y))

# run model ------
# choose tuning parameters
tuning <- list(
  size = c(40, 20, 20, 50, 50),
  maxit = c(60, 100, 100, 100, 100),
  shuffle = c(FALSE, FALSE, TRUE, FALSE, FALSE),
  params = list(FALSE, FALSE, FALSE, FALSE, c(0.1, 20, 3)))


# setup cluster using 3 cores
# load packages, export required data and variables
# and register as a backend for use with the doParallel package
library(doParallel)

cl <- makeCluster(3)
clusterEvalQ(cl, {
  library(RSNNS)
})
clusterExport(cl,
              c("tuning", "har_train_x", "har_train_y",
                "har_test_x", "har_test_y")
)
registerDoParallel(cl)

starttime <- Sys.time()
har_models <- foreach(i = 1:5, .combine = 'c') %dopar% {
  if (tuning$params[[i]][1]) {
    set.seed(1234)
    list(Model = mlp(
      as.matrix(har_train_x),
      decodeClassLabels(har_train_y$X1),
      size = tuning$size[[i]],
      learnFunc = "Rprop",
      shufflePatterns = tuning$shuffle[[i]],
      learnFuncParams = tuning$params[[i]],
      maxit = tuning$maxit[[i]]
    ))
  } else {
    set.seed(1234)
    list(Model = mlp(
      as.matrix(har_train_x),
      decodeClassLabels(har_train_y$X1),
      size = tuning$size[[i]],
      learnFunc = "Rprop",
      shufflePatterns = tuning$shuffle[[i]],
      maxit = tuning$maxit[[i]]
    ))
  }
}
endtime <- Sys.time()
endtime - starttime
# ?? mins


# predict model ------
# need to export the model results to each of the workers on our cluster
starttime <- Sys.time()
clusterExport(cl, "har_models")
har_yhat <- foreach(i = 1:5, .combine = 'c') %dopar% {
  list(list(
    Insample = encodeClassLabels(fitted.values(har_models[[i]])),
    Outsample = encodeClassLabels(predict(har_models[[i]],
                                          newdata = as.matrix(har_test_x)))
  ))
}

endtime <- Sys.time()
endtime - starttime
# 1.321595 mins



# calculate performance measusures ------
har_insample <- cbind(Y = har_train_y,
                      do.call(cbind.data.frame, lapply(har_yhat, `[[`, "Insample")))
colnames(har_insample) <- c("Y", paste0("Yhat", 1:5))

performance_insample <- do.call(rbind, lapply(1:5, function(i) {
  f <- substitute(~ Y + x, list(x = as.name(paste0("Yhat", i))))
  har_dat <- har_insample[har_insample[,paste0("Yhat", i)] != 0, ]
  har_dat$Y <- factor(har_dat$Y, levels = 1:6)
  har_dat[, paste0("Yhat", i)] <- factor(har_dat[, paste0("Yhat", i)], levels = 1:6)
  res <- caret::confusionMatrix(xtabs(f, data = har_dat))
  
  cbind(Size = tuning$size[[i]],
        Maxit = tuning$maxit[[i]],
        Shuffle = tuning$shuffle[[i]],
        as.data.frame(t(res$overall[c("AccuracyNull", "Accuracy", "AccuracyLower", "AccuracyUpper")])))
}))

har_outsample <- cbind(Y = har_test_y,
                       do.call(cbind.data.frame, lapply(har_yhat, `[[`, "Outsample")))
colnames(har_outsample) <- c("Y", paste0("Yhat", 1:5))

performance_outsample <- do.call(rbind, lapply(1:5, function(i) {
  f <- substitute(~ Y + x, list(x = as.name(paste0("Yhat", i))))
  har_dat <- har_outsample[har_outsample[,paste0("Yhat", i)] != 0, ]
  har_dat$Y <- factor(har_dat$Y, levels = 1:6)
  har_dat[, paste0("Yhat", i)] <- factor(har_dat[, paste0("Yhat", i)], levels = 1:6)
  res <- caret::confusionMatrix(xtabs(f, data = har_dat))
  
  cbind(Size = tuning$size[[i]],
        Maxit = tuning$maxit[[i]],
        Shuffle = tuning$shuffle[[i]],
        as.data.frame(t(res$overall[c("AccuracyNull", "Accuracy", "AccuracyLower", "AccuracyUpper")])))
}))


stopCluster(cl)


op <- options()
options(digits = 2)

performance_insample[,-4]
performance_outsample[,-4]

options(op) 

# conclusion
# First of all, these results show that we are able to classify 
# the types of activity people are engaged in quite accurately 
# based on the data from their smartphones. It also seems from the
# in-sample data that the more complex models do better. 
# However, examining the out-of-sample performance measures, 
# the reverse is actually true! Thus, not only are the in-sample 
# performance measures biased estimates of the models' 
# actual out-of-sample performance, they do not even provide the 
# best way to rank order model performance to choose the best performing model.


