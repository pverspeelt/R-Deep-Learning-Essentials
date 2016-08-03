

digits_train <- readr::read_csv("Data/train.csv")

dim(digits_train)
head(colnames(digits_train), 4)
tail(colnames(digits_train), 4)
head(digits_train[, 1:4])

## convert to factor
digits_train$label <- factor(digits_train$label, levels = 0:9)

