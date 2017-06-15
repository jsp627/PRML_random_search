library(ggplot2)
library(tidyr)
library(dplyr)
library(data.table)

wd <- 'D:\\Dropbox\\학업\\2017-1\\패턴 인식\\PRML_project'

setwd(wd)


# model <- 'logistic'
model <- 'nn'

if (model == 'logistic') {
    mnist_random <- read.csv('result/mnist_logistic_random.csv', stringsAsFactors = F)
    mnist_grid <- read.csv('result/mnist_logistic_grid.csv', stringsAsFactors = F)
} else if (model == 'nn') {
    mnist_random <- read.csv('result/mnist_nn_random.csv', stringsAsFactors = F)
    mnist_grid <- read.csv('result/mnist_nn_grid.csv', stringsAsFactors = F)
}



trial_out <- function (result, n) {
    n_data <- dim(result)[1]
    iter <- n_data %/% n
    
    idx <- sample(1:n_data)
    
    out <- NULL
    
    for (i in 1:iter) {
        out <- c(out, max(result$accuracy[idx[((i-1)*n+1):(i*n)]]))
    }
    return(out)
}

if (model == 'logistic') {
    n_experiment <- c(2, 4, 8, 20, 40, 100)
} else if (model == 'nn') {
    n_experiment <- c(4, 8, 16, 32, 64, 120, 240, 480)
}

dt_list <- list()
for (i in 1:length(n_experiment)) {
    dt_list[[2*i-1]] <- data.frame(max_acc = trial_out(mnist_grid, n_experiment[i]), number_experiment = n_experiment[i], search = 'grid', stringsAsFactors = F)
    dt_list[[2*i]] <- data.frame(max_acc = trial_out(mnist_random, n_experiment[i]), number_experiment = n_experiment[i], search = 'random', stringsAsFactors = F)
}

last_dt <- rbindlist(dt_list) %>% 
    mutate(number_experiment = factor(number_experiment))

ggplot(last_dt, aes(x = number_experiment, y = max_acc, fill = search)) + geom_boxplot()
