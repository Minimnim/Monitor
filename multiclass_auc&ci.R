install.packages('multiROC')
library('multiROC')
library('readxl')
data <- read_excel("multiclass_prob.xlsx", sheet = 1)
df <- c(data[1:3], data[10:12])
df <-  as.data.frame(do.call(cbind, df))
roc_test <- multi_roc(df, force_diag = TRUE)
roc_test$AUC
roc_ci_res <- roc_ci(df, conf= 0.95, type='basic', R = 100, index = 4)
unlist(roc_ci_res)

