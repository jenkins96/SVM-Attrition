
# Libraries ---------------------------------------------------------------
listOfPackages <-c("tidyverse","caret", "e1071",
"magrittr")

for (i in listOfPackages){
  if(! i %in% installed.packages()){
    install.packages(i, dependencies = TRUE)
  }
  lapply(i, require, character.only = TRUE)
  rm(i)
}


# Importing ---------------------------------------------------------------
data <- read_csv("dataset/WA_Fn-UseC_-HR-Employee-Attrition.csv")
attrition <- data
attrition %<>%
  select("Attrition", "MonthlyIncome", "OverTime", "JobLevel", "Age")

# One-Hot Encoding -------------------------------------------------------
dummy <- dummyVars( Attrition ~., data = attrition)
attrition_SVG <- data.frame(predict(dummy, newdata = attrition))
attrition_SVG %<>%
  select(-"OverTimeNo") %>%
  rename("OverTime" = "OverTimeYes")
attrition_SVG %<>%
  cbind(Attrition = attrition$Attrition)
attrition_SVG$Attrition <- factor(attrition_SVG$Attrition, ordered = TRUE, 
                              levels = c("No", "Yes"))


# SVG Model ---------------------------------------------------------------
  set.seed(1)
  training.ids <- createDataPartition(attrition_SVG$Attrition, p = 0.75, list = F)
  mod <- svm(Attrition ~ ., data = attrition_SVG[training.ids, ])
  summary(mod)
  ## Prediction
  pred <- predict(mod, attrition_SVG[-training.ids, ])
  my_table <- table(attrition_SVG[-training.ids, "Attrition"], pred, dnn = c("Actual", "Predicted"))
  
  ## Evaluation
  TP <- my_table[2,2]
  TN <- my_table[1,1]
  FP <- my_table[1,2]
  FN <- my_table[2,1]
  # ACCURACY 
  # = TP + TN / TP+FP+FN+TN
  accuracy <- sum(TP, TN)/sum(TP,FP,FN,TN)
  sprintf("Accuracy: %f", accuracy)
  
  # SENSITIVITY / RECALL]
  # = TP/ TP + FN 
  recall<- TP/(sum(TP, FN)) 
  sprintf("Recall: %f", recall[1])
  
  # SPECIFICITY
  # = TN / TN + FP 
  specificity <- TN/(sum(TN,FP))
  sprintf("Specificity: %f", specificity[1])
  
  # PRECISION
  # = TP / TP + FP 
  precision <-  TP/ sum(TP,FP)
  sprintf("Precision: %f", precision)
  
  #F1 Score = 2*(Recall * Precision) / (Recall + Precision)
  F1 <- 2 *(recall * precision)/(recall + precision)
  sprintf("F1 Score: %f", F1)


# Optimizing The Model ----------------------------------------------------
  
  tuned <- tune.svm(Attrition ~ ., data = attrition_SVG[training.ids, ], 
                    gamma = 10^(-6: -1), cost = 10^(1:3))
  summary(tuned)
  
  mod_fit <- svm(Attrition ~., data = attrition_SVG[training.ids, ], 
                 cost = 1000, gamma = 0.1, 
                 class.weights = c("No" = 0.3, "Yes" = 0.7),
                 kernel="radial")
  summary(mod_fit)
  ## Prediction
  pred_fit <- predict(mod_fit, attrition_SVG[-training.ids, ])
  
  my_table2 <- table(attrition_SVG[-training.ids, "Attrition"], pred_fit, 
        dnn = c("Actual", "Predicted"))
  
  ## Evaluation
  TP_2 <- my_table2[2,2]
  TN_2 <- my_table2[1,1]
  FP_2 <- my_table2[1,2]
  FN_2 <- my_table2[2,1]
  # ACCURACY 
  # = TP + TN / TP+FP+FN+TN
  accuracy_2 <- sum(TP_2, TN_2)/sum(TP_2,FP_2,FN_2,TN_2)
  sprintf("Accuracy: %f", accuracy_2)
  
  # SENSITIVITY / RECALL]
  # = TP/ TP + FN 
  recall_2<- TP_2/(sum(TP_2, FN_2)) 
  sprintf("Recall: %f", recall_2[1])
  
  # SPECIFICITY
  # = TN / TN + FP 
  specificity_2 <- TN_2/(sum(TN_2,FP_2))
  sprintf("Specificity: %f", specificity_2[1])
  
  # PRECISION
  # = TP / TP + FP 
  precision_2 <-  TP_2/ sum(TP_2,FP_2)
  sprintf("Precision: %f", precision_2)
  
  #F1 Score = 2*(Recall * Precision) / (Recall + Precision)
  F1_2 <- 2 *(recall_2 * precision_2)/(recall_2 + precision_2)
  sprintf("F1 Score: %f", F1_2)