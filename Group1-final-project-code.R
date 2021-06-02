#===============================================================================
#AUTHOR1 : RUCHITA BAGLANE NETID : baglane2@illinois.edu
#AUTHOR2 : AISHWARYA SHAH  NETID: as97@illinois.edu
#GROUP1

#TITLE : FINAL SUBMISSION
#DATASET : COVID-19

#CODE CONTENTS : 
#1. Filtering the dataset to extract only the unique studies.
#2. Converting the values in columns having %.
#3. PERFORMING TRANSFORMATIONS ON THE DATA
#4. Checking outliers in the predictors.
#5. Partitioning the data into training and test dataset.
#6. Performing the corss validation.
#7. Fitting the KNN model on train and test data.
#8. Making predictions with model (interpolation).
#===============================================================================

require(ggplot2)
require(tidyverse)
require(caret)
require(plotmo)

#Reading the data from the dataset
data_ = read.csv("C:\\Users\\ruchi\\OneDrive\\Documents\\Data Stats\\Project\\Covid Dataset-20210220\\covid_analytics_clinical_data.csv")
data_ = data.frame(lapply(data_, as.character))

#===============================================================================
#FILTERING OUT THE SUBGROUPS AND ELIMINATING THE TOP LEVEL STUDY POPULATION
#===============================================================================
fltr = function(x, ...){
  if(nrow(x) == 1){
    return(x)
  }
  x = x[x['SUB_ID'] != 0,]
  return(x)
}

data = data_ %>% 
  group_by(ID) %>%
  group_modify(fltr) %>% ungroup()
data = as.data.frame(data)

#===============================================================================
#Converting the values in columns having %.
#===============================================================================
# convert percents -- count which ones have %'s
# This can take a moment & you might get a few NA warnings
columnsInPercent = c()
for (i in 1:length(colnames(data))){
  # check for "%" in this column
  percentColumns = FALSE
  percentCount = c()
  for (j in 1:nrow(data)){
    if(grepl("%",data[colnames(data)[i]][j,],fixed=TRUE)){
      percentColumns = TRUE
      #if (percentColumns){break} # break if we find one
      # track individual
      percentCount = c(percentCount,TRUE)
    } else {
      percentCount = c(percentCount,FALSE)
    }
  }
  # save answer for later
  columnsInPercent = c(columnsInPercent, percentColumns)
  # if we found some percentages, convert this column
  if (percentColumns){
    dataC = c()
    xx = pull(data[colnames(data)[i]])
    for (j in 1:nrow(data)){
      if (percentCount[j]){
        dataC = c(dataC,as.numeric(sub("%","",xx[j]))/100.)
      } else {
        dataC = c(dataC,as.numeric(xx[j]))
      }
    }
    
    data[colnames(data)[i]] = dataC
    
  } else { 
    # check for and update numerical columns
    xx = pull(data[colnames(data)[i]])
    nNum = !is.na(as.numeric(xx))
    if (length(nNum[nNum]) > 10){
      data[colnames(data)[i]] = as.numeric(xx)
    }
  }
}
# look at which columns are in percentages:
#print(subset(colnames(data), columnsInPercent))

#===============================================================================
#PERFORMING TRANSFORMATIONS ON THE DATA
#1. Performing Transformations for Mean.Age column.
#2. Performing Transformations for Any.Comorbidity column.
#3. Performing Transformations for Severity column.
#4. Eliminating NA values from the resulting dataset.
#===============================================================================

#Eliminating the studies which have studied on Non ICU cases.
ICU_data = data

#Creating new column "Age" with values in Mean.Age
ICU_data$Age = ICU_data$Mean.Age
#Replace missing/NA values with corresponding values in Median.Age
ICU_data$Age[(is.na(ICU_data$Age)|ICU_data$Age == "")] = 
  ICU_data$Median.Age[(is.na(ICU_data$Age)|ICU_data$Age == "")]

#Creating new column "Comorbidity" with max values in all types of comorbidities.
#This transformation is to ensure minimum number of missing/NA values.
ICU_data <- mutate(ICU_data, Comorbidity = 
                   pmax(ICU_data$Any.Comorbidity, ICU_data$Hypertension, 
                        ICU_data$Diabetes, ICU_data$Cardiovascular.Disease..incl..CAD., 
                        ICU_data$Chronic.obstructive.lung..COPD., ICU_data$Cancer..Any., 
                        ICU_data$Liver.Disease..any., ICU_data$Cerebrovascular.Disease,
                        ICU_data$Chronic.kidney.renal.disease, ICU_data$Other, na.rm = TRUE))

#Creating new column "Severity_Level" with values in Severity.
ICU_data$Severity_Level = ICU_data$Severity
#Categorizing the similar levels of severity in one group. 
Severity_mild = c('Mild', 'Mild only', 'Mild Only')
Severity_severe = c('Severe', 'Severe/Critical Only', 'Severe/critical only')

#Renaming the similar levels of severity with appropriate label.
ICU_data$Severity_Level[ICU_data$Severity_Level %in% Severity_mild] = 0.25
ICU_data$Severity_Level[ICU_data$Severity_Level %in% Severity_severe] = 1.0
ICU_data$Severity_Level[ICU_data$Severity_Level == 'Both'] = 0.75
ICU_data$Severity_Level[ICU_data$Severity_Level == 'All'] = 0.5
ICU_data$Severity_Level[ICU_data$Severity_Level == 'Asymptomatic only'] = 0.0

#Changing the datatype of the Severity from string to numeric.
ICU_data$Severity_Level = as.numeric(ICU_data$Severity_Level)

#Creating dataset with required predictor variables and response variable.
ICU_data_Knn = data.frame(cbind(ICU_data$Age, ICU_data$Comorbidity, 
                                ICU_data$Severity_Level, ICU_data$ICU.admission))

#Renaming the columns.
colnames(ICU_data_Knn)<-c("Age", "Comorbidity", "Severity_Level", "ICU.admission")   

#Ommitting rows with NA values.
ICU_data_Knn_woNA = na.omit(ICU_data_Knn)

#===============================================================================
#FITTING THE KNN REGRESSION MODEL ON THE DATA
#1. Function to fit KNN regression model - perform_KNN()
#2. Function to partition the data into train and test data - partition_data()
#3. Function to compute train & test error for CV - get_knn_error_rates()
#4. Function to perform cross validation - perform_CV()
#5. 
#===============================================================================

#-------------------------------------------------------------------------------
#Function to fit KNN regression model on the part of the dataset which is 
#considered to be test data set.
#-------------------------------------------------------------------------------
perform_KNN = function(train_data, k){

  fit = train(form = ICU.admission~., data = train_data, method = 'knn', 
              tuneGrid = data.frame(k = k),
              preProcess = c('center', 'scale'))

  return (fit)
}

#-------------------------------------------------------------------------------
#Function to partition the data into train and test data
#-------------------------------------------------------------------------------
partition_data = function(data_KNN){
  
  train_ind = sample.int(n=nrow(data_KNN), size = nrow(data_KNN)*0.8, 
                         replace = FALSE)
  train_data = data_KNN[train_ind,]
  test_data = data_KNN[-train_ind,]
  
  return (list(train_data, test_data))
}

#-------------------------------------------------------------------------------
#Function to compute train and test error for cross validation.
#-------------------------------------------------------------------------------
get_knn_error_rates = function(train_data, test_data, k){
  # computes KNN test/train error rates
  
  # get predictions on training data
  knn_model <- perform_KNN(train_data, k=k)
  
  # get predictions on train data
  knn_train_prediction <- predict(knn_model, # pass the model
                                  train_data[-ncol(train_data)]) # set testing x
  
  # get predictions on test data
  knn_test_prediction <- predict(knn_model, # pass the model
                                 test_data[-ncol(test_data)]) # set testing x
  
  
  # training error rate
  tr_err <- RMSE(knn_train_prediction, pull(train_data[ncol(train_data)]))
  # training error rate
  tst_err <- RMSE(knn_test_prediction, pull(test_data[ncol(test_data)]))
  
  return (list(tr=tr_err, tst=tst_err))
}


#-------------------------------------------------------------------------------
#Function to perform cross validation to determne best k value.
#-------------------------------------------------------------------------------
perform_CV = function(train_data, test_data){
  #Specifying the value for K required for CV.
  k_folds_k = 5
  k_values = seq(4, 24, by = 2)
  
  error_matrix = matrix(0, nrow=length(k_values), ncol=k_folds_k)
  error_matrix = matrix(0, 
                        nrow=length(k_values), 
                        ncol=k_folds_k) %>% as_tibble() %>% add_column(k=k_values)
  
  colnames(error_matrix) = str_replace(colnames(error_matrix), "V", "fold")
  
  #Computing the number of rows to be included in the new training dataset.
  n_tr = floor(nrow(train_data)*(k_folds_k-1)/(k_folds_k))
  
  #looping over k_folds_k
  for (kf in 1:k_folds_k){
    #splitting the data into training and validation subsets
    tr_indicies = sample(x=1:nrow(train_data), size=n_tr, replace=FALSE)
    #splitting the original sample training data into subset training & validation 
    tr_data = train_data[tr_indicies,] #all of rows from tr_indices
    tst_data = train_data[-tr_indicies,] #everything else
    
    #Calculating the error rates
    for (kn in 1:length(k_values)){
      # calculate the test error from the test/validation set:
      errs = get_knn_error_rates(tr_data, tst_data, k_values[kn])
      error_matrix[kn, kf] = errs["tst"]
    }
  }
  
  #Plotting the 5 estimates to observe the error rates
  plot(k_values, pull(error_matrix[,1]), type='l', ylim=c(0.10, 0.40),
       xlab = "K value", ylab = "Test Error Rate")
  colors = c("black", "blue", "magenta", "red", "orange")
  lnames = c("k fold=1")
  for(i in 2:k_folds_k){
    lines(k_values, pull(error_matrix[,i]), col=colors[i])
    lnames = append(lnames, paste("k fold=", toString(i)))
  }
  legend("bottomright",lnames,col=colors, lwd=2)
  grid()
  
  #Calculating the average error rates for all the folds.
  cv_mean_test = rowMeans(error_matrix[-(k_folds_k+1)]) 
  
  #Plotting average error rate line.
  lines(k_values, cv_mean_test, col='brown', lwd=4)
  
  #compare to "real" test error
  plot(k_values, cv_mean_test, col='brown', lwd=4, type = 'line', 
       ylim=c(0.10, 0.40), xlab = "K Value", ylab = "Test Error Rate")
  
  tr = c() # save training error rate
  tst = c() # save test error rate
  #Calculating actual training and test error rates using different k values.
  for (i in 1:length(k_values)){
    errs = get_knn_error_rates(train_data, test_data, k_values[i])
    tr = append(tr, errs$tr)
    tst = append(tst, errs$tst)
  }
  
  
  #Plotting the training and test error rates with the CV error rate.
  lines(k_values, tst, col="magenta")
  lines(k_values, tr, col="blue")
  
  legend("bottomright", c("Test", "CV", "Training"), 
         col=c("magenta","black","blue"), 
         lwd = 2)
  
  #Returning the k value with the least error.
  return (k_values[which.min(tst)])
}

#-------------------------------------------------------------------------------
#Function to create a plot to summarise the fit of the model to the data.
#-------------------------------------------------------------------------------
eval_regression = function(model, data, title = ' '){
  y = pull(data[ncol(data)])
  pred_y = predict(model, data[-ncol(data)])
  #Calculating the residuals.
  residuals = pred_y - y
  #Plotting the residuals with the predicted values.
  plot(x= pred_y, y= residuals, ylab = 'residuals', 
       xlab= paste('fitted value -', title), 
       title = paste('residual plot', title))
  abline(0, 0)
  lines(lowess(residuals~pred_y), col = 'red')
  
  #Plotting QQ plot 
  qqnorm(residuals, main = paste('Residuals QQ plot -', title))
  qqline(residuals)
  
  #Plotting actual and predicted values.
  plot(x = c(1: length(y)), y = y, col = 'blue', type='l', 
       lwd = 2, xlab = '# observation')

  lines(x = c(1: length(y)), y = pred_y, col = 'red', lwd = 2)

  legend("topright",  legend = c("original-ICU rate", "predicted-ICU rate"), 
         fill = c("blue", "red"), col = 2:3,  adj = c(0, 0.6))
  grid()
  
  #Calculating accuracy metrics 
  cat('MSE: ', round(mean((pred_y - y)**2), 3), '\n')
  cat('RMSE: ', round(RMSE(pred_y, y), 3), '\n')
  cat('MAE: ', round(MAE(pred_y, y), 3), '\n')
  cat('R-Squared: ', round(R2(pred_y, y),3), '\n')
}

#-------------------------------------------------------------------------------
#Function to plot boxplots and qq plots to identify outliers in predictors.
#-------------------------------------------------------------------------------
check_outlier = function(column, name){
   boxplot(column, main = print(paste0("Boxplot for ",name)))
   #hist(column, main = print(paste0("Histogram of ",name)))
   qqnorm(column)
   qqline(column)
}

#===============================================================================
#This section makes all the function calls.
#===============================================================================
#Checking outliers. 
check_outlier(ICU_data_Knn_woNA$Age, 'Age')
check_outlier(ICU_data_Knn_woNA$Comorbidity, 'Comorbidity')

set.seed(21)
#Calling Function to partition the data.
partitions = partition_data(ICU_data_Knn_woNA)
train_data = partitions[[1]]
test_data = partitions[[2]]

#Calling function to perform cross validation to get best k value.
k = perform_CV(train_data, test_data)
model = perform_KNN(train_data, k)

#Calling function to plot the fit on training data and calculate accuracy metrics.
print('TRAIN set metrics:', quote = F)
eval_regression(model, train_data, 'Train')

#Calling function to plot the fit on training data and calculate accuracy metrics.
print('TEST set metrics:', quote = F)
eval_regression(model, test_data, 'Test')


#### Generating final model without train-test split
#### with above configs to fit entire data.
final_model = perform_KNN(ICU_data_Knn_woNA, k)

##Interpolation
ipolatn_data = head(ICU_data_Knn[is.na(ICU_data_Knn$ICU.admission) 
             | ICU_data_Knn$ICU.admission == "", ], 
             floor(nrow(ICU_data_Knn_woNA)*0.25))

pred_ICU_admission = predict(final_model, ipolatn_data[-ncol(ipolatn_data)])

plot(x = c(1: length(pred_ICU_admission)), y = pred_ICU_admission, 
     type = 'n', xlab = '# observation')
lines(x = c(1: length(pred_ICU_admission)), y = pred_ICU_admission, 
      col = 'blue')