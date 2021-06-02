PROMPT #1 - DESCRIPTION OF THE DATASET
We have selected the COVID-19 dataset to perform the analysis. The dataset is a combination of multiple studies carried across different countries. Each row of the dataset represents a cohort of patients. While some studies examine a single cohort, many of them examine multiple cohorts of patients. 

SELECTING THE DATA FOR ANALYSIS
The purpose of each study included in the dataset is different and the data being reported varies with every study. The papers which study multiple cohorts have divided their study populations in subgroups. Such studies have included top-level study populations which represent the overall analysis for populations in the subgroups. However, some papers which study multiple cohorts have not included any top-level population. We decided to perform the analysis on the subgroups where available as they give more granular information and drop the top-level populations as they caused repetition in the data being analyzed. We have used the group_by() function in R to group the data based on the ID column and selecting only the subgroups(and overall populations in case of papers studying single cohort) based on the SUB_ID column.

SUMMARY OF MIDTERM SUBMISSION
We had generated a few plots during midterm project submission which help understand the dataset better.
1. We wanted to observe the distribution of countries which were included in the studies. For this we have used a pie chart.
![image](https://user-images.githubusercontent.com/77983776/120548572-1fcf2500-c3b8-11eb-8c6d-1b87dc5b5bcd.png)

It can be observed that most of the studies(78%) were performed on the patients from China. Thus, we further checked which Provinces or states in China were the most included in the studies. We have created a bar plot for this.
![image](https://user-images.githubusercontent.com/77983776/120548605-26f63300-c3b8-11eb-8cc0-abd03e2ad0ad.png)

It can be observed from the above bar plot that most of the studies were carried out in Wuhan followed by Shenzhen.
2. Next, we wanted to study the distribution of the different levels of severities of COVID-19 found in the patients across the studies. We used a bar plot for this.
![image](https://user-images.githubusercontent.com/77983776/120548618-2bbae700-c3b8-11eb-9316-96588beba794.png)

 

The levels of severity mentioned in the dataset for different studies were not consistent. We had to perform classification of these levels before plotting the bar plot. For example, ‘Mild only’, ‘Mild Only’, ‘Mild’ all have the same meaning. Thus, we classified such rows as Mild Only.
Here, ‘All’ represents the studies which examine the patients with all levels of severity i.e., Mild, Severe and Asymptomatic. Similarly, ‘Mild and Severe’ represents the studies examining patients with both levels of severity.
3. We further wanted to check the types of patients i.e., survivors or non-survivors, for which the papers have reported the data. We have plotted a bar plot for this.
![image](https://user-images.githubusercontent.com/77983776/120548636-31183180-c3b8-11eb-9f6c-262111e06298.png)

It can be observed that most of the studies have reported data for the patients who survived as well as those who did not. 
4. Below histogram shows the distribution of number of days spent in the hospital across the studies.
![image](https://user-images.githubusercontent.com/77983776/120548662-36757c00-c3b8-11eb-979a-70bc521223a2.png)
 
As it can be observed, the distribution is nearly normal just a few outliers. This indicates that most of the people in study have spent 0 to 30 days in the hospital and a very few have spent over 50 days.
5. Below histogram shows the distribution of number of days it typically took for viral clearance.
![image](https://user-images.githubusercontent.com/77983776/120548694-3f664d80-c3b8-11eb-9a89-87b8d8d197eb.png)

Looking at the histogram above, we can say that there are approximately 27 studies which have reported the data of patients whose days of viral clearance is between 0 to 15 days. There are fewer studies who have reported data for days of viral clearance is between 15 to 35, and this causes right skew in the distribution.

Below scatter plots were created to check the if there is any impact of comorbidities, severity, age etc. on variables like length of hospitalization, ICU admission rate, etc.
![image](https://user-images.githubusercontent.com/77983776/120548744-4e4d0000-c3b8-11eb-8341-ff65f8cf59a9.png)
                 
             
From the above scatter plots, we can see that there isn’t a significant effect of comorbidities on the length of hospitalization. Similarly, levels of severity, mean age and comorbidities do not seem to have remarkable impact of the ICU admission rate individually (There might be a combined effect of these 3 factors on the ICU admission rate, which we will analyze in this project further.)
We could attribute this to the surveys being in the preliminary stage of the pandemic where there was a shortage of speedy recovery medicines leading to majority cases being hospitalized. Similarly, high number of cases with relatively a smaller number of ICU beds available could be the reason for low admission rate in most of the cases.
PROMPT #2 - STATEMENT OF THE PROPOSED QUESTION
We have proposed below question for our analysis – 
“How likely a patient would be admitted to an ICU considering his/her age, medical history(comorbidities), and severity of COVID-19?”

This question is particularly important because during the first wave of the pandemic countries around the world experienced a shortage of medical supplies and limited access to the healthcare services. Hospitals were facing shortage of ICU beds which led to a lot of patients not getting appropriate treatment on time and caused many deaths. 
By examining the effect of medical preconditions, age, and severity of the disease on the ICU admission rate across the studies, we aim to predict the ICU admission rates of the new cohorts of patients. This would help in better management of the healthcare resources and help in more people getting timely treatment. 
PROMPT #3 – MODEL TO BE USED TO ANSWER PROPOSED QUESTION
We plan to use KNN Regression for this question. We had initially proposed to KNN classification to predict if the ICU admission is required for a certain patient or not. However, from the feedback we received for the proposal we realized that we could not predict if a patient needs to be admitted in an ICU because not all the studies look at outcomes of admission vs no admission. Thus, we decided to modify the question and fit a model to predict how likely an ICU admission would be. This would result in the outcome being numeric and continuous. Therefore, we decided to perform regression instead of classification.
KNN is a locality bound algorithm i.e., it assumes samples within a vicinity will perform similarly. For a given sample it identifies the nearest K neighbors to the sample using a distance metric. Once the neighborhood is identified, for a regression problem an average of the target variable from the neighborhood is assigned as the sample's target value.
We selected KNN model to perform the regression for a number of reasons. 
1.	KNN is a lazy learning algorithm. It stores the training dataset and learns from it only at the time of making predictions on the test data. This means that KNN has no training period, which makes it faster than other algorithms.
2.	Since it requires no training period, we can add new data to the dataset without impacting the accuracy of the model.
3.	Being a memory based algorithm, KNN responds quickly to the changes in the input during real-time use.
Consider below corrplot showing the correlations between different comorbidities, age, and ICU admission.
![image](https://user-images.githubusercontent.com/77983776/120548781-5c028580-c3b8-11eb-92b4-6c873b087828.png)


Severity is a categorical variable and for KNN regression we usually calculate the distance between the observation and it’s neighbors. Thus, we have assigned numeric values to the different levels of severity. Looking at the plot, it can be said that not all the preconditions have a significant correlation with ICU admissions. 
Thus, we decided to go ahead with Any Comorbidity column which we are assuming to be a representative of all the comorbidities.
Predictor variables used in the model – Any Comorbidity, Mean Age, Severity Level
Response Variable – ICU admission
PROMPT #4 – THE RESULTS OF YOUR FITTED MODEL
1.	Checking Outliers
Out of the 3 predictor variables, Severity_Level is a categorical variable. Thus, we checked for outliers in only in Mean Age and Any Comorbidity.
We checked the distribution of the data using qqplot and boxplots to check for the outliers in these two columns:
(a)	Mean Age –
![image](https://user-images.githubusercontent.com/77983776/120548810-66bd1a80-c3b8-11eb-9b48-a755ac021520.png)
  
We can see from the boxplot that there are a couple of outliers in Mean Age column. However, looking at the qqplot we can say that they aren’t digressing much from the line of normality. 
(b)	Any Comorbidity –
![image](https://user-images.githubusercontent.com/77983776/120548829-6f155580-c3b8-11eb-8de0-6540e33803ed.png)
   
Looking at the boxplot, there seems to be no outliers in this column. 

Although there are outliers in the column Mean Age, we did not remove them. After removing the NA and missing values from the data, we have very few rows to fit the model. Removing these outliers will result in further reduction in the number of rows and with such less data to work with, the model might not produce accurate results.

2.	Data Cleaning 
Instead of straightaway removing the rows with missing data or NA values, we performed a few transformations on the required columns. This would ensure having reasonable amount of data for regression.
(a)	Mean Age –
We created a new column named ‘Age’ in the dataset by replicating Mean Age column. Next, we checked for the missing values in this column and checked the value in the Median Age column for that index. If the corresponding value in Median Age is not null, we substituted that in place of missing value in the Mean Age column.
Snippet of the code –
![image](https://user-images.githubusercontent.com/77983776/120548874-7dfc0800-c3b8-11eb-9c73-afbe7935aa10.png)


(b)	Severity –
Severity is a categorical column having unique values as follows :
"Severe/Critical Only", "All", "",  "Mild only", "Mild", "Severe", "Mild Only", "Both", "Severe/critical only", "Asymptomatic only"

Since KNN regression calculates the Euclidean distance between the k nearest neighbors and the observation for which the prediction is to be made, all the predictor variables need to be numeric. Thus, we assigned numeric levels to each value in severity. For this, we first create a new column Severity_levels by replicating Severity column. Then, we categorized similar levels of severity into same groups. For example, "Mild only", "Mild Only", “Mild” are categorized as Mild. Next, we assign following numeric label to each value in Severity.
a.	If the severity falls within the category Asymptomatic, it will be substituted by 0.0.
b.	If the severity falls within the category Mild, it will be substituted by 0.25.
c.	If the severity falls within the category All, it will be substituted by 0.5.
d.	If the severity falls within the category Both, it will be substituted by 0.75.
e.	If the severity falls within the category Severe, it will be substituted by 1.0.
Code Snippet :
![image](https://user-images.githubusercontent.com/77983776/120548855-76d4fa00-c3b8-11eb-936e-124e0bd61aab.png)
 
(c)	Any Comorbidity –
Assuming this column would be a representative of all the comorbidities, instead of considering separate columns for the comorbidities we are using Any Comorbidity as of the predictors along with Mean Age and Severity for regression.

To handle the missing values and NAs in this column, we first created a copy this column named ‘Comorbidity’ in the dataset. We replaced the missing and NA values in this column by finding out the maximum value amongst columns Hypertension, Diabetes, Cardiovascular.Disease..incl..CAD., Chronic.obstructive.lung..COPD., Cancer..Any., Liver.Disease..any., Cerebrovascular.Disease, Chronic.kidney.renal.disease, Other for that index.

Code Snippet:
![image](https://user-images.githubusercontent.com/77983776/120548914-8bb18d80-c3b8-11eb-894c-1e6931a7f5aa.png)
 
Finally, we created a new dataset with the transformed columns and the response variable ICU admission. We dropped the rows from the dataset which had NA or missing values even after the transformations we made.
3.	Selection of the parameter
To select the best value of K, we have performed cross validation using k-folds technique. We are taking value of k-folds as 5 folds and performing the cross validation for a range of k values from 4 to 24. We were taking a bigger range of k values earlier; however, it was slowing down the code execution tremendously. Thus, we narrowed down the range of k-values.

We are first dividing the original dataset into training and test data such that 85% percent of the total data would act as training data and 15% will act as the test data. While performing cross validation, we are partitioning the train data into training and validation/test data. To calculate the number of rows to be included in training data for CV we used below formula,
n_tr = floor(nrow(train_data)*(k_folds_k-1)/(k_folds_k))
We ran the cross validation by looping over the number of folds and running KNN regression using each value of K for each fold. An error matrix was created with the folds as the columns and the different k values as rows. In the end, we calculated the average test error for each row and plotted the 5 estimates along with the average error rate.
![image](https://user-images.githubusercontent.com/77983776/120548968-98ce7c80-c3b8-11eb-9f51-82e80999ca61.png)
 
Next, we plotted the average error rate calculated using CV with the actual training and test error rates.
![image](https://user-images.githubusercontent.com/77983776/120549058-ae43a680-c3b8-11eb-822d-1d1cc316af42.png)

 
The actual test error rate has similar pattern as that of the average error rate from CV. It can be observed that the k-values between ~5 to 10 can be considered as the optimal. 
We further checked for the k-value (between 5 to 10 )having the minimum error rate. We got k=6 as the best k value with an error rate of ~0.17.
4.	Plot Summarizing the Fit of the Model
Unlike SLR and MLR, we were not able to directly plot the model to obtain the four residual plots(Residuals vs Fitted and Normal Q-Q plots, Scale-Location, Residuals vs Leverage) to assess the fit of the model. Thus, we manually created Residuals vs Fitted and Normal Q-Q plots of the model.
Code Snippet:
![image](https://user-images.githubusercontent.com/77983776/120549078-b3085a80-c3b8-11eb-976d-ca82a3884640.png) 
	
Below are the Residuals vs Fitted and Normal Q-Q plots:
![image](https://user-images.githubusercontent.com/77983776/120549100-b8fe3b80-c3b8-11eb-860b-f506a55f5b6e.png)
![image](https://user-images.githubusercontent.com/77983776/120549112-bd2a5900-c3b8-11eb-94e8-aae9ba965636.png)
 
 
Looking at the QQ plot we can see that most of the points align well with line of normality. A few points are deviating from the line however overall fit of the model over the data seems reasonable.
5.	Measure of Significance  of the Result
We tried predicting ICU admission rate for the training dataset using the model we have fit over the data. Below plot shows how well the model predicts the ICU admission rate for the training data.

![image](https://user-images.githubusercontent.com/77983776/120549130-c4516700-c3b8-11eb-9baa-c552bb7cc9b1.png)
 

To assess the accuracy of this result, we have calculated the below 4 statistics –
1)	Mean Square Error (MSE)
2)	Mean Absolute Error (MAE)
3)	Root Mean Square Error (RMSE)
4)	R-squared
Below are the values of these statistics for the predictions made over training data with the model –
 
Looking at the R-Squared value, we can say that approximately 66% of the observed variation can be explained by the model's inputs.
As mentioned earlier, we have partitioned the original dataset into training and test data. Thus, we also performed regression on the test data using our model. Below is the plot showing the original and expected ICU admission for each observation –
![image](https://user-images.githubusercontent.com/77983776/120549145-cadfde80-c3b8-11eb-9bce-2d166dbac912.png)
 
Below are the values of four statistics to assess the accuracy of predictions made over test data with the model –
![image](https://user-images.githubusercontent.com/77983776/120549163-cfa49280-c3b8-11eb-838b-2a90a3d0859a.png)
 
Looking at the R-Squared value, we can say that 89.5% of the observed variation can be explained by the model's inputs.
PROMPT #5 – A PREDICTION MADE WITH THE MODEL (INTERPOLATION)
To predict ICU admission using interpolation, we have selected rows from the dataset with transformation and with NAs such that the values for Age, Severity_Level and Comorbidity are present, but the ICU admission rate is missing.
Code snippet:
![image](https://user-images.githubusercontent.com/77983776/120549169-d3d0b000-c3b8-11eb-81a4-34753b91116d.png)
 
The accuracy of the model can be measured using the 4 statistics mentioned in the prompt #4 point 5. We can see that when the predictions are made on the training data using our model, the R-squared value is 0.657 and the RMSE is 0.273.
So, for new observations within the range of the model(Interpolated observations), the R-squared value should remain the same or increase but it is unlikely to get worse. Similarly, the RMSE value should either remain same or decrease.
This was confirmed when we made the predictions on the test data. The R-squared value was 0.895 and the RMSE was 0.171.
Also, we have performed CV to ensure that the model runs with the best possible K value i.e., k = 6. Thus, the model will give reasonably accurate results on the interpolated data.
Post identifying the right k using the train-test split we trained a final model using the same k but the entire data as training dataset to avoid any data loss.
Below plot shows the values for ICU admission rates predicted by our model for the interpolated data:
![image](https://user-images.githubusercontent.com/77983776/120549183-da5f2780-c3b8-11eb-8848-99ad7e7ba0ae.png)


PROMPT #6 – CAVEATS IN THE MODEL
1. Measurement error:
The purpose of each study in the dataset is different, i.e., each study focuses on a different area of interest. For example, a study conducted in a particular country focuses on the cohort of patients who were on ventilators. Such studies will only report data which is significant to the area of interest and leave the rest of the sections empty. This will result in inconsistent data across the different columns in the dataset and produce many missing values. 
2. Sampling error: 
The dataset is formed by aggregating data from different countries across the world. 78% of the studies included in the dataset are carried out in China only. Also, out of 328 studies conducted in China, more than 120 are conducted in Wuhan focal point of the virus. Thus, the trends and statistics derived from the dataset will be highly biased towards population in China and cannot be generalized for the entire population in the world. In addition to this, the equipment and reporting standards used for reporting of the data may not be consistent across the studies. This leads to introduction of anomalies.
3. Modeling error: 
The accuracy of the KNN model is dependent on the quality of the data. KNN is also very sensitive to the irrelevant features in the data. Thus, if there is any noise or missing values in the dataset, the model does not provide accurate results. In case of COVID dataset, there are inconsistencies in the dataset because of the different reporting standards across studies. Thus, a lot of pre-processing is required before fitting the KNN model on the data. This leaves us with a very few rows for regression and the accuracy in predicting the ICU admission is compromised. Since KNN is a lazy algorithm, it is also computationally expensive.
4. Any other biases/caveats:
As mentioned earlier, the dataset consists of multiple studies and these studies include data from one or more cohorts. The varied purposes of the studies give rise to a lot of inconsistencies in data reporting for the same column as well. For example, in the Severity column, the presence of various terms like Severe/Critical Only reported by different studies meant the same so we had to assign them a common label ‘Severity’ while performing the analysis.
Since the dataset has all the summary statistics from different studies instead of the actual statistics for the general population, it is difficult to estimate any kind of trends like hospitalization rate, ICU admission rate, mortality rate etc. accurately for the general population.
