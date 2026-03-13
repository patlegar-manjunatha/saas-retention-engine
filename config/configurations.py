numerical_columns = ['TotalCharges', "MonthlyCharges", "tenure"]

categorical_columns = [
'SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines',
'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
'StreamingTV','StreamingMovies','PaperlessBilling']

encoding_columns = ["gender","InternetService","Contract","PaymentMethod"]

target = "Churn"