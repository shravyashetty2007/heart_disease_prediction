import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv(r'C:\Users\shrav\OneDrive\Desktop\Python VS\DA project\heart_2020_cleaned.csv')
data.head()

sex = data['Sex'].unique()
sex
data['HeartDisease'].value_counts(True)*100

X = data.drop('HeartDisease',axis=1)
y = data['HeartDisease']
X.info()

y_dummy = y.apply(lambda x: 1 if x == 'Yes' else 0)

numerical_pipe = make_pipeline(SimpleImputer(missing_values=np.nan,strategy = 'mean'),
    StandardScaler(), SimpleImputer(missing_values=np.nan,strategy = 'mean')
)

categorical_pipe = make_pipeline(SimpleImputer(missing_values=np.nan,strategy = 'most_frequent'),
    OneHotEncoder(handle_unknown='ignore',drop='if_binary'),
    SimpleImputer(missing_values=np.nan,strategy = 'most_frequent')
)

numerical_columns = [i for i in data.columns if data[i].dtype != 'object' and i not in ['BMI']]
numerical_columns 

categorical_columns = [i for i in data.columns if data[i].dtype == 'object' and i not in ['HeartDisease']]
categorical_columns

full_pipe = ColumnTransformer([('numerical', numerical_pipe, numerical_columns),
    ('categorical', categorical_pipe, categorical_columns)
])

X_train, X_test, y_train, y_test = train_test_split(X,y_dummy)

RF = make_pipeline(full_pipe,RandomForestClassifier())
RF.fit(X_train,y_train)
y_predict = RF.predict(X_test)

print('Random Forest Tree accuracy:',(accuracy_score(y_test,y_predict))*100)

pickle.dump(RF, open('final_model.sav', 'wb'))

RF

data = { "BMI":34.3,
    "Smoking":"Yes",
    "AlcoholDrinking":"No",
    "Stroke":"Yes",
    "PhysicalHealth":30,
    "MentalHealth":0,
    "DiffWalking":"Yes",
    "Sex":"Male",
    "AgeCategory":"60-64",
    "Race":"White",
    "Diabetic":"Yes",
    "PhysicalActivity":"No",
    "GenHealth":"Poor",
    "SleepTime":15,
    "Asthma":"Yes",
    "KidneyDisease":"No",
    "SkinCancer":"No"
       }


features = pd.DataFrame(data, index=[0])

pred = RF.predict_proba(features)
pred

