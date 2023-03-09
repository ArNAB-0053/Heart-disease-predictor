### Imports
# -------------------------------------------------------- #
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as lr
from sklearn.metrics import accuracy_score
# -------------------------------------------------------- #

#### Data Collection and Processing
heart_data = pd.read_csv('Dataset/heart.csv')

#### Checking how many data have heart desease 
# target = 1 -> Heart Disease
# target = 0 -> Not 
heart_data['target'].value_counts()

#### Spliting target 
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

#### Spiliting Data into Training & Test data
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, stratify=Y, random_state=2)


#### Train the model 
## Logistic Regretion
model = lr()

# training the model using Logistic Regression with traing data
model.fit(x_train, y_train)

#### Model Evaluaton
## Accuracy Score
# Accuracy on training data
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_prediction, y_train)
# print(training_data_accuracy)

# Accuracy on test data
x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_prediction, y_test)
# print(test_data_accuracy)

##### Prediction System
'''
age
sex(1->Male 0->Female)
chest pain type (0-4)
resting blood pressure(94-200)
serum cholestoral in mg/dl(126-564)
fasting blood sugar > 120 mg/dl(1->True 0->False)
resting electrocardiographic results (values 0,1,2)
maximum heart rate achieved(71-202)
exercise induced angina(1->Yes 0->No)
oldpeak = ST depression induced by exercise relative to rest(0-6.2)
the slope of the peak exercise ST segment(0-2)
number of major vessels (0-3) colored by flourosopy (0-4)
thalassemia: 0 = normal; 1 = fixed defect; 2 = reversable defect
'''

# ------------------------------------------------------------------------------------------------------- #
# Input
# ------------------------------------------------------------------------------------------------------- #
print('''

                INPUTS
---------------------------------------
---------------------------------------''')
age = input("Enter your age: ")
SexIn = input("Sex(Male/Female): ")

sex = 0
fps = 0
if(SexIn=='Male' or 'male'):
    sex = 1
elif(SexIn=='Female' or 'female'):
    sex = 0  
# print(sex)

cp = int(input("Chest Pain type(0-4): "))
bp = int(input("Resting Blood Pressure: "))
sc = int(input("Serum Cholestoral(in mg/dl): "))
fbsIn = input("Fasting Blood Sugar(Yes/No): ")

if(fbsIn=='Yes' or 'yes'):
    fbs = 1
elif(fbsIn=='No' or 'no'):
    fbs = 0

rer =  int(input("Resting Electrocardiographic results(0-2): "))
heartRate = int(input("Maximum heart rate: "))
anginaIn = input("Exercise induced angina(Yes/No): ")

if(anginaIn=='Yes' or 'yes'):
    angina = 1
elif(anginaIn=='No' or 'no'):
    angina = 0    

oldpeak = float(input("ST depression induced by exercise relative to rest(0-6.2): "))
slope = int(input("The slope of the peak exercise ST segment(0-2): "))
vessels = int(input("Number of major vessels (0-3) colored by flourosopy (0-4): "))
thal = int(input("Thalassemia(Normal=0, Fixed defect=1, Reversable defect=2): "))
print("---------------------------------------")
print("---------------------------------------\n\n")
# ------------------------------------------------------------------------------------------------------- #
# ------------------------------------------------------------------------------------------------------- #

input_data = (age, sex, cp, bp, sc, fbs, rer, heartRate, angina, oldpeak, slope, vessels, thal)
# Change input_data into array
input_array = np.asarray(input_data)

# Reshape the array that we can predict for only one isinstance
input_reshape = input_array.reshape(1,-1)
prediction = model.predict(input_reshape)
# print(prediction)

if(prediction[0]==0):
    print("\n\n---------------------***----------------------")
    print(" The person does not have any heart disease")
    print("---------------------***----------------------\n")

else:
    print("\n\n------------------***-------------------")
    print("    The person have heart disease")
    print("------------------***-------------------\n")