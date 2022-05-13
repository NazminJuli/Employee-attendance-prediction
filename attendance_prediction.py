        # Supervised Classification algorithm

import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import confusion_matrix,classification_report
from sklearn import metrics

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import GridSearchCV

df =pd.read_excel(r'F:\PyCharm Codes\pythonProject\retouch\employee_prediction\Train_GetEmployeeData.xlsx')

data = df.copy()
data = data.fillna(0)
data = data.drop(["UserId","UserName"], axis=1)
data["AttendanceStatus"] = np.where(data["AttendanceStatus"] == "Present", 1, 0)

# designation to numeric
def f(x):
  if x['Designation'] == 'User': return 2
  elif x['Designation'] == 'Admin': return 1
  elif x['Designation'] == 'HR':  return 3
  else: return 4


def date_to_weekday(date_value):
    return date_value.weekday()

data['Designation'] = data.apply(f, axis=1)

# def modify_joiningdate():
data['DateOfJoin'] = pd.to_datetime(data['DateOfJoin'], format='%m/%d/%Y')
lists_months =[]
for i in range(data.shape[0]):
    lists_months.append(data['DateOfJoin'][i].month)

data['Month Value'] = lists_months
# print(data['DateOfJoin'][2].month)
data['Attendance_date_only'] = data['AttendanceDate'].dt.date

data['Attendance_date_only'] = data['Attendance_date_only'].apply(date_to_weekday)
# print(data['Attendance_date_only'])

columns_name_reordered = ['Designation', 'Month Value', 'Attendance_date_only','WorkHour', 'AnnualLeave', 'SickLeave', 'AttendanceStatus' ]
# columns_name_reordered = ['Designation', 'Attendance_date_only', 'WorkHour','AttendanceStatus']
# columns_name_reordered = ['Attendance_date_only', 'WorkHour', 'AnnualLeave','AttendanceStatus']
data = data[columns_name_reordered]
# print(data.head(10))
X = data.iloc[:,0:6].values
# X = data.iloc[:,0:3].values
# print(X)
Y = data.iloc[:,-1:].values
# print("......:", data.info())

# Split the datasets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 100)

#Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Model selection
# classifier = SVC(C=10, gamma=0.001, kernel = 'linear', random_state = 1)
classifier = SVC(kernel='linear')
classifier.fit(X_train, Y_train)
print('SVM Classifier Training Accuracy:', classifier.score(X_train, Y_train))
Y_pred = classifier.predict(X_test)

### Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(Y_test, Y_pred))
print("Precision:",metrics.precision_score(Y_test, Y_pred))
print("Recall:",metrics.recall_score(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
# cm = confusion_matrix(Y_test, Y_pred)
# print("confusion matrix:", cm)
#

# defining parameter range
# param_grid = {'C': [0.1, 1, 10, 100, 1000],
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#               'kernel': ['linear']}
#
# grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
#
# # fitting the model for grid search
# grid.fit(X_train, Y_train)
#
# # print best parameter after tuning
# print(grid.best_params_)
# grid_predictions = grid.predict(X_test)
#
# # print classification report
# print(classification_report(Y_test, grid_predictions))


####  testing on prediction
test_data =pd.read_excel(r'F:\PyCharm Codes\pythonProject\retouch\employee_prediction\File Name.xlsx')
test_x = test_data.drop(["UserName", "AttendanceStatus"], axis=1)
print(test_x.info())
test_x = sc.fit_transform(test_x)


test_y_pred = pd.DataFrame(classifier.predict(test_x), columns=['AttendanceStatus'])
submission_df = pd.DataFrame({'Id': test_data['UserName'], 'status': test_y_pred['AttendanceStatus']})
print(submission_df)