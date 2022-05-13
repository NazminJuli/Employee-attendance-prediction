import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
# import seaborn as sns

df =pd.read_excel(r'F:\PyCharm Codes\pythonProject\retouch\employee_prediction\Train_GetEmployeeData.xlsx')
df = df.fillna(0)
train_x = df.drop(["UserId","UserName"], axis=1)
def f(x):
  if x['Designation'] == 'User': return 2
  elif x['Designation'] == 'Admin': return 1
  elif x['Designation'] == 'HR':  return 3
  else: return 4

def date_to_weekday(date_value):
    return date_value.weekday()

train_x['Designation'] = train_x.apply(f, axis=1)
# def modify_joiningdate():
# df['DateOfJoin'] = df['DateOfJoin'].values.astype('float64')
train_x['DateOfJoin'] = pd.to_datetime(df['DateOfJoin'], format='%m/%d/%Y')
# train_x['DateOfJoin'] = train_x['DateOfJoin'].values.astype('float64')

lists_months =[]
for i in range(train_x.shape[0]):
    lists_months.append(train_x['DateOfJoin'][i].month)

train_x['Month Value'] = lists_months
train_x['DateOfJoin'] = train_x['DateOfJoin'].values.astype('float64')
# print(data['DateOfJoin'][2].month)
train_x['Attendance_date_only'] = df['AttendanceDate'].dt.date

train_x['Attendance_date_only'] = train_x['Attendance_date_only'].apply(date_to_weekday)
train_x['Attendance_date_only'] = train_x['Attendance_date_only'].values.astype('float64')

columns_name_reordered = ['Designation', 'Month Value', 'Attendance_date_only','WorkHour', 'AnnualLeave', 'SickLeave','AttendanceStatus']
train_x = train_x[columns_name_reordered]
X = train_x.iloc[:,0:6].values
sc = StandardScaler()
X_train = sc.fit_transform(X)
# train_x = MinMaxScaler().fit_transform(train_x)
print(X)

train_x["AttendanceStatus"] = np.where(train_x["AttendanceStatus"] == "Present", 1, 0)
# train_y = df["AttendanceStatus"]
Y =train_x.iloc[:,-1:].values

# print("...............",train_y)

# Feature extraction
model = SVC(kernel='linear')
rfe = RFE(model, 3)
fit = rfe.fit(X, Y)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))

# f,ax = plt.subplots(figsize=(18, 18))
# corr_mat = sns.heatmap(train_x.corr(), annot=False, linewidths=.5, fmt= '.1f',ax=ax)
# Create the RFE object and compute a cross-validated score.
# svc = SVR(kernel='linear')
# rfecv = RFECV(estimator=svc, step=5, scoring='accuracy', min_features_to_select=5)
# rfecv.fit(train_x, train_y)
#
# print("Optimal number of features : %d" % rfecv.n_features_)
#
# # Plot number of Features Vs Cross-validation scores
# plot = plt.figure()
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (nb of correct classifications)")
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.show()
#
# # Feature extraction
# test = SelectKBest(score_func=chi2, k=4)
# fit = test.fit(X, train_y)
#
# # Summarize scores
# np.set_printoptions(precision=3)
# print(fit.scores_)
#
# features = fit.transform(X)
# # Summarize selected features
# print(features[0:5,:])