import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn import metrics

# file_path = "F:\\PyCharm Codes\\pythonProject\\retouch\\employee_prediction\\data\\GetEmployeeData.xlsx"
df =pd.read_excel(r'F:\PyCharm Codes\pythonProject\retouch\employee_prediction\Test_GetEmployeeData.xlsx')
# print(df)
# print(df.head(10))
# data = pd.DataFrame(df)
# print(data)
data = df.copy()
data = data.fillna(0)
data = data.drop(["UserId"], axis=1)
data["AttendanceStatus"] = np.where(data["AttendanceStatus"] == "Present", 1, 0)
#
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
print(data['DateOfJoin'][1].month)
data['Attendance_date_only'] = data['AttendanceDate'].dt.date

data['Attendance_date_only'] = data['Attendance_date_only'].apply(date_to_weekday)
print(data['Attendance_date_only'])

columns_name_reordered = ['UserName','Designation', 'Month Value', 'Attendance_date_only','WorkHour', 'AnnualLeave', 'SickLeave', 'AttendanceStatus' ]
# columns_name_reordered = ['UserName','Designation', 'Attendance_date_only', 'WorkHour','AttendanceStatus']
# columns_name_reordered = ['UserName','Attendance_date_only', 'WorkHour','AnnualLeave', 'AttendanceStatus']
data = data[columns_name_reordered]
print(data)

data.to_excel(r'F:\PyCharm Codes\pythonProject\retouch\employee_prediction\File Name.xlsx', index = False)