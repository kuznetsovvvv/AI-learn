# LeaveOrNot - целевая переменная

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

education_mapping = {
    'Bachelors': 0,
    'Masters': 1,
    'PHD': 2
}
city_mapping = {
    'Bangalore': 0,
    'Pune': 1,
    'New Delhi': 2
}
gender_mapping = {
    'Male': 0,
    'Female': 1
}
benched_mapping = {
    'No': 0,
    'Yes': 1
}



dataset = pd.read_csv('Employee.csv')
print(dataset.head())
print(dataset.info())
cleaned_df = dataset.dropna()
print("Очищенный Dataset:\n", cleaned_df)


print(dataset['Education'].value_counts(normalize=True))
print(dataset['JoiningYear'].value_counts(normalize=True))
print(dataset['City'].value_counts(normalize=True))
print(dataset['PaymentTier'].value_counts(normalize=True))
print(dataset['Age'].value_counts(normalize=True))
print(dataset['Gender'].value_counts(normalize=True))
print(dataset['EverBenched'].value_counts(normalize=True))
print(dataset['ExperienceInCurrentDomain'].value_counts(normalize=True))
print(dataset['LeaveOrNot'].value_counts(normalize=True))
print(f"Число увольнений:",dataset['LeaveOrNot'].value_counts())

dataset['Education'] = dataset['Education'].map(education_mapping)
dataset['City'] = dataset['City'].map(city_mapping)
dataset['Gender'] = dataset['Gender'].map(gender_mapping)
dataset['EverBenched'] = dataset['EverBenched'].map(benched_mapping)



numeric = ['JoiningYear', 'PaymentTier', 'Age', 'ExperienceInCurrentDomain','LeaveOrNot']
numeric_=['Education', 'City', 'Gender', 'EverBenched', 'LeaveOrNot']







sns.pairplot(dataset[numeric])
plt.show()
sns.pairplot(dataset[numeric_])
plt.show()



pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(dataset[numeric].corr(method='spearman'))
print(dataset[numeric_].corr(method='spearman'))



fig, axes = plt.subplots(2, 3, figsize=(18, 10))

pd.crosstab(dataset['PaymentTier'], dataset['LeaveOrNot'], normalize='index').plot(kind='bar', stacked=True, ax=axes[0,0])
axes[0,0].set_title('PaymentTier')
pd.crosstab(dataset['Gender'], dataset['LeaveOrNot'], normalize='index').plot(kind='bar', stacked=True, ax=axes[0,1])
axes[0,1].set_title('Gender')
pd.crosstab(dataset['Education'], dataset['LeaveOrNot'], normalize='index').plot(kind='bar', stacked=True, ax=axes[0,2])
axes[0,2].set_title('Education')
pd.crosstab(dataset['JoiningYear'], dataset['LeaveOrNot'], normalize='index').plot(kind='bar', stacked=True, ax=axes[1,0])
axes[1,0].set_title('JoiningYear')
pd.crosstab(dataset['City'], dataset['LeaveOrNot'], normalize='index').plot(kind='bar', stacked=True, ax=axes[1,1])
axes[1,1].set_title('City')
axes[1,2].axis('off')

plt.tight_layout()
plt.show()




fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.boxplot(x='LeaveOrNot', y='City', data=dataset, ax=axes[0,0])
axes[0,0].set_title('City')

sns.boxplot(x='LeaveOrNot', y='Gender', data=dataset, ax=axes[0,1])
axes[0,1].set_title('Gender')

sns.boxplot(x='LeaveOrNot', y='Education', data=dataset, ax=axes[1,0])
axes[1,0].set_title('Education')

sns.boxplot(x='LeaveOrNot', y='JoiningYear', data=dataset, ax=axes[1,1])
axes[1,1].set_title('JoiningYear')

plt.tight_layout()
plt.show()




print("\n\n\n\nЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ")
df_1 = dataset.dropna()
df_3 = pd.get_dummies(df_1, columns=['Education', 'City', 'Gender', 'JoiningYear'])

X = df_3.drop('LeaveOrNot', axis=1)
y = df_1['LeaveOrNot']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

lr_params = {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'lbfgs']}
lr_grid = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), lr_params, cv=StratifiedKFold(5), scoring='f1')
lr_grid.fit(X_train, y_train)

model = lr_grid.best_estimator_
print(f"Лучшие параметры: {lr_grid.best_params_}")
y_pred = model.predict(X_test)
print(f'Precision: {precision_score(y_test, y_pred):.3f}')
print(f'Recall: {recall_score(y_test, y_pred):.3f}')
print(f'F1: {f1_score(y_test, y_pred):.3f}')
print(f'Accuracy :{accuracy_score(y_test, y_pred)}')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))



print("\n\n\n\nСЛУЧАЙНЫЙ ЛЕС")
X = df_3.drop('LeaveOrNot', axis=1)
y = df_1['LeaveOrNot']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=17, stratify=y)
forest = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=17)

forest_params = {
    'max_depth': range(1, 6),
    'max_features': range(4, 10),
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

forest_grid = GridSearchCV(forest, forest_params, cv=5, n_jobs=-1, verbose=True, scoring='f1')
forest_grid.fit(X_train, y_train)
print(f"Лучшие параметры: {forest_grid.best_params_}")
forest_pred = forest_grid.predict(X_test)
print(confusion_matrix(y_test, forest_pred))
print(classification_report(y_test, forest_pred))





print("\n\n\n\nSVM")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto', 0.1, 1],
    'degree': [2, 3]
}

svm_grid = GridSearchCV(SVC(random_state=42), svm_params, cv=StratifiedKFold(5), scoring='f1', n_jobs=-1, verbose=True)
svm_grid.fit(X_train_scaled, y_train)

print(f"Лучшие параметры: {svm_grid.best_params_}")
svc_model = svm_grid.best_estimator_
predictions = svc_model.predict(X_test_scaled)

print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
























def remove_outliers_iqr(df, columns):
    df_clean = df.copy()
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
    return df_clean

dataset = remove_outliers_iqr(dataset, ['Age', 'ExperienceInCurrentDomain', 'JoiningYear'])
print(dataset.info)