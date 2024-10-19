
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import seaborn as sns

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

#Support Vector Machine
from sklearn.svm import SVC

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

#Performance Evaluation
from sklearn.metrics import precision_score, accuracy_score, f1_score, confusion_matrix

#Stacked Model
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

#Step 1
df = pd.read_csv('Project_1_Data.csv')
# print(df.info())

#Step 2
my_splitter = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state = 775)
for train_index, test_index in my_splitter.split(df, df["Step"]):
    strat_df_train = df.loc[train_index].reset_index(drop=True)
    strat_df_test = df.loc[test_index].reset_index(drop=True)
    
X_train = strat_df_train.drop("Step", axis = 1)
y_train = strat_df_train["Step"]

X_test = strat_df_test.drop("Step", axis = 1)
y_test = strat_df_test["Step"]

# #Histogram
# fig, axes = plt.subplots(2, 2, figsize=(10, 8))  

# #Stepvalues
# axes[0, 0].hist(y_train, bins=13, color='blue')
# axes[0, 0].set_title('Step Values')
# axes[0, 0].set_xlabel('Step')


# #X values
# axes[0, 1].hist(X_train['X'], bins=20, color='green')
# axes[0, 1].set_title('X Values')
# axes[0, 1].set_xlabel('X Coordinate')


# #Y values
# axes[1, 0].hist(X_train['Y'], bins=20, color='orange')
# axes[1, 0].set_title('Y Values')
# axes[1, 0].set_xlabel('Y Coordinate')


# #Z values
# axes[1, 1].hist(X_train['Z'], bins=20, color='red')
# axes[1, 1].set_title('Z Values')
# axes[1, 1].set_xlabel('Z Coordinate')


# fig.tight_layout()
# plt.show()

# #3D Scatter plot 
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection='3d')


# scatter = ax.scatter(X_train['X'], X_train['Y'], X_train['Z'], c=y_train, cmap='viridis', s=50)

# ax.set_title('3D Scatter Plot of X, Y, Z Coordinates by each Step')
# ax.set_xlabel('X Coordinate')
# ax.set_ylabel('Y Coordinate')
# ax.set_zlabel('Z Coordinate')

# cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
# cbar.set_label('Step')
# plt.show()

# #Step 3
# corr_matrix = df.corr(method='pearson')
# sns.heatmap(corr_matrix)
# plt.title('Correlation Matrix X, Y, Z, Step')
# plt.show()

# corr_matrix = df.drop('Step', axis=1).corr(method='pearson')
# sns.heatmap(corr_matrix)
# plt.title('Correlation Matrix X, Y, Z)')
# plt.show()

#Step 4

#RandomForestClassifier 
rf_model = RandomForestClassifier(random_state=45)

param_grid_rf = {
    'n_estimators': [150, 250, 350],
    'max_depth': [15, 25, 35],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [3, 4]
}

grid_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='accuracy')
grid_rf.fit(X_train, y_train)

print("Best RFC Parameters:", grid_rf.best_params_)
print("Best RFC Accuracy:", grid_rf.best_score_)


#Support Vector Machine
svm_model = SVC()

param_grid_svm = {
    'C': [0.01, 0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'degree': [2, 3, 4],
    'coef0': [0.0, 0.1, 0.5]
}


grid_svm = GridSearchCV(svm_model, param_grid_svm, cv=5, scoring='accuracy')
grid_svm.fit(X_train, y_train)

print("\nBest SVM Parameters:", grid_svm.best_params_)
print("Best SVM Accuracy:", grid_svm.best_score_)


#Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
param_dist_dt = {
    'max_depth': randint(10, 50),          
    'min_samples_split': randint(2, 20),    
    'min_samples_leaf': randint(1, 10),     
    'criterion': ['gini', 'entropy']        
}

random_search_dt = RandomizedSearchCV(dt_model, param_distributions=param_dist_dt, 
                                      n_iter=100, cv=5, random_state=42, scoring='accuracy')
random_search_dt.fit(X_train, y_train)


print("\nDecision Tree Parameters:", random_search_dt.best_params_)
print("Decision Tree Accuracy:", random_search_dt.best_score_)

#Step 5

def evaluate_model_performance(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    return y_pred


print("\nRandom Forest Classifier Performance:")
y_pred_rf = evaluate_model_performance(grid_rf, X_test, y_test)


print("\nSupport Vector Machine Performance:")
y_pred_svm = evaluate_model_performance(grid_svm, X_test, y_test)


print("\nDecision Tree Classifier Performance:")
y_pred_dt = evaluate_model_performance(random_search_dt, X_test, y_test)

conf_matrix = confusion_matrix(y_test, y_pred_rf)

plt.figure()
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap='Blues', cbar=False)
plt.title('Confusion Matrix of Random Forest Classifier')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

#Step 6

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

estimators = [
    ('rf', grid_rf.best_estimator_),
    ('svm', grid_svm.best_estimator_)
]

stacked_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000)
)

stacked_model.fit(X_train_scaled, y_train)

y_pred_stacked = stacked_model.predict(X_test_scaled)

accuracy_stacked = accuracy_score(y_test, y_pred_stacked)
precision_stacked = precision_score(y_test, y_pred_stacked, average='weighted')
f1_stacked = f1_score(y_test, y_pred_stacked, average='weighted')

print(f"\nStacked Model Performance:")
print(f"Accuracy: {accuracy_stacked:.4f}")
print(f"Precision: {precision_stacked:.4f}")
print(f"F1-Score: {f1_stacked:.4f}")

conf_matrix_stacked = confusion_matrix(y_test, y_pred_stacked)
plt.figure()
sns.heatmap(conf_matrix_stacked, annot=True, fmt="d", cmap='Blues', cbar=False)
plt.title('Confusion Matrix Stacked Model of RF & SVM')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


