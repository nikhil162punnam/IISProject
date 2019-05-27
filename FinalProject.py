#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import neighbors, metrics
from xgboost import XGBClassifier as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from sklearn.ensemble import VotingClassifier
import pickle
from xgboost import plot_importance


# In[2]:


# Reading the dataframe
feat_df = pd.read_csv('distance_features_15.csv')
# Removing the id and label from the df
wlabel_df = feat_df.drop(['id','emotion'], axis = 1)


# In[3]:


# Applying PCA on the data
pca = PCA(n_components = 7)
graph_table = pca.fit_transform(wlabel_df)
pca.explained_variance_ratio_


# In[4]:


# Plot function of two features from the PCA
def plot_2data(table):
    fig, ax = plt.subplots()
    colors = {0:'r',1:'b',2:'c',3:'g',4:'y',5:'k'}
    colors_emotions = {0:'SURPRISE', 1:'ANGER',2:'HAPPY',3:'SADNESS',4:'DISGUST',5:'FEAR'}
    for i in range(len(table)):
        ax.scatter(table[i:,1], table[i:,4], color = colors[feat_df['emotion'][i]], label = '')
    # plt.legend(colors_emotions)
    plt.title('PCA of features where no.of components =7')
    plt.rcParams["figure.figsize"] = (10,7)


# In[5]:


# Plotting the scatter plot
plot_2data(graph_table)


# In[6]:


# Splitting the data into train and test data
X_train, X_test, y_train, y_test = train_test_split(wlabel_df, feat_df['emotion'], test_size = 0.2, random_state = 0, stratify = feat_df['emotion'])


# In[7]:


# Scaling the data
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)


# In[126]:


# Parameters to search for in KNN model
parameters_knn = {
    'weights': ['uniform', 'distance'],
    'leaf_size':[1,2,3,4,5],
    'algorithm':['auto', 'ball_tree','kd_tree','brute'],
    'n_jobs':[-1],
    'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
}


# In[67]:


# Initialize the GridSearch for KNN
model_KNNCV_1 = GridSearchCV(KNeighborsClassifier(n_jobs = -1), parameters_knn, cv=10, n_jobs =-1, verbose = True)
# Fitting the data to the parameters
model_KNNCV_1.fit(X_train, y_train)
# Printing the best score found
print(model_KNNCV_1.score(X_train, y_train))
# Printing the parameters for whom the best score was found
print(model_KNNCV_1.best_params_)


# In[127]:


# Initializing the model with found parameters
model_knn = KNeighborsClassifier(weights='uniform',
                                 leaf_size=1,
                                 n_neighbors =12,
                                 algorithm = 'auto',
                                 n_jobs = -1)
# Fitting the model to our data
model_knn.fit(X_train, y_train)
# Accuracy Score
accuracy_score(model_knn.predict(X_test), y_test)


# In[128]:


# Parameters for SVM
parameters_svm = {
    'kernel':('linear', 'rbf','poly', 'sigmoid'),
    'C':(1,0.25,0.5,0.75),
    'gamma': (1,2,3,'auto'),
    'decision_function_shape':('ovo','ovr'),
    'shrinking':(True,False)
}


# In[75]:


model_SVMCV_1 = GridSearchCV(SVC(), parameters_svm, cv=10, n_jobs =-1,verbose = True)
model_SVMCV_1.fit(X_train, y_train)
print(model_SVMCV_1.score(X_train, y_train))
print(model_SVMCV_1.best_params_)


# In[8]:


# Initializing a model with found parameters
model_svm_1 = SVC(C= 1,
                  decision_function_shape= 'ovo',
                  gamma= 'auto',
                  kernel= 'sigmoid',
                  shrinking= True,
                 probability = True)
model_svm_1.fit(X_train, y_train)
accuracy_score(model_svm_1.predict(X_test), y_test)


# In[130]:


# Parameters for Random Forest
parameters_rf = {
   'criterion':['gini','entropy'],
   'n_estimators':[10,20,30,40,50,60,70,80,90,100],
   'min_samples_leaf':[1,2,3],
   'min_samples_split':[3,4,5,6,7],
   'random_state':[123],
   'n_jobs':[-1],
   'max_features':[None, 'auto','sqrt','log2'],
   'bootstrap':[True, False],
    'class_weight':['balanced_subsample','balanced']
}


# In[79]:


model_RFCV_1 = GridSearchCV(RandomForestClassifier(n_jobs = -1), parameters_rf, cv=5, n_jobs =-1, verbose = True)

model_RFCV_1.fit(X_train, y_train)
print(model_RFCV_1.score(X_train, y_train))
print(model_RFCV_1.best_params_)


# In[131]:


# Initializing the model with found parameters
model_rf = RandomForestClassifier(criterion= 'entropy',
                                  n_estimators=40,
                                  min_samples_leaf=1,
                                  min_samples_split=5,
                                  random_state=123,
                                  n_jobs=-1,
                                  max_features='auto',
                                  bootstrap=True,
                                  class_weight='balanced_subsample')
model_rf.fit(X_train, y_train)
accuracy_score(model_rf.predict(X_test), y_test)


# In[132]:


# Parameters for XGB
parameters_xgb = {
   "eta": [0.05, 0.075, 0.1, 0.15, 0.2],
   "subsample": [0.6, 0.8, 1.0],
   "colsample_bytree": [0.6, 0.8, 1.0],
   "max_depth":[3,4,5,6],
   "gamma" : [0, 1, 2, 3, 4],
   "n_estimators":[250,500],
   "max_delta_step": [0,2,4,6,8],
   "min_child_weight": [0,5,10,15],
   "objective": ['multi:softmax','multi:softprob'],
   "n_jobs":[-1]
   }


# In[60]:


# Searching through the parameters
model_xgbcv_1 = GridSearchCV(xgb(n_jobs=-1), parameters_xgb, cv=2, n_jobs=-1, verbose = True)

model_xgbcv_1.fit(X_train, y_train)
print(model_xgbcv_1.score(X_train, y_train))
print(model_xgbcv_1.best_params_)


# In[9]:


# Initializing and predicting using found parameters
model_xgb = xgb(eta=0.05,
                gamma=0,
                max_delta_step=2,
                max_depth=6,
                min_child_weight=0,
                n_estimators=500,
                objective='multi:softmax',
                subsample=0.6,
                colsample_bytree=1,
                n_jobs = -1)
model_xgb.fit(X_train, y_train)
accuracy_score(model_xgb.predict(X_test), y_test)


# In[134]:


plot_importance(model_xgb)


# In[135]:


# Parameters for MLP
parameters_mlp = {
    'solver':['adam','lbfgs', 'sgd'],
    'learning_rate' : ['adaptive','constant','invscaling'],
    'activation':['tanh','relu','logistic'],
    'early_stopping': [True],
    'max_iter': [1000,1100,1200,1300,1400],
    'alpha': 10.0 ** -np.arange(1, 10),
    'hidden_layer_sizes':np.arange(5, 15),
    'random_state':[0,1,2,3]
}


# In[73]:


# Searching through the parameters
model_MLPCV_1 = GridSearchCV(MLPClassifier(), parameters_mlp, cv=2, n_jobs =-1,verbose = True)

model_MLPCV_1.fit(X_train, y_train)
print(model_MLPCV_1.score(X_train, y_train))
print(model_MLPCV_1.best_params_)


# In[10]:


# Initializing and predicting using found parameters
model_mlp = MLPClassifier(solver='adam',
                          learning_rate = 'adaptive',
                          alpha=1e-5,
                          random_state=1,
                          hidden_layer_sizes=(6,5),
                          activation='tanh',
                          max_iter=500)
model_mlp.fit(X_train, y_train)
accuracy_score(model_mlp.predict(X_test), y_test)


# In[11]:


model_vote = VotingClassifier(estimators=[('xgb', model_xgb), ('mlp', model_mlp), ('svm',model_svm_1)],
                             voting='soft', weights =[1,1,3])


# In[12]:


model_vote.fit(X_train, y_train)
accuracy_score(model_vote.predict(X_test), y_test)


# In[15]:


with open('finalVotingModel.pkl','wb') as fp:
    pickle.dump(model_vote,fp)


# In[16]:


with open('finalScaler.pkl','wb') as fp:
    pickle.dump(scaler, fp)


# In[139]:


print(metrics.classification_report(y_test,model_vote.predict(X_test)))
print(metrics.confusion_matrix(y_test,model_vote.predict(X_test)))


# In[140]:


# Reading the dataframe
feat_df_new = pd.read_csv('final_features_15.csv')
feat_df_new = feat_df_new.dropna().reset_index(drop = True)
# Removing the id and label from the df
wlabel_df_new = feat_df_new.drop(['subject_identifier','emotion'], axis = 1)


# In[141]:


# Splitting the data into train and test data
X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(wlabel_df_new, feat_df_new['emotion'], test_size = 0.2, random_state = 0, stratify = feat_df_new['emotion'])
# Scaling the data
scaler_new = StandardScaler()
scaler_new.fit(X_train_new)
X_train_new = scaler_new.transform(X_train_new)
X_test_new = scaler_new.transform(X_test_new)


# In[142]:


# Fitting the model to our data
model_knn.fit(X_train_new, y_train_new)
# Accuracy Score
print(accuracy_score(model_knn.predict(X_test_new), y_test_new))
# Fitting the model to our data
model_svm_1.fit(X_train_new, y_train_new)
# Accuracy Score
print(accuracy_score(model_svm_1.predict(X_test_new), y_test_new))
# Fitting the model to our data
model_rf.fit(X_train_new, y_train_new)
# Accuracy Score
print(accuracy_score(model_rf.predict(X_test_new), y_test_new))
# Fitting the model to our data
model_xgb.fit(X_train_new, y_train_new)
# Accuracy Score
print(accuracy_score(model_xgb.predict(X_test_new), y_test_new))
# Fitting the model to our data
model_mlp.fit(X_train_new, y_train_new)
# Accuracy Score
print(accuracy_score(model_mlp.predict(X_test_new), y_test_new))


# In[144]:


fig = plt.figure()
ax = fig.add_subplot(111)
#ax.set_ylim(0,10)
plt.plot(xAxis,without_dist)
for i,j in zip(xAxis,without_dist):
    ax.annotate(str(int(j)),xy=(i,j))
plt.plot(xAxis,with_dist)
for i,j in zip(xAxis,with_dist):
    ax.annotate(str(int(j)),xy=(i,j))
plt.legend(['Coordinates','Distances'])
plt.title('Accuracy graph of all 5 methods over coordinates and distances')
plt.show()


# In[ ]:




