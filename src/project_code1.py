#Run to import and initialize
# %%
import sklearn
import plot_learning_curve
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn import linear_model
from sklearn import model_selection
from sklearn import ensemble 
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn import tree 
from sklearn.neural_network import MLPClassifier
class Learn:
    #Returns the number of wrong predictions
    def get_error(predictions,y_vals):
        error_count = 0
        for i in range(0,len(predictions)):
            if predictions[i] != y_vals.iloc[i]:
                error_count += 1
        return error_count
        #Hot encodes the data (Turns non binary attributes to binary attributes)
    def get_hot_data(train_file,test_file):
        data_path = Path('../Data')
        train_data = pd.read_csv(data_path/ train_file)
        split = len(train_data)
        test_data = pd.read_csv(data_path/ test_file)
        dum_train = train_data.iloc[:,0:-1]
        dum_test = test_data.iloc[:,1:]
        id_arr = test_data.iloc[:,0]
        comb_data = dum_train.append(dum_test)
        dum_data = pd.get_dummies(comb_data)
        return dum_data.iloc[0:split],dum_data.iloc[split:],train_data.iloc[:,-1],id_arr 
##Data     
train_data,final_test_data,y_data,ids = Learn.get_hot_data('train_final.csv','test_final.csv')
#Preprocessing the data for testing
#%%
scalar = StandardScaler().fit(train_data,y_data)
scaled_training_data = scalar.transform(train_data)
scaled_test_data = scalar.transform(final_test_data)
x_train_arr = []
x_test_arr = []
y_train_arr = []
y_test_arr = []
for i in range(5):
    x_train,x_test,y_train,y_test = model_selection.train_test_split(scaled_training_data,y_data,test_size=.25)
    x_train_arr.append(x_train)
    x_test_arr.append(x_test)
    y_train_arr.append(y_train)
    y_test_arr.append(y_test)
# %%
#svc testing
for i in [.01,.1,1,10]:
    for m in ['linear']:
        error = 0
        print(i)
        for p in range(5):
            print(p)
            model = svm.SVC(C=i,kernel=m)
            model.fit(x_train_arr[p],y_train_arr[p])
            preds = model.predict(x_test_arr[p])
            error +=Learn.get_error(preds,y_test_arr[p])/len(preds)
        print(error/5)
#%%
#svc model
model = svm.SVC(C=.01,probability=True)

#%%
#ensemble adaboost testing
for j in [1,3,5,7,10]:
    error = 0
    print(str(j))
    for p in range(5):
        model = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='entropy',max_depth=5),learning_rate=.1)
        model.fit(x_train_arr[p],y_train_arr[p])
        preds = model.predict(x_test_arr[p])
        error +=Learn.get_error(preds,y_test_arr[p])/len(preds)
    print(error/5)
#%%
#adaboost 
model = ensemble.AdaBoostClassifier(tree.DecisionTreeClassifier(criterion='entropy',max_depth=5),learning_rate=.1)
#%%
#ensemble extratrees 
error = 0
for i in [10,50,100,500]:
    model = ensemble.ExtraTreesClassifier(n_estimators=i)
# %% 
#neural net
model = MLPClassifier(alpha=1,learning_rate='adaptive')
#%%
#getting error from models
error = 0
for p in range(5):
    model.fit(x_train_arr[p],y_train_arr[p])
    preds = model.predict(x_test_arr[p])
    error += Learn.get_error(preds,y_test_arr[p])/len(preds)
print(error/5)
#%%
#submission to Kaggle 
model.fit(scaled_training_data,y_data)
predictions = model.predict_proba(scaled_test_data)[:,1]
frame = pd.DataFrame(pd.Series(ids,name='ID')).join(pd.DataFrame(pd.Series(predictions,name='Prediction')))
frame.to_csv('../Data/prediction_result.csv',index=False)
#%%
#plot learning curve
plot_learning_curve.plot_learning_curve(model,"learning curve",scaled_training_data,y_data)



