
# # Goal of the project :

# The "AyuRythm" company launch there android and iOS mobile base app and want to grow there business. 
# But there is problem how to recomended this app and offer who realy want to use it. 
# So for that company desided to give free trial to each and every customer for 24 hours
# and collect data from the customers. In this senariao, some customer purchase the app and someone not.
# According to this data company want to give special offer to the customer who are not interested to buy without offer
# and grow thre business.
# 
#  This is classification problem
# 

# # Import essential libraries

# In[1]:


import numpy as np # for numeric calculation
import pandas as pd # for data analysis and manupulation
import matplotlib.pyplot as plt # for data visualization
import seaborn as sns # for data visualization
from dateutil import parser # convert time in date time data type


# # Import dataset & explore

# In[2]:


AyuRythm_appData = pd.read_csv("Dataset/FineTech appData/AyuRythm_appData.csv")


# In[3]:


AyuRythm_appData.shape


# In[4]:


AyuRythm_appData.head(6) # show fisrt 6 rows of AyuRythm_appData DataFrame  *****code 1


# In[5]:


AyuRythm_appData.tail(6) # show last 6 rows of AyuRythm_appData DataFrame  *****code 2


# In[6]:


for i in [1,2,3,4,5]:
    print(AyuRythm_appData.loc[i,'screen_list'],'\n')


# In[7]:


AyuRythm_appData.isnull().sum() # take summation of null values


# In[8]:


AyuRythm_appData.info() # brief inforamtion about Dataset


# In[9]:


AyuRythm_appData.describe() # give the distribution of numerical variables  *****code 3


# In[10]:


# Get the unique value of each columns and it's length
features = AyuRythm_appData.columns
for i in features:
    print("""Unique value of {}\n{}\nlen is {} \n........................\n
          """.format(i, AyuRythm_appData[i].unique(), len(AyuRythm_appData[i].unique())))


# In[11]:


AyuRythm_appData.dtypes


# In[12]:


#  hour data convert string to int
AyuRythm_appData['hour'] = AyuRythm_appData.hour.str.slice(1,3).astype(int) 


# In[13]:


# get data type of each columns
AyuRythm_appData.dtypes


# In[14]:


AyuRythm_appData.columns


# In[15]:


# drop object dtype columns
AyuRythm_appData2 = AyuRythm_appData.drop(['user', 'first_open', 'screen_list', 'enrolled_date'], axis = 1)


# In[16]:


AyuRythm_appData2.head(6) # head of numeric dataFrame *****code 4


# # Data Visualization

# ## Heatmap Using Correlation matrix

# In[17]:


# Heatmap
plt.figure(figsize=(16,9)) # heatmap size in ratio 16:9

sns.heatmap(AyuRythm_appData2.corr(), annot = True, cmap ='coolwarm') # show heatmap

plt.title("Heatmap using correlation matrix of AyuRythm_appData2", fontsize = 25) # title of heatmap *****code 5


# ## Pairplot of AyuRythm_appData2

# In[18]:


# Pailplot of AyuRythm_appData2 Dataset

#%matplotlib qt5 # for show graph in seperate window
sns.pairplot(AyuRythm_appData2, hue  = 'enrolled') # *****code 6


# ## Countplot of enrolled

# In[19]:


# Show counterplot of 'enrolled' feature
sns.countplot(AyuRythm_appData.enrolled) # *****code 7


# In[20]:


# value enrolled and not enrolled customers
print("Not enrolled user = ", (AyuRythm_appData.enrolled < 1).sum(), "out of 50000")
print("Enrolled user = ",50000-(AyuRythm_appData.enrolled < 1).sum(),  "out of 50000")


# ## Histogram of each feature of fineTech_appData2

# In[21]:


# plot histogram 

plt.figure(figsize = (16,9)) # figure size in ratio 16:9
features = AyuRythm_appData2.columns # list of columns name
for i,j in enumerate(features): 
    plt.subplot(3,3,i+1) # create subplot for histogram
    plt.title("Histogram of {}".format(j), fontsize = 15) # title of histogram
    
    bins = len(AyuRythm_appData2[j].unique()) # bins for histogram
    plt.hist(AyuRythm_appData2[j], bins = bins, rwidth = 0.8, edgecolor = "y", linewidth = 2, ) # plot histogram
    
plt.subplots_adjust(hspace=0.5) # space between horixontal axes (subplots) *****code 8


# In[22]:


for i,j in enumerate(features):
    print(i,j)


# ## Correlation barplot with 'enrolled' feature

# In[23]:


# show corelation barplot 

sns.set() # set background dark grid
plt.figure(figsize = (14,5))
plt.title("Correlation all features with 'enrolled' ", fontsize = 20)
AyuRythm_appData3 = AyuRythm_appData2.drop(['enrolled'], axis = 1) # drop 'enrolled' feature
ax =sns.barplot(AyuRythm_appData3.columns,AyuRythm_appData3.corrwith(AyuRythm_appData2.enrolled)) # plot barplot 
ax.tick_params(labelsize=15, labelrotation = 20, color ="k") # decorate x & y ticks font *****code 9


# In[24]:


# parsing object data into data time format

AyuRythm_appData['first_open'] =[parser.parse(i) for i in AyuRythm_appData['first_open']]


# In[25]:


AyuRythm_appData['enrolled_date'] =[parser.parse(i) if isinstance(i, str) else i for i in AyuRythm_appData['enrolled_date']]


# In[26]:


AyuRythm_appData.dtypes


# In[27]:


AyuRythm_appData['time_to_enrolled']  = (AyuRythm_appData.enrolled_date - AyuRythm_appData.first_open).astype('timedelta64[h]')


# In[28]:


# plot histogram
plt.hist(AyuRythm_appData['time_to_enrolled'].dropna()) # *****code 10


# In[29]:


# Plot histogram
plt.hist(AyuRythm_appData['time_to_enrolled'].dropna(), range = (0,100)) # *****code 11


# In[30]:


# Those customers have enrolled after 48 hours set as 0
AyuRythm_appData.loc[AyuRythm_appData.time_to_enrolled > 48, 'enrolled'] = 0


# In[31]:


AyuRythm_appData


# In[32]:


AyuRythm_appData.drop(columns = ['time_to_enrolled', 'enrolled_date', 'first_open'], inplace=True)


# In[33]:


AyuRythm_appData


# In[34]:


# read csv file and convert it into numpy array
AyuRythm_app_screen_Data = pd.read_csv("Dataset/AyuRythm appData/top_screens.csv").top_screens.values


# In[35]:


AyuRythm_app_screen_Data


# In[36]:


type(AyuRythm_app_screen_Data)


# In[37]:


# Add ',' at the end of each string of  'sreen_list' for further operation.
AyuRythm_appData['screen_list'] = fineTech_appData.screen_list.astype(str) + ','


# In[38]:


AyuRythm_appData


# In[39]:


# string into to number

for screen_name in AyuRythm_app_screen_Data:
    AyuRythm_appData[screen_name] = AyuRythm_appData.screen_list.str.contains(screen_name).astype(int)
    AyuRythm_appData['screen_list'] = AyuRythm_appData.screen_list.str.replace(screen_name+",", "")


# In[40]:


# test
AyuRythm_appData.screen_list.str.contains('Splash').astype(int)


# In[41]:


# test
AyuRythm_appData.screen_list.str.replace('Splash'+",", "")


# In[42]:


# get shape
AyuRythm_appData.shape


# In[43]:


# head of DataFrame
AyuRythm_appData.head(6) # *****code 12


# In[44]:


# remain screen in 'screen_list'
AyuRythm_appData.loc[0,'screen_list']


# In[45]:


AyuRythm_appData.screen_list.str.count(",").head(6)


# In[46]:


# count remain screen list and store counted number in 'remain_screen_list'

AyuRythm_appData['remain_screen_list'] = AyuRythm_appData.screen_list.str.count(",")


# In[47]:


# Drop the 'screen_list'
AyuRythm_appData.drop(columns = ['screen_list'], inplace=True)


# In[48]:


AyuRythm_appData


# In[49]:


# total columns
AyuRythm_appData.columns


# In[50]:


# take sum of all saving screen in one place
saving_screens = ['Saving1',
                  'Saving2',
                  'Saving2Amount',
                  'Saving4',
                  'Saving5',
                  'Saving6',
                  'Saving7',
                  'Saving8',
                  'Saving9',
                  'Saving10',
                 ]
AyuRythm_appData['saving_screens_count'] = AyuRythm_appData[saving_screens].sum(axis = 1)
AyuRythm_appData.drop(columns = saving_screens, inplace = True)


# In[51]:


AyuRythm_appData


# In[52]:


credit_screens = ['Credit1',
                  'Credit2',
                  'Credit3',
                  'Credit3Container',
                  'Credit3Dashboard',
                 ]
AyuRythm_appData['credit_screens_count'] = AyuRythm_appData[credit_screens].sum(axis = 1)
AyuRythm_appData.drop(columns = credit_screens, axis = 1, inplace = True)


# In[53]:


AyuRythm_appData


# In[54]:


cc_screens = ['CC1',
              'CC1Category',
              'CC3',
             ]
AyuRythm_appData['cc_screens_count'] = AyuRythm_appData[cc_screens].sum(axis = 1)
AyuRythm_appData.drop(columns = cc_screens, inplace = True)


# In[55]:


AyuRythm_appData


# In[56]:


loan_screens = ['Loan',
                'Loan2',
                'Loan3',
                'Loan4',
               ]
AyuRythm_appData['loan_screens_count'] = fineTech_appData[loan_screens].sum(axis = 1)
AyuRythm_appData.drop(columns = loan_screens, inplace = True)


# In[57]:


AyuRythm_appData


# In[58]:


AyuRythm_appData.shape


# In[59]:


AyuRythm_appData.info()


# In[60]:


AyuRythm_appData.describe()


# ## Heatmap with correlation matrix of new fineTech_appData

# In[61]:


# Heatmap with correlation matrix of new fineTech_appData

plt.figure(figsize = (25,16)) 
sns.heatmap(AyuRythm_appData.corr(), annot = True, linewidth =2) #*****code 13


# In[62]:


AyuRythm_appData.columns


# In[63]:


AyuRythm_appData['ProfileChildren '].unique()


# In[64]:


corr_matrix = AyuRythm_appData.corr()
corr_matrix['ProfileChildren ']


# In[65]:


AyuRythm_appData['ProfileChildren ']


# # Data Preprocessing

# ## Split dataset in Train and Test

# In[66]:


clean_AyuRythm_appData = AyuRythm_appData
target = AyuRythm_appData['enrolled'] 
AyuRythm_appData.drop(columns = 'enrolled', inplace = True)


# In[67]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(AyuRythm_appData, target, test_size = 0.2, random_state = 0)


# In[68]:


print('Shape of X_train = ', X_train.shape)
print('Shape of X_test = ', X_test.shape)
print('Shape of y_train = ', y_train.shape)
print('Shape of y_test = ', y_test.shape)


# In[69]:


# take User ID in another variable 
train_userID = X_train['user']
X_train.drop(columns= 'user', inplace =True)
test_userID = X_test['user']
X_test.drop(columns= 'user', inplace =True)


# In[70]:


print('Shape of X_train = ', X_train.shape)
print('Shape of X_test = ', X_test.shape)
print('Shape of train_userID = ', train_userID.shape)
print('Shape of test_userID = ', test_userID.shape)


# # Feature Scaling

# In[71]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


# # Model Building

# In[72]:


# impoer requiede packages
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# # Decision Tree

# In[73]:


# Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(criterion= 'entropy', random_state=0)
dt_model.fit(X_train, y_train)
y_pred_dt = dt_model.predict(X_test)

accuracy_score(y_test, y_pred_dt)


# In[91]:


# train with Standert Scaling dataset
dt_model2 = DecisionTreeClassifier(criterion= 'entropy', random_state=0)
dt_model2.fit(X_train_sc, y_train)
y_pred_dt_sc = dt_model2.predict(X_test_sc)

accuracy_score(y_test, y_pred_dt_sc)


# # K-NN

# In[75]:


from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2,)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

accuracy_score(y_test, y_pred_knn)


# In[76]:


# train with Standert Scaling dataset
knn_model2 = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2,)
knn_model2.fit(X_train_sc, y_train)
y_pred_knn_sc = knn_model2.predict(X_test_sc)

accuracy_score(y_test, y_pred_knn_sc)


# # Naive Bayes

# In[77]:


# Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
y_pred_nb = nb_model.predict(X_test)

accuracy_score(y_test, y_pred_nb)


# In[78]:


# train with Standert Scaling dataset
nb_model2 = GaussianNB()
nb_model2.fit(X_train_sc, y_train)
y_pred_nb_sc = nb_model2.predict(X_test_sc)

accuracy_score(y_test, y_pred_nb_sc)


# # Random Forest

# In[79]:


# Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

accuracy_score(y_test, y_pred_rf)


# In[80]:


# train with Standert Scaling dataset
rf_model2 = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)
rf_model2.fit(X_train_sc, y_train)
y_pred_rf_sc = rf_model2.predict(X_test_sc)

accuracy_score(y_test, y_pred_rf_sc)


# # Logistic Regression

# In[82]:


# Logistic Regression
from sklearn.linear_model import LogisticRegression
lr_model = LogisticRegression(random_state = 0, penalty = 'l1')
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

accuracy_score(y_test, y_pred_lr)


# In[83]:


# train with Standert Scaling dataset
lr_model2 = LogisticRegression(random_state = 0, penalty = 'l1')
lr_model2.fit(X_train_sc, y_train)
y_pred_lr_sc = lr_model2.predict(X_test_sc)

accuracy_score(y_test, y_pred_lr_sc)


# # Support Vector Machine

# In[85]:


# Support Vector Machine
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_pred_svc = svc_model.predict(X_test)

accuracy_score(y_test, y_pred_svc)


# In[86]:


# train with Standert Scaling dataset
svc_model2 = SVC()
svc_model2.fit(X_train_sc, y_train)
y_pred_svc_sc = svc_model2.predict(X_test_sc)

accuracy_score(y_test, y_pred_svc_sc)

'''from sklearn.svm import SVC
grid_para = {'C':[1,10,100], 'gamma':[1, 0.01, 0.001], 'kernel':['rbf']} 
from sklearn.model_selection import GridSearchCV
grid_lr = GridSearchCV(SVC(), param_grid = grid_para, refit = True, verbose = 4, n_jobs = -1)
grid_lr.fit(X_train, y_train)
grid_pred_lr = grid_lr.predict(X_test)

cm_grid_lr = confusion_matrix(y_test, grid_pred_lr)
sns.heatmap(cm_grid_lr, annot = True, fmt = 'g')

accuracy_score(y_test, grid_pred_lr)'''
# # XGBoost

# In[87]:


# XGBoost Classifier
from xgboost import XGBClassifier
xgb_model = XGBClassifier()
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

accuracy_score(y_test, y_pred_xgb)


# In[88]:


# train with Standert Scaling dataset
xgb_model2 = XGBClassifier()
xgb_model2.fit(X_train_sc, y_train)
y_pred_xgb_sc = xgb_model2.predict(X_test_sc)

accuracy_score(y_test, y_pred_xgb_sc)


# In[89]:


# XGB classifier with parameter tuning
xgb_model_pt1 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

xgb_model_pt1.fit(X_train, y_train)
y_pred_xgb_pt1 = xgb_model_pt1.predict(X_test)

accuracy_score(y_test, y_pred_xgb_pt1)


# In[90]:


# XGB classifier with parameter tuning
# train with Standert Scaling dataset
xgb_model_pt2 = XGBClassifier(
 learning_rate =0.01,
 n_estimators=5000,
 max_depth=4,
 min_child_weight=6,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.8,
 reg_alpha=0.005,
 objective= 'binary:logistic',
 nthread=4,
 scale_pos_weight=1,
 seed=27)

xgb_model_pt2.fit(X_train_sc, y_train)
y_pred_xgb_sc_pt2 = xgb_model_pt2.predict(X_test_sc)

accuracy_score(y_test, y_pred_xgb_sc_pt2)


# In[102]:


# confussion matrix
cm_xgb_pt2 = confusion_matrix(y_test, y_pred_xgb_sc_pt2)
sns.heatmap(cm_xgb_pt2, annot = True, fmt = 'g')
plt.title("Confussion Matrix", fontsize = 20)  # *****code 14


# In[107]:


# Clasification Report
cr_xgb_pt2 = classification_report(y_test, y_pred_xgb_sc_pt2)

print("Classification report >>> \n", cr_xgb_pt2)


# In[ ]:


# Cross validation
from sklearn.model_selection import cross_val_score
cross_validation = cross_val_score(estimator = xgb_model_pt2, X = X_train_sc, y = y_train, cv = 10)
print("Cross validation of XGBoost model = ",cross_validation)
print("Cross validation of XGBoost model (in mean) = ",cross_validation.mean())


# # Mapping predicted output to the target

# In[108]:


final_result = pd.concat([test_userID, y_test], axis = 1)
final_result['predicted result'] = y_pred_xgb_sc_pt2

final_result


# # Save the Model

# In[118]:


## Pickle
import pickle

# save model
pickle.dump(xgb_model_pt2, open('AyuRythm_ML_model.pickle', 'wb'))

# load model
ml_model_pl = pickle.load(open('AyuRythm_app_ML_model.pickle', 'rb'))

# predict the output
y_pred_pl = ml_model_pl.predict(X_test_sc)

# confusion matrix
cm_pl = confusion_matrix(y_test, y_pred_pl)
print('Confussion matrix = \n', cm_pl)

# show the accuracy
print("Accuracy of model = ",accuracy_score(y_test, y_pred_pl))


# In[119]:


## Joblib
from sklearn.externals import joblib

# save model
joblib.dump(xgb_model_pt2, 'AyuRythm_app_ML_model.joblib')

# load model
ml_model_jl = joblib.load('AyuRythm_app_ML_model.joblib')

# predict the output 
y_pred_jl = ml_model_jl.predict(X_test_sc)

cm_jl = confusion_matrix(y_test, y_pred_jl)
print('Confussion matrix = \n', cm_jl)

print("Accuracy of model = ", accuracy_score(y_test, y_pred_jl))
