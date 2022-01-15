import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
import xgboost as xgb
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestRegressor

np.random.seed(42)

import warnings
warnings.filterwarnings("ignore")

train = pd.read_csv('train.csv')
train['First_Customer_id'] = train['customer_id'].astype(str).str[0]
songs = pd.read_csv('songs.csv')

songs['First_song_id'] = songs['platform_id'].astype(str).str[0]
train = train.merge(songs, how = 'outer' , on = 'song_id')
train[train['number_of_comments'].isnull()] = 0

songs_edited = (train.groupby(["song_id"]).mean())
songs_edited = songs_edited.reset_index()
cust_edited = train.groupby(["customer_id"]).mean()
cust_edited = cust_edited.reset_index()

train = train.merge(songs_edited, how = 'outer' , on = 'song_id')

train = train.rename(columns={'score_x': 'score', 'released_year_x': 'released_year', 
                              'number_of_comments_x': 'number_of_comments', 'score_y':'song_avg'})
train = train.drop(['released_year_y','number_of_comments_y'], axis=1)
train = train.merge(cust_edited, how = 'outer' , on = 'customer_id')
train = train.rename(columns={'score_x': 'score', 'released_year_x': 'released_year', 
                              'number_of_comments_x': 'number_of_comments', 'score_y':'cust_avg'})
train = train.drop(['released_year_y','number_of_comments_y'], axis=1)
train = train.rename(columns={'song_id_x': 'song_id'})
train = train.drop(['song_id_y'], axis=1)

train= train[train['customer_id'] != 0]

train.loc[train['First_song_id'].isnull(),['First_song_id']] = 'O'

train.loc[train['released_year'].isnull(),['released_year']] = 1979
train.loc[train['released_year']<0,['released_year']] = 1979

train.loc[train['number_of_comments'].isnull(),['number_of_comments']] = 12903

train.loc[train['song_avg'].isnull(),['song_avg']] = 3.935059


test = pd.read_csv('test.csv')

test['First_Customer_id'] = test['customer_id'].astype(str).str[0]
test = test.merge(songs, how = 'outer' , on = 'song_id')
test[test['number_of_comments'].isnull()]['number_of_comments'] = 0

test = test.merge(songs_edited, how = 'outer' , on = 'song_id')

test= test[test['song_id'] != 0]
test = test.rename(columns={'released_year_x': 'released_year', 
                              'number_of_comments_x': 'number_of_comments', 'score':'song_avg'})
test = test.drop(['released_year_y','number_of_comments_y'], axis=1)
test = test.merge(cust_edited, how = 'outer' , on = 'customer_id')
test = test.rename(columns={'released_year_x': 'released_year', 'song_id_x': 'song_id',
                              'number_of_comments_x': 'number_of_comments', 'score':'cust_avg'})
test = test.drop(['released_year_y','number_of_comments_y', 'song_id_y'], axis=1)
test= test[test['customer_id'] != 0]

test.loc[test['First_song_id'].isnull(),['First_song_id']] = 'O'
test.loc[test['released_year'].isnull(),['released_year']] = 0
test.loc[test['released_year']==0,['released_year']] = 1979
test.loc[test['released_year']<0,['released_year']] = 1979
test.loc[test['number_of_comments'].isnull(),['number_of_comments']] = 12903
test.loc[test['song_avg'].isnull(),['song_avg']] = 3.935059

save_test = test

data = train
data.loc[data['language'].isnull(), 'language'] = 'eng'

save_for_later = pd.read_csv('save_for_later.csv')
song_labels = pd.read_csv('song_labels.csv')
save_for_later['count'] = 1

train, val = train_test_split(data, test_size=0.05, random_state=42, stratify = data['score'])

song_labels = song_labels.drop_duplicates(subset=['label_id', 'platform_id'], keep='last')
song_labels = song_labels.pivot(index='platform_id', columns='label_id', values='count')
save_for_later_user = save_for_later.pivot(index='customer_id', columns='song_id', values='count')
save_for_later_song = save_for_later.pivot(index='song_id', columns='customer_id', values='count')

train_pivot_user = train.pivot(index='customer_id', columns='song_id', values='score')
train_pivot_song = train.pivot(index='song_id', columns='customer_id', values='score')


X  = np.nan_to_num(song_labels, nan = 0)
pca = KernelPCA(n_components=100, random_state = 42, kernel='poly')
X = pca.fit_transform(X)
song_labels = pd.DataFrame(X, index = song_labels.index)

X  = np.nan_to_num(save_for_later_user, nan = 0)
pca = KernelPCA(n_components=19, random_state = 42, kernel='poly')
X = pca.fit_transform(X)
save_for_later_user = pd.DataFrame(X, index = save_for_later_user.index)

X  = np.nan_to_num(save_for_later_song, nan = 0)
pca = KernelPCA(n_components=24, random_state = 42, kernel='poly')
X = pca.fit_transform(X)
save_for_later_song = pd.DataFrame(X, index = save_for_later_song.index)


X  = np.nan_to_num(train_pivot_user, nan = 0)
pca = KernelPCA(n_components=50, random_state = 42, kernel='poly')
X = pca.fit_transform(X)
train_pivot_user = pd.DataFrame(X, index = train_pivot_user.index)

X  = np.nan_to_num(train_pivot_song, nan = 0)
pca = KernelPCA(n_components=71, random_state = 42, kernel='poly')
X = pca.fit_transform(X)
train_pivot_song = pd.DataFrame(X, index = train_pivot_song.index)


def merge_feature(train, df, feat_m, feat_trans, name, func, test = False):
    if func == 1:
        edited = train.groupby([feat_m]).count()
    if func == 2:
        edited = train.groupby([feat_m]).var()
    if func == 3:
        edited = train.groupby([feat_m]).std()
    if func == 4:
        edited = train.groupby([feat_m]).mean()
        
    edited = edited.reset_index()
    df = df.merge(edited[[feat_m, feat_trans]], how = 'left' , on = feat_m)
    if test == False:
        df = df.rename(columns ={feat_trans + '_x': feat_trans, feat_trans+'_y': name})
    else:
        df = df.rename(columns ={feat_trans : name})
    

    return df

train = merge_feature(train, train, 'customer_id', 'song_id','cust_count', 1)
val = merge_feature(train, val, 'customer_id', 'song_id','cust_count', 1)
train = merge_feature(train, train, 'customer_id', 'score','cust_var', 2)
val = merge_feature(train, val, 'customer_id', 'score','cust_var', 2)
train = merge_feature(train, train, 'customer_id', 'score','cust_std', 3)
val = merge_feature(train, val, 'customer_id', 'score','cust_std', 3)


train = merge_feature(train, train, 'song_id', 'customer_id','song_count', 1)
val = merge_feature(train, val, 'song_id', 'customer_id','song_count', 1)
train = merge_feature(train, train,'song_id', 'score','song_var', 2)
val = merge_feature(train, val, 'song_id', 'score','song_var', 2)
train = merge_feature(train, train,'song_id', 'score','song_std', 3)
val = merge_feature(train, val, 'song_id', 'score','song_std', 3)

train=train.fillna(0)
val=val.fillna(0)

Y_train = train.score
X_train = train[['song_avg', 'cust_avg' ,'cust_var','song_var', 'platform_id', 'customer_id', 
                 'number_of_comments', 'released_year', 'song_id']]
X_test = val[['song_avg', 'cust_avg' ,'cust_var','song_var', 'platform_id', 'customer_id', 
                 'number_of_comments', 'released_year', 'song_id']]

X_train = X_train.merge(song_labels, how = 'left' , on = 'platform_id')
X_train = X_train.drop(['platform_id'],axis=1)

X_train = X_train.merge(save_for_later_user, how = 'left' , on = 'customer_id')
X_train = X_train.merge(train_pivot_user, how = 'left' , on = 'customer_id')
X_train = X_train.drop(['customer_id'],axis=1)

X_train = X_train.merge(save_for_later_song, how = 'left' , on = 'song_id')
X_train = X_train.merge(train_pivot_song, how = 'left' , on = 'song_id')
X_train = X_train.drop(['song_id'],axis=1)


X_test = X_test.merge(song_labels, how = 'left' , on = 'platform_id')
X_test = X_test.drop(['platform_id'],axis=1)

X_test = X_test.merge(save_for_later_user, how = 'left' , on = 'customer_id')
X_test = X_test.merge(train_pivot_user, how = 'left' , on = 'customer_id')
X_test = X_test.drop(['customer_id'],axis=1)

X_test = X_test.merge(save_for_later_song, how = 'left' , on = 'song_id')
X_test = X_test.merge(train_pivot_song, how = 'left' , on = 'song_id')
X_test = X_test.drop(['song_id'],axis=1)

X_train.columns = list(range(len(X_train.columns)))
X_test.columns = list(range(len(X_test.columns)))

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

model =  RandomForestRegressor(random_state = 42)
model.fit(X_train, Y_train)

model1 =  xgb.XGBRegressor(learning_rate = 0.1, verbosity = 1, random_state=42)
model1.fit(X_train, Y_train)

val['RF_preds'] = (model.predict(X_test)+ model1.predict(X_test))/2

songs_df=pd.read_csv('songs.csv')
test = val
df_submission = pd.read_csv('dummy_submission.csv')
Y_test = test.score
test = test.drop(['score'], axis=1)

user_ids = train.customer_id.unique()
total_users = len(user_ids)

### TRUNCSVD ###

#Removing duplicates
song_matrix = pd.concat([train,test]).drop_duplicates(subset = ['customer_id','song_id'],keep = 'first')
#Creates a song matrix of #numofusers vs #noofsongs
song_matrix = song_matrix.pivot('customer_id','song_id','score')

song_means = song_matrix.mean()
user_means = song_matrix.mean(axis=1)
#Mean shifting
song_shifted_temp = song_matrix-song_means
song_shifted = song_shifted_temp.fillna(0)
#To get locations where we have ratings
mask = -song_shifted_temp.isnull()

def repeated_matrix_reconstruction(num_pcs,num_iterations):
    global song_shifted
    for i in range(num_iterations):
        SVD = TruncatedSVD(n_components=num_pcs,random_state=42)
        SVD.fit(song_shifted)
        #For the ease of applying masks we work with pandas
        song_represented =  pd.DataFrame(SVD.inverse_transform(SVD.transform(song_shifted)),columns=song_shifted.columns,index=song_shifted.index)
        loss = mean_squared_error(song_represented[mask].fillna(0),song_shifted_temp[mask].fillna(0))
        print('Iteration: {} , Loss: {} '.format(i,loss))
        #To just update the non-zero values of song_reprented values to the true ratings
        
        if i < (num_iterations - 1):
            song_represented[mask] = song_shifted_temp[mask]
        
        song_shifted = song_represented
            
    #Mean shifting it back
    song_mat = song_shifted + song_means
    song_mat = song_mat.clip(lower=1,upper=5)
    return song_mat
print("Starting truncated svd with number of components as 20")
representative_matrix_20 = repeated_matrix_reconstruction(50,30)
print("Done")
print("Starting truncated svd with number of components as 15")
representative_matrix_15 = repeated_matrix_reconstruction(15,10)
print("Done")
#bagging
rating_matrix = (representative_matrix_15+representative_matrix_20)/2

trunc_prediction = np.zeros(len(test))
for i in range(len(test)):
    userid =  test.iloc[i,0]
    songid = test.iloc[i,1]
    trunc_prediction[i] = rating_matrix[rating_matrix.index==userid][songid].values[0]
    
val['RM_preds'] = trunc_prediction


X_train_final = val[['RM_preds','RF_preds']]
Y_train_final = Y_test

final_model =  LinearRegression()
final_model.fit(X_train_final, Y_train_final)

save_val = val

train = data
val = save_test
test = pd.read_csv('test.csv')


train = merge_feature(train, train, 'customer_id', 'song_id','cust_count', 1)
val = merge_feature(train, val, 'customer_id', 'song_id','cust_count', 1)
train = merge_feature(train, train, 'customer_id', 'score','cust_var', 2)
val = merge_feature(train, val, 'customer_id', 'score','cust_var', 2, True)
train = merge_feature(train, train, 'customer_id', 'score','cust_std', 3)
val = merge_feature(train, val, 'customer_id', 'score','cust_std', 3, True)

train = merge_feature(train, train, 'song_id', 'customer_id','song_count', 1)
val = merge_feature(train, val, 'song_id', 'customer_id','song_count', 1)
train = merge_feature(train, train,'song_id', 'score','song_var', 2)
val = merge_feature(train, val, 'song_id', 'score','song_var', 2, True)
train = merge_feature(train, train,'song_id', 'score','song_std', 3)
val = merge_feature(train, val, 'song_id', 'score','song_std', 3, True)

train=train.fillna(0)
val=val.fillna(0)

Y_train = train.score
X_train = train[['song_avg', 'cust_avg' ,'cust_var','song_var', 'platform_id', 'customer_id', 
                 'number_of_comments', 'released_year', 'song_id']]
X_test = val[['song_avg', 'cust_avg' ,'cust_var','song_var', 'platform_id', 'customer_id', 
                 'number_of_comments', 'released_year', 'song_id']]

X_train = X_train.merge(song_labels, how = 'left' , on = 'platform_id')
X_train = X_train.drop(['platform_id'],axis=1)

X_train = X_train.merge(save_for_later_user, how = 'left' , on = 'customer_id')
X_train = X_train.merge(train_pivot_user, how = 'left' , on = 'customer_id')
X_train = X_train.drop(['customer_id'],axis=1)

X_train = X_train.merge(save_for_later_song, how = 'left' , on = 'song_id')
X_train = X_train.merge(train_pivot_song, how = 'left' , on = 'song_id')
X_train = X_train.drop(['song_id'],axis=1)


X_test = X_test.merge(song_labels, how = 'left' , on = 'platform_id')
X_test = X_test.drop(['platform_id'],axis=1)

X_test = X_test.merge(save_for_later_user, how = 'left' , on = 'customer_id')
X_test = X_test.merge(train_pivot_user, how = 'left' , on = 'customer_id')
X_test = X_test.drop(['customer_id'],axis=1)

X_test = X_test.merge(save_for_later_song, how = 'left' , on = 'song_id')
X_test = X_test.merge(train_pivot_song, how = 'left' , on = 'song_id')
X_test = X_test.drop(['song_id'],axis=1)

X_train.columns = list(range(len(X_train.columns)))
X_test.columns = list(range(len(X_test.columns)))

X_train = X_train.fillna(0)
X_test = X_test.fillna(0)

model =  RandomForestRegressor(random_state = 42)
model.fit(X_train, Y_train)

model1 =  xgb.XGBRegressor(learning_rate = 0.1, verbosity = 1, random_state=42)
model1.fit(X_train, Y_train)

val['RF_preds'] = (model.predict(X_test)+ model1.predict(X_test))/2

songs_df=pd.read_csv('songs.csv')
test = val
df_submission = pd.read_csv('dummy_submission.csv')

user_ids = train.customer_id.unique()
total_users = len(user_ids)


### TRUNCSVD ###

#Removing duplicates
song_matrix = pd.concat([train,test]).drop_duplicates(subset = ['customer_id','song_id'],keep = 'first')
#Creates a song matrix of #numofusers vs #noofsongs
song_matrix = song_matrix.pivot('customer_id','song_id','score')

song_means = song_matrix.mean()
user_means = song_matrix.mean(axis=1)
#Mean shifting
song_shifted_temp = song_matrix-song_means
song_shifted = song_shifted_temp.fillna(0)
#To get locations where we have ratings
mask = -song_shifted_temp.isnull()

def repeated_matrix_reconstruction(num_pcs,num_iterations):
    global song_shifted
    for i in range(num_iterations):
        SVD = TruncatedSVD(n_components=num_pcs,random_state=42)
        SVD.fit(song_shifted)
        #For the ease of applying masks we work with pandas
        song_represented =  pd.DataFrame(SVD.inverse_transform(SVD.transform(song_shifted)),columns=song_shifted.columns,index=song_shifted.index)
        loss = mean_squared_error(song_represented[mask].fillna(0),song_shifted_temp[mask].fillna(0))
        print('Iteration: {} , Loss: {} '.format(i,loss))
        #To just update the non-zero values of song_reprented values to the true ratings
        
        if i < (num_iterations - 1):
            song_represented[mask] = song_shifted_temp[mask]
        
        song_shifted = song_represented
            
    #Mean shifting it back
    song_mat = song_shifted + song_means
    song_mat = song_mat.clip(lower=1,upper=5)
    return song_mat
print("Starting truncated svd with number of components as 20")
representative_matrix_20 = repeated_matrix_reconstruction(50,30)
print("Done")
print("Starting truncated svd with number of components as 15")
representative_matrix_15 = repeated_matrix_reconstruction(15,10)
print("Done")
#bagging
rating_matrix = (representative_matrix_15+representative_matrix_20)/2


trunc_prediction = np.zeros(len(test))
for i in range(len(test)):
    userid =  test.iloc[i,0]
    songid = test.iloc[i,1]
    trunc_prediction[i] = rating_matrix[rating_matrix.index==userid][songid].values[0]

val['RM_preds'] = trunc_prediction

X_test = val[['RM_preds','RF_preds']]

X_test = np.clip(X_test, a_min= 1, a_max = 5)
X_test.loc[np.isfinite(X_test['RM_preds']) == False, 'RM_preds'] = X_test.loc[np.isfinite(X_test['RM_preds']) == False, 'RF_preds']
X_test.loc[np.isnan(X_test['RM_preds']), 'RM_preds'] = X_test.loc[np.isnan(X_test['RM_preds']), 'RF_preds']

preds = final_model.predict(X_test)
PRED = np.around(preds,1)

PRED = np.clip(PRED, a_min# SUBMISSION
 = 1, a_max = 5)

val['preds'] = PRED

test = pd.read_csv('test.csv')

test = test.merge(val[['customer_id','song_id','preds']], how = 'left', on = ['customer_id', 'song_id'])

PRED = test.preds
df_submission = pd.read_csv('dummy_submission.csv')
df_submission.score = PRED
df_submission.to_csv('Final_Submission.csv',index=False)