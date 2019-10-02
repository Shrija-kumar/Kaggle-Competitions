#import packages
import numpy as np
import pandas as pd

#import randomforest
from sklearn.ensemble import RandomForestClassifier

#import the required files
train_df = pd.read_csv("train.csv")
test_df =pd.read_csv("test.csv")

#exploring data
print(train_df.head(10))
print(train_df.info())
print(train_df.describe())

#missing values
print(round(train_df.isnull().sum()/train_df.isnull().count()*100))

#drop poolqc
train_df = train_df.drop(['PoolQC'], axis = 1)
test_df = test_df.drop(['PoolQC'], axis = 1)

#drop miscfeature
train_df = train_df.drop(['MiscFeature'], axis = 1)
test_df = test_df.drop(['MiscFeature'], axis = 1)

#drop fence
train_df = train_df.drop(['Fence'], axis = 1)
test_df = test_df.drop(['Fence'], axis = 1)

#drop alley
train_df = train_df.drop(['Alley'], axis = 1)
test_df = test_df.drop(['Alley'], axis = 1)

#treating the missing values
print(train_df['MasVnrType'].describe())
common_value = 'none'
data = [train_df, test_df]

for dataset in data:
    dataset['MasVnrType'] =dataset['MasVnrType'].fillna(method= 'ffill')
    
print(train_df['GarageType'].describe())

comm_val = 'Attchd'
for dataset in data:
    dataset['GarageType'] =dataset['GarageType'].fillna('comm_val')

print(train_df['GarageType'].isnull().sum())

print(train_df['FireplaceQu'].describe())

fp = 'Gd'
for dataset in data:
    dataset['FireplaceQu'] = dataset['FireplaceQu'].fillna('fp')
    
print(round(train_df.isnull().sum()/train_df.isnull().count()*100)) 

print(round(train_df['LotFrontage'].mean()))

mean_fp = 70
for dataset in data:
    dataset['FireplaceQu'] =dataset['FireplaceQu'].fillna('mean_fp')
    
mean_fp = 70
Mean_ga = 104
for dataset in data:
    dataset['GarageArea'] =dataset['GarageArea'].fillna('mean_ga')
print(train_df['GarageQual'].describe())

mean_fp = 70
for dataset in data:
    dataset['GarageQual'] =dataset['GarageQual'].fillna("TA")

print(train_df['GarageQual'].isnull().sum())

print(train_df['BsmtQual'].describe())
for dataset in data:
    dataset['BsmtQual'] =dataset['BsmtQual'].fillna("TA")
print(train_df.isnull().sum())

print(round(train_df['MasVnrArea'].mean()))
for dataset in data:
    dataset['MasVnrArea'] =dataset['MasVnrArea'].fillna(104)
    
print(train_df['GarageCond'].describe())
for dataset in data:
    dataset['GarageCond'] =dataset['GarageCond'].fillna("TA")
    
print(round(train_df['GarageYrBlt'].mean()))    
for dataset in data:
    dataset['GarageYrBlt'] =dataset['GarageYrBlt'].fillna(1979)    

print(train_df['GarageFinish'].describe())   
   
for dataset in data:
    dataset['GarageFinish'] =dataset['GarageFinish'].fillna("Unf") 
print(train_df.info())

train_df = train_df.drop(['LotFrontage'], axis = 1)
test_df = test_df.drop(['LotFrontage'], axis = 1)

train_df = train_df.drop(['GarageYrBlt'], axis = 1)
test_df = test_df.drop(['GarageYrBlt'], axis = 1)

train_df = train_df.drop(['MasVnrArea'], axis = 1)
test_df = test_df.drop(['MasVnrArea'], axis = 1)

MSZ = {"RL": 0, "RM":1, "C (all)":3, "FV":4}
data = [train_df, test_df]
for dataset in data:
    dataset['MSZoning'] = dataset['MSZoning'].map(MSZ)
    dataset['MSZoning'] = dataset['MSZoning'].fillna(method= 'ffill')

st = {"Grvl": 0, "Pave":1}
data = [train_df, test_df]
for dataset in data:
    dataset['Street'] = dataset['Street'].map(st)

ls = {"Reg": 0, "IR1":1, "IR2":2}
data = [train_df, test_df]
for dataset in data:
    dataset['LotShape'] = dataset['LotShape'].map(ls)
    dataset['LotShape'] = dataset['LotShape'].fillna(method= 'ffill')

lc = {"Lvl": 0, "Bnk":1, "HLS":2, "Low":3}
data = [train_df, test_df]
for dataset in data:
    dataset['LandContour'] = dataset['LandContour'].map(lc)
  
train_df = train_df.drop(['Utilities'], axis = 1)
test_df = test_df.drop(['Utilities'], axis = 1)

lotc = {"Inside": 0, "FR2":1, "Corner":2, "CulDSac":3}
data = [train_df, test_df]
for dataset in data:
    dataset['LotConfig'] = dataset['LotConfig'].map(lotc)
    dataset['LotConfig'] = dataset['LotConfig'].fillna(method= 'ffill')

lotsl = {"Gtl": 0, "Sev":1, "Mod":2}
data = [train_df, test_df]
for dataset in data:
    dataset['LandSlope'] = dataset['LandSlope'].map(lotsl)
    
train_df = train_df.drop(['Neighborhood'], axis = 1)
test_df = test_df.drop(['Neighborhood'], axis = 1)

con= {"Norm": 0, "Feedr":1,"PosN":2, "Artery":3,"RRAe":4,"RRNn":5,"PosA":6}
data = [train_df, test_df]
for dataset in data:
    dataset['Condition1'] = dataset['Condition1'].map(con)    
    dataset['Condition2'] = dataset['Condition2'].map(con)  
    dataset['Condition1'] = dataset['Condition1'].fillna(method= 'ffill')
    dataset['Condition2'] = dataset['Condition2'].fillna(method= 'ffill')
bltp = {"1Fam":1, "2fmCon":2, "Duplex":3,"TwnhsE":4}
data = [train_df, test_df]
for dataset in data:
    dataset['BldgType'] = dataset['BldgType'].map(bltp)
    dataset['BldgType'] = dataset['BldgType'].fillna(method= 'ffill')

hs= {"1Story":0,"1.5Fin":1, "1.5Unf":2, "2Story":3,"2.5Unf":4,"SLvl":5}
data = [train_df, test_df]
for dataset in data:
    dataset['HouseStyle'] = dataset['HouseStyle'].map(hs)
    dataset['HouseStyle'] = dataset['HouseStyle'].fillna(method= 'ffill')

rs= {"Gable":0,"Hip":1, "Gambrel":2, "Mansard":3,"Flat":4}
data = [train_df, test_df]
for dataset in data:
    dataset['RoofStyle'] = dataset['RoofStyle'].map(rs)
    dataset['RoofStyle'] = dataset['RoofStyle'].fillna(method= 'ffill')

rm= {"CompShg":0,"WdShngl":1,"Metal":2,"WdShake":3,"Membran":4,"Tar&Grv":5}
data = [train_df, test_df]
for dataset in data:
    dataset['RoofMatl'] = dataset['RoofMatl'].map(rm)
    dataset['RoofMatl'] = dataset['RoofMatl'].fillna(method= 'ffill')

ext= {"VinylSd":0,"MetalSd":1,"Wd Sdng":2,"HdBoard":3,"Plywood":4,"Stucco":5,"CemntBd":6}
data = [train_df, test_df]
for dataset in data:
    dataset['Exterior1st'] = dataset['Exterior1st'].map(ext)
    dataset['Exterior2nd'] = dataset['Exterior2nd'].map(ext)  
    dataset['Exterior1st'] =dataset['Exterior1st'].fillna(method= 'ffill')
    dataset['Exterior2nd'] =dataset['Exterior2nd'].fillna(method= 'ffill')
    
mvt = {"None":0, "BrkFace":1,"BrkCmn":2,"Stone":3,"NA":4}
data = [train_df, test_df]
for dataset in data:
    dataset['MasVnrType'] = dataset['MasVnrType'].map(mvt)
    
train_df = train_df.drop(['ExterQual'], axis = 1)
test_df = test_df.drop(['ExterQual'], axis = 1)

train_df = train_df.drop(['ExterCond'], axis = 1)
test_df = test_df.drop(['ExterCond'], axis = 1)

train_df = train_df.drop(['Foundation'], axis = 1)
test_df = test_df.drop(['Foundation'], axis = 1)

train_df = train_df.drop(['BsmtQual'], axis = 1)
test_df = test_df.drop(['BsmtQual'], axis = 1)

train_df = train_df.drop(['BsmtCond'], axis = 1)
test_df = test_df.drop(['BsmtCond'], axis = 1)
    
bft1 = {"GLQ":0, "ALQ":1,"Unf":2,"Rec":3,"LwQ":4,"NA":5,"BLQ":6,"Fin":7,"RFn":8}
data = [train_df, test_df]
for dataset in data:
    dataset['GarageFinish'] = dataset['GarageFinish'].map(bft1)

train_df = train_df.drop(['BsmtExposure'], axis = 1)
test_df = test_df.drop(['BsmtExposure'], axis = 1)


train_df = train_df.drop(['BsmtFinType1'], axis = 1)
test_df = test_df.drop(['BsmtFinType1'], axis = 1)

train_df = train_df.drop(['BsmtFinType2'], axis = 1)
test_df = test_df.drop(['BsmtFinType2'], axis = 1)

train_df = train_df.drop(['Heating'], axis = 1)
test_df = test_df.drop(['Heating'], axis = 1)

hqc = {"TA":0, "Gd":1,"Ex":2,"NA":3,"Fa":4,"Po":5}
data = [train_df, test_df]
for dataset in data:
    dataset['HeatingQC'] = dataset['HeatingQC'].map(hqc)
    
ca = {"N":0, "Y":1}
data = [train_df, test_df]
for dataset in data:
    dataset['CentralAir'] = dataset['CentralAir'].map(ca)
 
train_df = train_df.drop(['Electrical'], axis = 1)
test_df = test_df.drop(['Electrical'], axis = 1)
    

kq = {"TA":0, "Gd":1,"Ex":2,"NA":3,"Fa":4}
data = [train_df, test_df]
for dataset in data:
    dataset['KitchenQual'] = dataset['KitchenQual'].map(kq)

fun = {"Typ":0, "Min1":1,"Min2":2,"Maj1":1,"Mod":4,"Maj2":5,"Sev":6}
data = [train_df, test_df]
for dataset in data:
    dataset['Functional'] = dataset['Functional'].map(fun)

train_df = train_df.drop(['FireplaceQu'], axis = 1)
test_df = test_df.drop(['FireplaceQu'], axis = 1)

train_df = train_df.drop(['GarageType'], axis = 1)
test_df = test_df.drop(['GarageType'], axis = 1)

train_df = train_df.drop(['GarageQual'], axis = 1)
test_df = test_df.drop(['GarageQual'], axis = 1)

train_df = train_df.drop(['GarageCond'], axis = 1)
test_df = test_df.drop(['GarageCond'], axis = 1)

train_df =train_df.drop(['GarageArea'], axis = 1)
test_df = test_df.drop(['GarageArea'], axis = 1)

train_df = train_df.drop(['SaleType'], axis = 1)
test_df = test_df.drop(['SaleType'], axis = 1)

train_df = train_df.drop(['SaleCondition'], axis = 1)
test_df = test_df.drop(['SaleCondition'], axis = 1)

pd = {"N":0, "Y":1,"P":2}
data = [train_df, test_df]
for dataset in data:
    dataset['PavedDrive'] = dataset['PavedDrive'].map(pd)
    dataset.fillna(dataset.mean(), inplace=True)
print(train_df.info())
train_df = train_df.fillna(method='ffill')

test_df = test_df.fillna(method = 'ffill')
np.nan_to_num(train_df)
print(np.where(np.isnan(train_df)))
print(np.where(np.isnan(test_df)))


#building models
X_train = train_df.drop(['SalePrice'], axis = 1)
X_train = X_train.astype('float64')
Y_train = train_df['SalePrice']
Y_train = Y_train.astype('float64')
X_test  = test_df
X_test = X_test.astype('float64')


#randomforest
random_forest = RandomForestClassifier(n_estimators = 10)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)
random_forest.score(X_train,Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train)*100,2)
print(acc_random_forest)

#saving the predictions in a csv file
print(Y_prediction)
open('predictions.csv','w')
np.savetxt('predictions.csv', Y_prediction, delimiter=',')

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_train, Y_prediction)
print(cm)

