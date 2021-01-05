
import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import OneHotEncoder 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

#importing the dataset
housing=pd.read_csv(r'/Users/rohansingh/Desktop/housing.csv')
#housing.head(10)
housing.info()
housing.describe()
#housing.hist( bins=50, figsize=(20,15))
#plt.show()


housing["income_cat"]=pd.cut(housing["median_income"], bins=[0,1.5,3.0,4.5,6, np.inf],labels=[1,2,3,4,5])
housing["income_cat"].hist()
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set=housing.loc[train_index]
    strat_test_set=housing.loc[test_index]
    strat_test_set["income_cat"].value_counts()/len(strat_test_set)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

#housing=strat_train_set.copy()
#housing.plot(kind="scatter",x="longitude",y="latitude",alpha=0.4,s=housing["population"]/100,label="population",figsize=(10,7),c="median_house_value",cmap=plt.get_cmap("jet"))
#plt.legend()

#Preparing for MAchine Learning
housing=strat_train_set.drop("median_house_value",axis=1)
housing_labels=strat_train_set["median_house_value"].copy()
housing_num=housing.drop("ocean_proximity",axis=1)
housing_cat=housing[["ocean_proximity"]]
room_ix, bedrooms_ix, population_ix, households_ix=3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room =True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self 
    def transform(self, X):
        rooms_per_household=X[:,room_ix]/ X[:,households_ix]
        population_per_household =X[:,population_ix]/X[:,households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room=X[:,bedrooms_ix]/X[:,room_ix]
            return np.c_[X, rooms_per_household,population_per_household,bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household,population_per_household]
attr_adder=CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs=attr_adder.transform(housing.values)

num_pipeline=Pipeline([('imputer',SimpleImputer(strategy="median")),
                        ('attribs_adder',CombinedAttributesAdder()),
                        ("std_scaler",StandardScaler()),])

housing_num_tr=num_pipeline.fit_transform(housing_num)

num_attribs=list(housing_num)
cat_attribs=["ocean_proximity"]

full_pipeline=ColumnTransformer([
            ("num",num_pipeline, num_attribs),
            ("cat",OneHotEncoder(),cat_attribs),
        ])

housing_prepared=full_pipeline.fit_transform(housing)

#training and Evaluating model on training set

        


