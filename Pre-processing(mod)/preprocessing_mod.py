
# coding: utf-8
# In[1]:
# Import Library
# import numpy as np
from datetime import datetime

from sklearn.preprocessing import Imputer

from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Create data frame from excel file
X = pd.read_csv("/Users/cristian/Downloads/censored_data_v2.csv", na_filter=False)

# Change the creation date column to datetime
X['CREATION DATE'] = pd.to_datetime(X['CREATION DATE'])

# Add column for day of week
X['weekday'] = X['CREATION DATE'].dt.weekday_name

# Separate rows without birth date
noDate = X[X['BIRTH DATE'] == '']
X = X[X['BIRTH DATE'] != '']

# Calculate age
X['AGE'] = ((datetime.today() - pd.to_datetime((X["BIRTH_YEAR"]*10000+X["BIRTH_MONTH"]*100+X["BIRTH_DAY"]).apply(str),
                                               format='%Y%m%d'))/365).astype('timedelta64[D]').astype(int)
X['AGE'] = X['AGE'].astype(int)

# Removes unwanted sub types
table = X
table = table[~table['SUB TYPE'].isin(['DEN', 'ELD', 'REF'])]

# Creates columns for every disability, mobility aid and pass on
encode = LabelEncoder()
toExpand = 'DISABILITY,MOBILITY AID,SUB TYPE,weekday'.split(',')
for col in toExpand:
    table = pd.concat([table, table[col].str.get_dummies(sep=',')], axis=1)

table = pd.concat([table, pd.get_dummies(table['PURPOSE ID'])], axis=1)

table = pd.concat([table, pd.get_dummies(table['GENDER '])], axis=1)

# Adds block reservation values
blocks = table.groupby(['ID', 'CREATION DATE']).size().reset_index(name='block')
table = table.merge(blocks, how="left", on=['ID', 'CREATION DATE'])

# Adds anticipated column - with days of anticipation
reservation_date = pd.to_datetime((table["Create_YEAR"] * 10000 + table["Create_MONTH"] * 100 + table["Create_DAY"])
                                  .apply(str), format='%Y%m%d')

trip_date = pd.to_datetime((table["DATE_YEAR"] * 10000 + table["DATE_MONTH"] * 100 + table["DATE_DAY"])
                           .apply(str), format='%Y%m%d')

table["ANTICIPATED"] = (trip_date - reservation_date).astype('timedelta64[D]').astype(int)

table = table.sort_values(by=['ID', 'CREATION DATE'])

# Removes unnecessary columns
toRemove = 'CREATION DATE,BIRTH_DAY,BIRTH_MONTH,GENDER ,PURPOSE ID,MOBILITY AID,DATE YYYY/MM/DD,DISABILITY,' \
           'BIRTH DATE,SCH TIME,weekday,SUB TYPE'.split(',')
for item in toRemove:
    del table[item]

# Substitue missing values using the mode.
imr = Imputer(missing_values='NaN', strategy='mean')
imr = imr.fit(table)
table["AGE"] = imr.fit_transform(table[["AGE"]]).ravel()

table.to_csv("preprocessed-data.csv")
