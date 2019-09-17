import pandas as pd

# csv with class 1 only
cOneData = pd.read_csv('Class1Data.csv')
# csv with all data
originData = pd.read_csv('updated_data.csv')

filteredUNOS = cOneData.ix[~(cOneData.index > 198625)]
filtered = originData.ix[~(originData['CLASS'] < 2)]

balancedData = pd.concat([filteredUNOS, filtered], ignore_index=True)

balancedData.to_csv('balanced_data.csv', index=False)
