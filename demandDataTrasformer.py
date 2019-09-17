import os
import pandas

def transformDemandData():
    #transform data for demand forecasting model
    for file in os.listdir('../demandData'):
        data = pandas.read_csv('../demandData/'+file)
        for i, row in data.iterrows():
            data.set_value(i, 'y', 1 - data.iloc[i]['y'] )
            data.set_value(i, 'y_hat', 1 - data.iloc[i]['y_hat'])
        data.to_csv('../demandData/transformed/'+ file, index=False)
