import numpy as np
from flask import Flask, jsonify, make_response
from flask_restplus import Resource, Api, fields
from flask import request, stream_with_context, Response
from flask_cors import CORS
import json, csv
from werkzeug.utils import cached_property
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
import yfinance as yf
import pandas_datareader as pdr
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM

app = Flask(__name__)
CORS(app)
api = Api(app,default='Modules',default_label='Stock Forecasting Model')

name = api.model('name', {
        'name_id': fields.String(description='Enter Name', required=True, example='ABCD')
    })



# @app.route('/')
# def home():
#     return render_template('index.html')

@api.route('/prediction')
class LargeNew(Resource):
    @api.expect(name)
    def post(self):
        data = request.json
        name_id = data['name_id']
        print(name_id)

        if name_id=="ASHOKLEY.NS":
            modelname='my_model_ashok'
            model_new = keras.models.load_model(modelname)
        elif name_id=='TATACOFFEE.NS':
            modelname='my_model_coffee'
            model_new = keras.models.load_model(modelname)
        else:
            modelname='my_model_generic'
            model_new = keras.models.load_model(modelname)


        import pandas_datareader as pdr
        import pandas as pd
        msft = yf.Ticker(name_id)
        df_ori = msft.history('10y',interval='1d')
        df=df_ori.reset_index()
        print(df.tail(1))
        import datetime
        # hour=datetime.datetime.now().hour
        from datetime import datetime
        from pytz import timezone    

        ist = timezone('Asia/Kolkata')
        ist_time = datetime.now(ist)
        hour=ist_time.strftime('%H')
        # hour=9
        print('hour',hour)
        if int(hour) in [16,17,18,19,20,21,22,23,0,1,2,3,4,5,6,7]:
            print('if')
            df=df[0:df.shape[0]]
        else:
            print('else')
            df=df[0:df.shape[0]-1]    

        # df=df[0:2461]
        # df.tail(3)

        df1=df[['Date','Close']]
        df1['Date']=pd.DatetimeIndex(df1['Date'])
        df1['Date2']=pd.DatetimeIndex(df1['Date']).date


        print(df1.shape)

        df1.head(3)

        del df1['Date']
        df1=df1.set_index('Date2')
        print(df1.shape)

        import pandas as pd
#         df=pd.read_csv('AAPL.csv')

#         df1=df.reset_index()['close']
        df=df1.copy()
        train = df[0:round(df1.shape[0]*0.70)]
        valid = df[round(df1.shape[0]*0.70):]
        # creating dataframe
        # training_size=int(len(df1)*0.65)
        # test_size=len(df1)-training_size
        # train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1),:1]
        # print('train_data',train_data.shape)
        # print('test_data',test_data.shape)

        scaler = MinMaxScaler(feature_range=(0, 1))
        valid_scaled_data = scaler.fit_transform(valid)
        

        x_input=valid_scaled_data[valid_scaled_data.shape[0]-100:].reshape(1,-1)
        x_input.shape
        
        temp_input=list(x_input)
        temp_input=temp_input[0].tolist()
        print('tempinput',len(temp_input))
        # demonstrate prediction for next 10 days
        # demonstrate prediction for next 10 days
        from numpy import array

        lst_output=[]
        n_steps=100
        i=0
        while(i<7):

            if(len(temp_input)>100):
                #print(temp_input)
                x_input=np.array(temp_input[1:])
                # print("{} day input {}".format(i,x_input))
                x_input=x_input.reshape(1,-1)
                x_input = x_input.reshape((1, n_steps, 1))
                #print(x_input)
                yhat = model_new.predict(x_input, verbose=0)
                # print("{} day output {}".format(i,yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input=temp_input[1:]
                #print(temp_input)
                lst_output.extend(yhat.tolist())
                i=i+1
            else:
                x_input = x_input.reshape((1, n_steps,1))
                yhat = model_new.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                lst_output.extend(yhat.tolist())
                i=i+1


    #     print(lst_output)
    #     day_new=np.arange(1,101)
    #     day_pred=np.arange(101,131)
    #     df3=df1.tolist()
    #     df3.extend(lst_output)

    # #     df3=scaler.inverse_transform(df3).tolist()
    #     df3=scaler.inverse_transform(df3)
    #     pred=pd.DataFrame(df3)
    #     pred['Time']='2020-11-12'
    #     pred.columns=['pred','timestamp']
    #     print(pred.head(3))
    #     print('out',json.dumps(pred.to_dict(orient='records')))
    #     # pred.head(3)
    #     response = make_response(json.dumps(pred.to_dict(orient='records')))
    #     # response.headers['content-type'] = 'application/octet-stream'
    #     return response
        
#         plt.plot(day_pred,scaler.inverse_transform(lst_output),color='blue')
        # df1['Date']=pd.DatetimeIndex(df1['Date'])
        print('output',len(lst_output))
        last=df1.index[-1]
        print('last',last)
        import datetime

        future_dates=pd.date_range(start=last+datetime.timedelta(days=1), periods=7)
        future_dataset=pd.DataFrame()
        future_dataset=pd.DataFrame()
        future_dataset['date']=future_dates
        future_dataset['Close']=scaler.inverse_transform(lst_output)

        future_dataset['date']=pd.DatetimeIndex(future_dataset['date'])
        future_dataset['date']=pd.DatetimeIndex(future_dataset['date']).date
        future_dataset['weekday']=pd.DatetimeIndex(future_dataset['date']).weekday
        future_dataset=future_dataset[future_dataset['weekday'].isin([0,1,2,3,4])]
        future_dataset['date']=future_dataset['date'].astype('str')
        del future_dataset['weekday']

        future_dataset['StockName']=name_id
        print(future_dataset.shape)
        print(future_dataset.head(3))
        print('out',json.dumps(future_dataset.to_dict(orient='records')))
        response = make_response(json.dumps(future_dataset.to_dict(orient='records')))
        response.headers['content-type'] = 'application/octet-stream'
        return response
        

name = api.model('name', {
        'name_id': fields.String(description='Enter Name', required=True, example='TATACOFFEE.NS')
    })


# @api.route('/model_validation')
# class LargeNew2(Resource):
#     @api.expect(name)
#     def post(self):
#         data = request.json
#         name_id = data['name_id']
#         if name_id=='ASHOKLEY.NS':
#             data=pd.read_csv('model_validations_ASHOKLEY.NS.csv')
#         elif name_id=='TATACOFFEE.NS':
#             data=pd.read_csv('model_validations_TATACOFFEE.NS.csv')
            
#         response = make_response(json.dumps(data.to_dict(orient='records')))
#         response.headers['content-type'] = 'application/octet-stream'
#         return response


@api.route('/model_validation')
class LargeNew3(Resource):
    @api.expect(name)
    def post(self):

        data = request.json
        name_id = data['name_id']
        
        if name_id=="ASHOKLEY.NS":
            modelname='my_model_ashok'
            model_new = keras.models.load_model(modelname)
        elif name_id=='TATACOFFEE.NS':
            modelname='my_model_coffee'
            model_new = keras.models.load_model(modelname)
        else:
            modelname='my_model_generic'
            model_new = keras.models.load_model(modelname)


        msft = yf.Ticker(name_id)
        df_ori = msft.history('10y',interval='1d')
        df=df_ori.reset_index()
        # df=df[0:df.shape[0]-1]
        import datetime
        # hour=datetime.datetime.now().hour
        from datetime import datetime
        from pytz import timezone    

        ist = timezone('Asia/Kolkata')
        ist_time = datetime.now(ist)
        hour=ist_time.strftime('%H')
        
        # hour=9

        if int(hour) in [16,17,18,19,20,21,22,23,0,1,2,3,4,5,6,7]:
            df=df[0:df.shape[0]]
        else:
            df=df[0:df.shape[0]-1]    

        df1=df[['Date','Close']]
        df1['Date']=pd.DatetimeIndex(df1['Date'])
        df1['Date2']=pd.DatetimeIndex(df1['Date']).date
        #creating dataframe
        # data = df1.sort_index(ascending=True, axis=0)
        del df1['Date']
        df1=df1.set_index('Date2')
        df1.tail(3)

        #creating train and test sets
        dataset = df1.values

        train = dataset[0:round(df1.shape[0]*0.70),:]
        valid = dataset[round(df1.shape[0]*0.70):,:]
        # dataset.shape
        print(train.shape)
        print(valid.shape)

        #converting dataset into x_train and y_train
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        x_train, y_train = [], []
        for i in range(100,len(train)):
            x_train.append(scaled_data[i-100:i,0])
            y_train.append(scaled_data[i,0])
        x_train, y_train = np.array(x_train), np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))


        inputs = df1[len(df1) - len(valid) - 100:].values
        inputs = inputs.reshape(-1,1)
        inputs  = scaler.transform(inputs)


        X_test = []
        for i in range(100,inputs.shape[0]):
            X_test.append(inputs[i-100:i,0])
        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        closing_price = model_new.predict(X_test)
        closing_price = scaler.inverse_transform(closing_price)

        train2 = df1[:round(df1.shape[0]*0.70)]
        valid2= df1[round(df1.shape[0]*0.70):]
        valid2['Predictions'] = closing_price
        valid2['Predictions'] = round(valid2['Predictions'],3)
        valid2.columns=['Original','Predictions']
        valid3=valid2.tail(20)
        valid3=valid3.reset_index()
        valid3.columns=['Date','Original','Predictions']
        valid3['StockName']=name_id
        valid3['Date']=valid3['Date'].astype('str')


        a_dict={}
        def mean_absolute_percentage_error(y_true, y_pred): 
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        print('MAPE',mean_absolute_percentage_error(closing_price, valid))
        print(hour)
        error=round(mean_absolute_percentage_error(closing_price, valid),2)
        a_dict['Model Accuracy']=str(100-error)+'%'

        for i in valid3['Date'].unique():
            dataq=valid3[valid3['Date']==i]
            del dataq['Date']
            a_dict[i]=dataq.to_dict(orient='records')
            



        response = make_response(json.dumps(a_dict))
        response.headers['content-type'] = 'application/octet-stream'
        return response




if __name__ == "__main__":
    app.run(port=6002,debug=True)
