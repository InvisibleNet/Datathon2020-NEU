#!/usr/bin/env python
# coding: utf-8

# In[296]:


import sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dateutil import parser
#import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import init
import numpy as np


#model and function need to use
class gruModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, hidden_layer):
        super().__init__()
        self.gruLayer = nn.GRU(in_dim, hidden_dim, hidden_layer)
        self.fcLayer = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        out, _ = self.gruLayer(x)
        #out = out[12:]
        out = self.fcLayer(out)
        return out

#function of GRU
class GRU(nn.Module):
    # 输入维度为1，输出维度为1，隐藏层维数为5, 定义LSTM层数为2
    def __init__(self, data_ori, start_date, end_date, train = False):
        super(GRU, self).__init__()
        self.gru = gruModel(1, 5, 1, 2)
        #print(data_ori)
        if not train: #there have model in our file
            self.gru.load_state_dict(torch.load("./gru_model.pth"))  #the model should in the same file
        self.cuda = torch.cuda.is_available() 
        self.data_full_version = data_ori
        self.start_index = 0#index(data[:,0]==start_date)
        self.end_index = 0
        for i in range(len(data_ori)):
            #print(self.data)
            #print(self.data_full_version[i,0], start_date)
            if self.data_full_version[i,0]==start_date:
                self.start_index = i
            if self.data_full_version[i,0]==end_date:
                self.end_index = i
        #print(data_ori[-1,1],data_ori[end_index,1])
        sc = MinMaxScaler(feature_range=(1e-10,1))  #MinMaxScaler(feature_range=(1e-5,1))
        temp = data_ori[:,1].astype("float32")
        self.back_value = max(data_ori[:,1])#*10
        temp1 = torch.from_numpy(temp)
        temp2 = temp1.unsqueeze(1)
        self.data = torch.from_numpy(sc.fit_transform(temp2).astype("float32"))

    def train(self, start_date = '2020-03-01', end_date = '2020-04-15'):
        if self.cuda:
            self.gru = self.gru.cuda()
            self.data = self.data.cuda()
        # 定义损失函数和优化函数
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.gru.parameters(), lr=0.01)
        #determine the training set
        #length = int(0.5*len(self.data))
        #temp = self.data[:length]
        #train_set = torch.from_numpy(temp)
        #train_set = temp.unsqueeze(1)
        self.train_set = self.data.unsqueeze(2)
        # 训练模型
        frq, sec = 4000, 400
        loss_set = []
        start_index = 0
        end_index = 0
        start_index = self.start_index#index(data[:,0]==start_date)
        end_index = self.end_index
        self.num_data_train = end_index - start_index # to April 1st
        predict_step = 1
        start = start_index
        #print(self.train_set)
        #print(start_index, end_index)
        x = self.train_set[start_index:end_index]#.transpose
        #assert num_data_train < length 
        #print("num_data_train is too large!")#decide

        y = self.train_set[start_index+predict_step:(end_index+predict_step)]#.transpose
        iteration = 2
        for i in range(iteration):
            for e in range(1, frq + 1):
                #print(x,y)
                inputs = Variable(x)
                #print(inputs)
                target = Variable(y)
                #forward
                output = self.gru(inputs)
                loss = criterion(output, target)
                # update paramters
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #print training information
                print_loss = loss.item()
                loss_set.append((e, print_loss))
                if e % sec == 0 :
                    print('Epoch[{}/{}], Loss: {:.5f}'.format(e, frq, print_loss))
                #start += 1
                #if start + num_data_train + predict_step >= length:
                #    start = start_date
                #x = train_set[start:(start+num_data_train)]
                #y = train_set[(start+predict_step):(start+num_data_train+predict_step)]
        self.gru = self.gru.eval()
        torch.save(self.gru.state_dict(), "./gru_model_new.pth")
        
    def test(self, prediction_nums,start_date_train = '2020-03-01', end_date_train = '2020-04-15'):
        result_list = []
        start_index = self.start_index#index(data[:,0]==start_date)
        end_index = self.end_index
        
        num_data_test = end_index - start_index
        #num_data_test = 70
        temp = self.data[start_index:(end_index)]
        #temp = temp.unsqueeze(1)
        data_test = temp.unsqueeze(2)
        use = self.data.unsqueeze(2)
        if not start_index == 0:
            result_list = list(self.back_value*np.array(use[:start_index,0,0].cpu().detach().numpy()))
        result_list.extend(list(self.back_value*np.array(data_test[:,0,0].cpu().detach().numpy())))#[:,0,0]
        #print(result_list)
        assert num_data_test+prediction_nums < len(self.data)
        for i in range(num_data_test, num_data_test+prediction_nums):
            #print(len(result_list))
            use_data = Variable(data_test)
            predict = self.gru(use_data)
            result_list.append(self.back_value*predict[-1,0,0].cpu().detach().numpy())
            data_test[:-1] = data_test[1:]
            data_test[-1] = predict[-1]
            #print(predict[-1])
        # if want to predict future 11,12,1/21, you can comment the plot
        plt.figure()
#       plt.plot([parser.parse(str(data_full_version[i,0])) for i in range(data_full_version.shape[0])], data_full_version[:,1],'r')
#       plt.plot([parser.parse(str(data_full_version[i,0])) for i in range(data_full_version.shape[0])], result_list[:], 'b')
        plt.plot([parser.parse(str(self.data_full_version[i,0])) for i in range(end_index + prediction_nums)], self.data_full_version[:(end_index + prediction_nums),1],'r')
        plt.plot([parser.parse(str(self.data_full_version[i,0])) for i in range(end_index + prediction_nums)], result_list[:], 'g')
        plt.plot([parser.parse(str(self.data_full_version[i,0])) for i in range(end_index,end_index + prediction_nums)], result_list[end_index:], 'b')
        plt.legend(["Ground_truth","Prediction train part","Prediction predict part"])
        plt.title( "From 1/22/20 to "+ self.data_full_version[end_index+prediction_nums,0])
        plt.savefig('./save_imge.png')
        plt.figure()
        plt.plot([parser.parse(str(self.data_full_version[i,0])) for i in range(start_index,end_index + prediction_nums)], self.data_full_version[start_index:(end_index + prediction_nums),1],'r')
        plt.plot([parser.parse(str(self.data_full_version[i,0])) for i in range(start_index,end_index + prediction_nums)], result_list[start_index:], 'g')
        plt.plot([parser.parse(str(self.data_full_version[i,0])) for i in range(end_index,end_index + prediction_nums)], result_list[end_index:], 'b')
        plt.legend(["Ground_truth","Prediction train part","Prediction predict part"])
        plt.title( "From "+ self.data_full_version[start_index,0]+" to "+ self.data_full_version[end_index+prediction_nums,0])
        #print([parser.parse(str(self.data_full_version[i,0])) for i in range(start_index,end_index + prediction_nums)])
        plt.savefig('./day_by_day_imge.png')
        #plot the population map
        population_pred = []
        population_ground = []
        acc = 0
        for i in range(len(result_list)):
            print(result_list[i], self.data_full_version[i])
            if i == 0:
                population_pred.append(result_list[i])
                population_ground.append(self.data_full_version[i,1])
                acc_pred = result_list[i]
                acc_ground = self.data_full_version[i,1]
            else:
                acc_pred += result_list[i]
                acc_ground += self.data_full_version[i,1]
                population_pred.append(acc_pred)
                population_ground.append(acc_ground)
        #print(population_ground[start_index:(end_index + prediction_nums)])
        plt.figure()
        plt.plot([parser.parse(str(self.data_full_version[i,0])) for i in range(start_index,end_index + prediction_nums)], population_ground[start_index:(end_index + prediction_nums)],'r')
        plt.plot([parser.parse(str(self.data_full_version[i,0])) for i in range(start_index,end_index + prediction_nums)], population_pred[start_index:], 'g')
        plt.plot([parser.parse(str(self.data_full_version[i,0])) for i in range(end_index,end_index + prediction_nums)], population_pred[end_index:], 'b')
        plt.legend(["Ground_truth","Prediction train part","Prediction predict part"])
        plt.title( "From "+ self.data_full_version[start_index,0]+" to "+ self.data_full_version[end_index+prediction_nums,0])
        plt.savefig('./accumalate_imge.png')
        
        return result_list
#process data
#read in the data
def readDataFromSource(source, colname):
    df = pd.read_csv(source)
    df = df[colname]
    return df
def getDataFromColcsv(source,statecolname,startcol,state):
    #df =pd.read_csv(source)
    #print(df)
    #df = df.groupby(statecolname).get_group(state)#.sum()
    #print(df)
    #df = df[startcol:]
    #df = df.reset_index()
    #df = df[]#df[:,4]]
    table = pd.read_csv(source)
    casesByDate = table[['date','state','cases']]
    stateData = pd.DataFrame(casesByDate.loc[casesByDate['state']==state])
    stateData['y'] = stateData['cases'].diff(1)
    stateData = stateData[["date","y"]]
    #print(stateData)
    return stateData.values#df.values#to_numpy()
def getStateData(df, state):
    data = np.flip(df.groupby('state').get_group(state).to_numpy(),0)
    return data[:,1:3]
def plotStateTrend(df, state):
    data = getStateData(df, state)
    plt.plot([parser.parse(str(data[i,0])) for i in range(data.shape[0])],data[:,1])
    plt.title("Cases over time in "+state )
def plotStateTrendWithData(data, state):
    plt.plot([parser.parse(str(data[i,0])) for i in range(data.shape[0])],data[:,1])
    plt.title("Cases over time in "+state )
    

def GRU_prediction(data_path, statename, startdate, enddate, prediction_nums, train_model = False):
    data_ori = getDataFromColcsv(data_path, 'state', 4, statename)
    if data_ori[0,1]:
        data_ori[0,1] = 0
    #print(data_ori)
    gru_model = GRU(data_ori, startdate, enddate, train = train_model)
    if train_model:
        gru_model.train(startdate, enddate)
    predict_result = gru_model.test(prediction_nums, startdate, enddate)
    return predict_result[-prediction_nums:]
#data[:,0] means date, data[:,1] means number of person


# In[297]:


#read out the data
all_state_dataset_path = "dataset/all-states-history.csv" #state
USA_dataset_path = "dataset/covid_confirmed_usafacts.csv" #State, WY
state_dataset_path = "dataset/covid-19-state-level-data.csv" #state, California
time_series_dataset_path = "dataset/time_series_covid19_confirmed_US.csv" # Province_State, Alabama
#use the function Example is here
result = GRU_prediction(state_dataset_path,'New York', '2020-03-01','2020-04-15', 30, False) #ture or false means use the trained module or train the model by yourself
#result = GRU_prediction(USA_dataset_path,'CA', '4/1/20','5/1/20', 30, False)


# In[ ]:





# In[ ]:





# In[ ]:




