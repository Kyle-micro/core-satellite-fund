# -*- coding: utf-8 -*-

#重跑時只需確認 rets,noa 

import math
import numpy as np
import pandas as pd
from pylab import plt
np.set_printoptions(suppress=True)
from scipy.optimize import minimize

def portfolio_return(weights, rets):
    return np.dot(weights.T, rets.mean()) * 252

def portfolio_variance(weights, rets):
    return np.dot(weights.T, np.dot(rets.cov(), weights)) * 252

def portfolio_volatility(weights, rets):
    return math.sqrt(portfolio_variance(weights, rets))
#定義相對的風險歸因函數
def rel_risk_contributions(weights, rets):
    vol = portfolio_volatility(weights, rets)
    cov = rets.cov()
    mvols = np.dot(cov, weights) / vol
    rc = mvols * weights
    rrc = rc / rc.sum()
    return rrc

noa = 3 #標的資產數量
weights =  np.array(noa * [1 / noa]) #等權重
rets = pd.read_csv('因子投組.csv').drop(['Unnamed: 0','cen','mom'],axis=1)
n=30 #以30天的報酬率計算投組風險，決定下一天的權重
rets1 = rets.iloc[:n,1:] 

#最小化實際的rrc與目標的rrc
def mse_risk_contributions(weights, target, rets):
    rrc = rel_risk_contributions(weights, rets)
    mse = ((rrc - target) ** 2).mean()
    return mse * 100
#限制式 : 權重加總為1
cons = {'type': 'eq', 'fun': lambda weights: weights.sum() - 1}
#限制式 :權重介於0-1之間 
bnds = noa * [(0, 1),]
#目標的相對風險歸因
target = noa * [1 / noa,]
#求解符合目標下的標的資產權重
opt = minimize(lambda w: mse_risk_contributions(w, target=target,rets=rets1), weights, bounds=bnds, constraints=cons)
weights_ = opt['x']

#再不調整權重的情況下，新權重=原本權重*(1+報酬率)/sum(原本權重*(1+報酬率))
w0 =  weights_ #初始權重
dev = 0.15
up_bw = np.array(target)*(1+dev)
low_bw = np.array(target)*(1-dev)
wlist = [list(weights_)+['Risk Parity']]
for i in range(1,rets.shape[0]-n+1):
    #隨著時間經過資料筆數越來越多，不是移動窗格
    rets1 = rets.iloc[:n+i]
    w1 = np.array((w0*(1+rets.iloc[n+i-1,1:]))/sum(w0*(1+rets.iloc[n+i-1,1:])))
    rrc1 = rel_risk_contributions(w1,rets1)
    # Opportunistic Rebalancing
    if sum(rrc1 > low_bw)+sum(rrc1 < up_bw)==2*noa:
        wlist.append(list(w1)+[0])
    else:
        opt = minimize(lambda w: mse_risk_contributions(w, target=target,rets=rets1), w0, bounds=bnds, constraints=cons)
        w1 = opt['x']
        wlist.append(list(w1)+[1])
    w0 = w1
    
wT = pd.DataFrame(wlist)
Date = list(rets['Date'][n:].values)+['2022-11-28']
wT.insert(0,'Date',Date)      
wT.columns = ['Date'] + list(rets.columns[1:].values) + ['Rebalancing']     
wT.to_csv('Allocation敘統/多因子投組權重.csv')
