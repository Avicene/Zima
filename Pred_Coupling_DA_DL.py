
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
import math
import copy
import pickle
import seaborn as sns


def Calc_Param_bar(df_Param,CFG):
    df_Param_bar =copy.copy(df_Param)
    for i,parm_nm in enumerate(CFG.Param_Name): 
        df_Param_bar[parm_nm] =( 2*df_Param[parm_nm] - CFG.Param_Bounds[i][0] -CFG.Param_Bounds[i][1])\
                        / (CFG.Param_Bounds[i][1]-CFG.Param_Bounds[i][0])
    return df_Param_bar

def Calc_Param(df_Param_bar,CFG):
    df_Param =copy.copy(df_Param_bar)
    for i,parm_nm in enumerate(CFG.Param_Name): 
        df_Param[parm_nm] =0.5*(CFG.Param_Bounds[i][1] -CFG.Param_Bounds[i][0])*df_Param_bar[parm_nm] + 0.5*(CFG.Param_Bounds[i][1] + CFG.Param_Bounds[i][0])
    return df_Param

def Compute_Bary(Param) :
    bary = np.zeros(Param.shape[1])
    for prm in Param :
        bary += prm
    bary = (1/len(Param))*bary
    return bary

def Calc_Cg(df, i_main_loop,Param_Name):
    df_1 = df.loc[(df["DA_Iter"] == i_main_loop-1)]
    Cg1 = Compute_Bary(df_1[Param_Name].values)
    ## Add Cg in DataFram :
    return Cg1


def Selc_pred_Data(df_prm, Param_Name, Cg, DL_Rc, i_main_loop, DL_Pred_max):
    Param = df_prm.loc[ (df_prm["DA_Iter"] ==i_main_loop) ][Param_Name].values
    Pred_Param = []
    all_r      = []
    
    ## Critr : If param in Dique (Cg,r)
    for prm in Param :
        r = np.sqrt(np.sum((prm - Cg)**2))
        if r <= DL_Rc :
            if len(Pred_Param)<1 :
                Pred_Param = np.array([prm]) 
                all_r      = np.array([r])
            else :
                Pred_Param = np.vstack([Pred_Param,prm])
                all_r      = np.append(all_r, r)    
    
    ## Critr : len(Pred_Param)<= DL_pred_max : 
    ## on filtre en fonction de la distance de Cg
    if len(Pred_Param) > DL_Pred_max : 
        I_sort          = np.argsort(all_r)
        Pred_Param_temp = copy.copy(Pred_Param[I_sort])
        Pred_Param      =  Pred_Param_temp[:DL_Pred_max]
    
    ## on update Data Frame :
    for prm in Pred_Param :
        code  = "df_prm.loc["
        
        for d in range(len(Param_Name)):
            code +="(df_prm[Param_Name["+str(d)+"]]==prm["+str(d)+"])"
            if d < len(Param_Name)-1 :
                code+=" & "
            else :
                code+= ", 'Prediction' ] = True"

        exec(code)
    return df_prm, Pred_Param

def Selc_Train_Data(df_prm, Param_Name, Cg, DL_use_old_data,i_main_loop, DL_Train_max):
    Param = df_prm.loc[ (df_prm["DA_Iter"] <=i_main_loop)  & ( df_prm["Prediction"]== False) ][Param_Name].values 
    all_r = []

    if not DL_use_old_data :

        Param = df_prm.loc[((df_prm["DA_Iter"] ==i_main_loop-1) | (df_prm["DA_Iter"] ==i_main_loop)) & ( df_prm["Prediction"]== False)   ][Param_Name].values 
        
   
    ## On calcul la distance de chaque prm a Cg :
    for prm in Param :
        r = np.sqrt(np.sum((prm - Cg)**2))
        if len(all_r)<1 :
            all_r      = np.array([r])
        else :
            all_r      = np.append(all_r, r) 

    ## On range Tout les param en fonction de leur distance a Cg 
    I_sort           = np.argsort(all_r)
    Param_train_temp = copy.copy(Param[I_sort])

    ## On prends les DL_Train_max premiers param
    if len(Param_train_temp) <=  DL_Train_max :
        Train_Param = Param_train_temp
    else :
        Train_Param = Param_train_temp[:DL_Train_max]
    
    ## on update Data Frame :
    for prm in Train_Param :
        code  = "df_prm.loc["
        
        for d in range(len(Param_Name)):
            code +="(df_prm[Param_Name["+str(d)+"]]==prm["+str(d)+"])"
            if d < len(Param_Name)-1 :
                code+=" & "
            else :
                code+= ", 'Training' ] = True"

        exec(code)
    return df_prm,Train_Param

def Get_Train_Pred_Ensemble(df_prm, Param_Name, DL_use_old_data, DL_Pred_max, DL_Train_max, DL_Rc, i_main_loop):
    df_prm["Training"]    = False
    df_prm["Prediction"]  = False

    DL_Train_max = int(DL_Train_max)
    DL_Pred_max  = int(DL_Pred_max)
    
    Cg           = Calc_Cg(df_prm, i_main_loop,Param_Name)
    ### -------- Select Pred Data : Use Cg and DL_Rc 
    df_prm, Pred_Param = Selc_pred_Data(df_prm, Param_Name, Cg, DL_Rc, i_main_loop, DL_Pred_max)
    
    ### -------- Select Train Data : Use Cg and DL_Train_max
    df_prm,  Train_Param = Selc_Train_Data(df_prm, Param_Name, Cg, DL_use_old_data, i_main_loop, DL_Train_max)

    ####   Data to run with HF
    HF_param = df_prm.loc[ (df_prm["DA_Iter"] ==i_main_loop)  & ( df_prm["Prediction"]== False) ][Param_Name].values 
    # return df_prm, Pred_Param, Train_Param, HF_param

    return df_prm