
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
try:
  import cPickle as pickle
except:
  import pickle
import seaborn as sns

# Cor = False

class POSTRAIT:
    
    #-------------------------------------------------------
    #              Initialize object
    #-------------------------------------------------------
    
    def __init__(self,Path_CFG,CFG_file_name = "CONFIG.pkl") :
        self.Path_CFG        = Path_CFG
        self.CFG_file_name   = CFG_file_name

        self.load_cfg()
        self.get_jcost()
        self.get_ind_opt()
        self.get_param_opti()
        self.get_time_and_timeobs_yobs()
        self.observables_ref  = self.get_observables("ref")
        self.observables_est  = self.get_observables("est")
        self.observables_opti = self.get_observables("opti")
    #-------------------------------------------------------
    #              Read the DA Output
    #-------------------------------------------------------
    def load_cfg(self):
        file     = open(self.Path_CFG+self.CFG_file_name, "rb")
        self.CFG = pickle.load(file)
    def get_ind_opt(self):
        self.ind_opti = len(self.J_cost) -1 -1 #<= -i= 0 - J_o 
    def get_jcost(self):
        J_both = np.loadtxt(self.Path_CFG +"CostFunction"+self.CFG.ext)
        if self.CFG.DA_Version == 1 :
            self.J_cost = J_both[:,0]
        if self.CFG.DA_Version == 2 :
            self.J_cost = J_both[:,1]
            
    def get_param_opti(self):
        all_param = np.loadtxt(self.Path_CFG +"Param_ref_est_all_opti"+self.CFG.ext)
        all_param = all_param[:,1:]
        # all param opti in each iteration
        self.all_param_opti = all_param[2:,:]
        
        # the last param opti 
        self.Param_Opti    = self.all_param_opti[-1,:]
    
    def get_time_and_timeobs_yobs(self):
        self.time         = np.loadtxt(self.Path_CFG +"time"+self.CFG.ext)
        self.time_obs     = np.loadtxt(self.Path_CFG +"time_obs"+self.CFG.ext)
        self.observations = np.loadtxt(self.Path_CFG +"y_obs_ref"+self.CFG.ext)
        
    def get_observables(self, typ :str= "ref") :
        
        if typ == "ref" or typ == "est" :
            observables = np.loadtxt(self.Path_CFG +"Observables_"+typ+self.CFG.ext)
        
        elif typ == "opti":
            st = str(self.ind_opti)
            file = self.Path_CFG+self.CFG.folder_Opti+"Observablesopti_"+st+self.CFG.ext
            observables = np.loadtxt(file)
        # if Cor ==True :
        #     _,col = np.shape(observables)
            
        #     if col >2:

        #         observables = observables[:,::2]
        #         if typ == "ref" or typ == "est" :
        #             np.savetxt(self.Path_CFG +"Observables_"+typ+self.CFG.ext,observables,header="CD         CM")
        #         if typ == "opti" :

        #             np.savetxt(file,observables,header="CD         CM")
            
        return observables
    
    def get_observables_of_sample(self,N_iter, N_sample):
        st1  = str(N_iter); st2 = str(N_sample)

        try :
            file = self.Path_CFG+self.CFG.folder_Ensemble+"Iter_"+st1+"/Observablesens_"+st2+self.CFG.ext
            observables = np.loadtxt(file)
        except :
            file = self.Path_CFG+self.CFG.folder_Ensemble+"Iter_"+st1+"/Observables_ens_"+st2+self.CFG.ext
            observables = np.loadtxt(file)

        # if Cor ==True :
        #     _,col = np.shape(observables)
        #     if col >2:
        #         observables = observables[:,::2]
        #         np.savetxt(file,observables,header="CD         CM")
        return observables

    def get_samples_df(self):
            # ----- DA Paramters : 
        Nb_iter    = self.ind_opti+1
        Nb_ens     = self.CFG.DA_N_ens
        Dim_param  = self.CFG.Param_Dim
        Name_param = self.CFG.Param_Name


        # ----- Read Samples :
        All_Samples = np.zeros((Nb_iter * Nb_ens ,Dim_param))
        DA_iter = np.zeros((Nb_iter * Nb_ens), dtype=int)
        N_Sample  = np.zeros((Nb_iter * Nb_ens), dtype=int)

        for iter in range(Nb_iter):
            file        = self.Path_CFG+self.CFG.folder_Ensemble+"Iter_"+str(iter)+"/Param_ens"+self.CFG.ext
            All_Samples[iter*Nb_ens : Nb_ens*(iter+1) ,:] = np.loadtxt(file) 
            # add colum DA_iter
            DA_iter[iter*Nb_ens : Nb_ens*(iter+1) ]  = iter*np.ones(Nb_ens)
            N_Sample[iter*Nb_ens : Nb_ens*(iter+1) ]   = np.arange(Nb_ens)
        # ---- Covert to DataFram
        dict_Samples = {}
        for dim in range(Dim_param) :
            dict_Samples[Name_param[dim]] = All_Samples[:,dim]
        # add colum DA_iter
        dict_Samples["DA_Iter"] = DA_iter
        dict_Samples["Sample"]  = N_Sample

        # convert dict to DATAfram
        df_Sampls = pd.DataFrame.from_dict(dict_Samples)
        
        return df_Sampls
    #-------------------------------------------------------
    #             Visualize DA Output
    #------------------------------------------------------- 

    # ------------- Plot ------------------
    def plot_Jcost(self, label ='') : 
    
        N_iter = [i for i in range(len(self.J_cost))]
        if len(label)>0 :
            plt.plot(N_iter,np.log10(self.J_cost),'-o',color='k', label = label)
            plt.legend(fontsize=18)
        else : 
            plt.plot(N_iter,np.log10(self.J_cost),'-o',color='k')
        plt.xlabel('N_iter',fontsize=15)
        plt.ylabel('J / Jo  (Log)',fontsize=15)
        plt.grid()
    
    def plot_obs_ref_est_opt(self):
        
         
        for i,name_obs in enumerate(self.CFG.Obs_Name) :
            plt.subplot(self.CFG.Number_Observation, 1 , i+1)
            plt.plot(self.time_obs, self.observations[:,i], 'o',color='r',label='Observations')
            plt.plot(self.time,self.observables_est[:,i],'--',color='grey',label='Initialisation' ,linewidth=2)
            plt.plot(self.time,self.observables_ref[:,i],'-',color='k',label='Reference calculation',linewidth=4)
            plt.plot(self.time,self.observables_opti[:,i],color='c',label='Data Assimalation',linewidth=2 )
            plt.plot(self.time_obs, self.observations[:,i], 'o',color='r')
    
            min = np.min(self.observables_opti[:,i]) ; min=min - (min/abs(min))*min*0.2
            max = np.max(self.observables_opti[:,i]) ; max=max + (max/abs(max))*max*0.2 
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.ylabel(name_obs,fontsize=20)
            plt.ylim([min,max])
            plt.grid()
            if   i== 0:
                plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                            mode="expand", borderaxespad=0, ncol=4,fontsize=18)
        plt.xlabel('time (s)',fontsize=20)

    def plot_culster_train_pred(self, Samples_Train, Samples_Pred = []):
        
        plt.scatter(Samples_Train[:,0],Samples_Train[:,1], label = "Training",c ="royalblue" )


    # ------------------- Print ---------------  
    
    def show_param_ref_opti_est(self, PRINT = False):
        
        df_param = pd.DataFrame([self.CFG.Param_Ref, self.CFG.Param_Est, self.Param_Opti],
                                index=["Reference", "Initialisation" ,"Optimized"],
                                columns=self.CFG.Param_Name,
                                )
        if PRINT : 
            print(tabulate(df_param, tablefmt='grid',headers='keys'))
        else :
            return df_param

    def show_cfg(self):
        print_ref = ''
        for i in range(self.CFG.Param_Dim):
            print_ref += self.CFG.Param_Name[i] +' = '+str(self.CFG.Param_Ref[i])
            print_ref += '   '

        print_est = ''
        for i in range(self.CFG.Param_Dim):
            print_est += self.CFG.Param_Name[i] +' = '+str(self.CFG.Param_Est[i])
            print_est += '   '

        print_obs =''
        for i in range(len(self.CFG.Obs_Name)) :
            print_obs += self.CFG.Obs_Name[i] 
            print_obs += '  ' 

        if self.CFG.Obs_Noise :
            print_noise=''
            for i in range(len(self.CFG.Obs_Name)) :
                print_noise  += self.CFG.Obs_Name[i] + ' : ' + str(self.CFG.Obs_Sig_noise[self.CFG.Obs_Name[i]])
                print_noise  += '  ' 
        

        print('-------------------------------------------------------------------------------------------------------------------')
        print('                                                Parametres                                         ')
        print('--------------------------------------------------------------------------------------------------------------------')
        print('     Modele name      :                   '+ self.CFG.Solver_Name+'                ')
        print('     DA Method        :                   ' +self.CFG.DA_Methode+' (V'+str(self.CFG.DA_Version)+')        ')
        print('     Uncertains param :                                                       ')
        print('                  name     : '+ self.CFG.Param_Name_join                            )
        print('                 reference : '+ print_ref                                      )
        print('                 estimate  : '+ print_est                                      )    
        print('                                                                         ')
        print('     Nbr Iteration = {0:3d}  |  Nbr Ensemble = {1:4d}  | Param_sigma = {2:4f}'.format(self.CFG.DA_Max_Iter,self.CFG.DA_N_ens,self.CFG.DA_Param_Sigma))
        print('                                                                         ')
        print('     Observations     :                                                       ')
        print('               Periode = {0:}  |  time_start = {1:}  | time_end = {2:}'.format(self.CFG.Obs_Period,self.CFG.Obs_T0,self.CFG.Obs_Tf))                        
        print('               observables         : '+ print_obs       )  
        print('               Nbr observations    : '+ str(self.CFG.Number_Observation )                                    )
        print('               Noise               : '+str(self.CFG.Obs_Noise )                          )

        if self.CFG.Obs_Noise :
            print('               sig_noise           : '+str(print_noise)                          )   
        print('                                                                         ')

