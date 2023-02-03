
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate
import math
import copy
import pickle
import seaborn as sns

# Cor = False

class POSTRAIT:
    
    #-------------------------------------------------------
    #              Initialize object
    #-------------------------------------------------------
    
    def __init__(self,Path_CFG,label="EnVar(2)",color="k",CFG_file_name = "CONFIG.pkl") :
        self.Path_CFG        = Path_CFG
        self.CFG_file_name   = CFG_file_name
        self.label           = label
        self.color           = color


        self.load_cfg()
        self.get_jcost()
        self.get_ind_opt()
        self.get_param_opti()
        self.get_time_and_timeobs_yobs()
        self.Get_Param_error()
        self.get_ensembles_df()
        self.Get_HF_DL_ratio()
        # self.get_all_mu_pen()
        self.get_best_param_opti()

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
        self.Jcost_all = np.loadtxt(self.Path_CFG +"CostFunction"+self.CFG.ext)
        self.J_cost    = self.Jcost_all[:,0]

    def get_param_opti(self):
        all_param = np.loadtxt(self.Path_CFG +"Param_ref_est_all_opti"+self.CFG.ext)
        all_param = all_param[:,1:]
        # all param opti in each iteration
        self.all_param_opti = all_param[1:,:]
        
        # the last param opti 
        self.Param_Opti    = self.all_param_opti[-1,:]
    
    def get_best_param_opti(self) :
        center = np.array(self.CFG.Param_Ref)
        R_list = []
        for prm in self.all_param_opti :
            rayon  = np.sqrt( np.sum((prm-center)**2) )
            if len(R_list)<1 :
                R_list        = np.array([rayon])
            else :
                R_list = np.append(R_list, rayon)

        I_sort = np.argsort(R_list)
        self.all_best_opti = self.all_param_opti[I_sort]
        self.best_opti     = self.all_best_opti[-1]
        all_best_rayon    = R_list[I_sort]
        self.best_rayon   = all_best_rayon[-1]

    # def get_all_mu_pen(self):
    #     if self.CFG.Use_SWAG > 0 and self.CFG.Use_DL:
    #         self.all_mu = np.loadtxt(self.Path_CFG +"All_mu_PEN"+self.CFG.ext)
        
    
    def get_time_and_timeobs_yobs(self):
        self.time         = np.loadtxt(self.Path_CFG +"time"+self.CFG.ext)
        self.time_obs     = np.loadtxt(self.Path_CFG +"time_obs"+self.CFG.ext)
        self.observations = np.loadtxt(self.Path_CFG +"y_obs_ref"+self.CFG.ext).reshape(len(self.time_obs),self.CFG.Obs_Dim)
        
    def get_observables(self, typ :str= "ref") :
       
        if typ == "ref" or typ == "est" :
            observables = np.loadtxt(self.Path_CFG +"Observables_"+typ+self.CFG.ext).reshape(len(self.time),self.CFG.Obs_Dim)
        
        elif typ == "opti":
            st = str(self.ind_opti)
            file = self.Path_CFG+self.CFG.folder_Opti+"Observablesopti_"+st+self.CFG.ext
            observables = np.loadtxt(file).reshape(len(self.time),self.CFG.Obs_Dim)
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
            observables = np.loadtxt(file).reshape(len(self.time),self.CFG.Obs_Dim)
        except :
            file = self.Path_CFG+self.CFG.folder_Ensemble+"Iter_"+st1+"/Observables_ens_"+st2+self.CFG.ext
            observables = np.loadtxt(file).reshape(len(self.time),self.CFG.Obs_Dim)

        # if Cor ==True :
        #     _,col = np.shape(observables)
        #     if col >2:
        #         observables = observables[:,::2]
        #         np.savetxt(file,observables,header="CD         CM")
        return observables
    def get_DataFram(self):
                # ----------------  Save the Data Fram 
        name_file = self.CFG.Path_Output +'DataFram_Param_ens'+self.CFG.ext
        try :
            df_Param_Ens = pd.read_csv(name_file)
            return df_Param_Ens
        except :
            pass


    def get_ensembles_df(self):
            # ----- DA Paramters : 
        Nb_iter    = self.ind_opti+1
        Nb_ens     = self.CFG.DA_N_ens
        Dim_param  = self.CFG.Param_Dim
        Name_param = self.CFG.Param_Name

        if not self.CFG.Use_DL:
            self.i_start_ML = np.nan
            # ----- Read Samples :
            All_Samples = np.zeros((Nb_iter * Nb_ens ,Dim_param))
            DA_iter = np.zeros((Nb_iter * Nb_ens), dtype=int)
            N_Sample  = np.zeros((Nb_iter * Nb_ens), dtype=int)

            for iter in range(Nb_iter):
                file        = self.Path_CFG+self.CFG.folder_Ensemble+"Iter_"+str(iter)+"/Param_ens"+self.CFG.ext
                Samples_lu  = np.loadtxt(file) 
                Samples_lu  = Samples_lu.reshape(Nb_ens,Dim_param)

                All_Samples[iter*Nb_ens : Nb_ens*(iter+1) ,:] = Samples_lu
                # add colum DA_iter
                DA_iter[iter*Nb_ens : Nb_ens*(iter+1) ]  = (iter+1)*np.ones(Nb_ens)
                N_Sample[iter*Nb_ens : Nb_ens*(iter+1) ]   = np.arange(Nb_ens)
            # ---- Covert to DataFram
            dict_Ensembles = {}
            for dim in range(Dim_param) :
                dict_Ensembles[Name_param[dim]] = All_Samples[:,dim]
            # add colum DA_iter
            dict_Ensembles["DA_Iter"] = DA_iter
            dict_Ensembles["Sample"]  = N_Sample

            # convert dict to DATAfram
            df_Ensembles = pd.DataFrame.from_dict(dict_Ensembles)
            df_Ensembles["Solver"]=self.CFG.Solver_Name
        else :
            All_Samples = np.zeros((Nb_iter * Nb_ens ,Dim_param))
            DA_iter = np.zeros((Nb_iter * Nb_ens), dtype=int)
            N_Sample  = np.zeros((Nb_iter * Nb_ens), dtype=int)
            Solver    = []
            start_ML = False
            
            for iter in range(Nb_iter) :
                Path_HF_solver =  self.Path_CFG+self.CFG.folder_Ensemble+"Iter_"+str(iter)+"/HF_solver/Param_ens"+self.CFG.ext
                Param_ens_HF = np.loadtxt(Path_HF_solver) ## surement error dim 1 : taille (n,) au lieu de (n,1)
                N_ens_HF  = len(Param_ens_HF)
                N_ens_DL  = 0

                all_param_ens = Param_ens_HF.reshape(N_ens_HF,Dim_param)
                if N_ens_HF < Nb_ens :
                    if start_ML == False :
                        self.i_start_ML = iter
                        start_ML  = True
                    Path_DL_solver =  self.Path_CFG+self.CFG.folder_Ensemble+"Iter_"+str(iter)+"/DL_solver/Param_ens"+self.CFG.ext
                    Param_ens_DL = np.loadtxt(Path_DL_solver)
                        
                    if len(Param_ens_DL.shape) <2 :
                        Param_ens_DL = Param_ens_DL.reshape( (len(Param_ens_DL),1) )
                    N_ens_DL  = len(Param_ens_DL)
                    all_param_ens = np.vstack([all_param_ens,Param_ens_DL])
                
                All_Samples[iter*Nb_ens : Nb_ens*(iter+1) ,:] = all_param_ens
                DA_iter[iter*Nb_ens : Nb_ens*(iter+1) ]  = (iter+1)*np.ones(Nb_ens)
                N_Sample[iter*Nb_ens : Nb_ens*(iter+1) ]   = np.arange(Nb_ens)
                Solver+=N_ens_HF*[self.CFG.Solver_Name] + N_ens_DL*[self.CFG.DL_Model]

            # ---- Covert to DataFram
            dict_Ensembles = {}
            for dim in range(Dim_param) :
                dict_Ensembles[Name_param[dim]] = All_Samples[:,dim]
            # add colum DA_iter
            dict_Ensembles["DA_Iter"] = DA_iter
            dict_Ensembles["Sample"]  = N_Sample
            dict_Ensembles["Solver"]  = Solver
            # convert dict to DATAfram
            df_Ensembles = pd.DataFrame.from_dict(dict_Ensembles)

        self.df_Ensembles =df_Ensembles
        return df_Ensembles

    def Get_Param_error(self,typ= 'rltv'):
        if typ == "rltv" : 
            self.Param_error =100* np.abs(self.all_param_opti - self.CFG.Param_Ref) /  self.CFG.Param_Ref 
            
    def Get_HF_DL_ratio (self):
        n_iter = len(self.J_cost) -1
        N_ML = np.zeros(n_iter)
        N_HF = np.zeros(n_iter)
        for i in range(n_iter):
            N_HF[i] = self.df_Ensembles.loc[(self.df_Ensembles["DA_Iter"] == i +1) & (self.df_Ensembles["Solver"] == self.CFG.Solver_Name )].count()["Solver"]
            N_ML[i] = self.df_Ensembles.loc[(self.df_Ensembles["DA_Iter"] == i +1) & (self.df_Ensembles["Solver"] == self.CFG.DL_Model )].count()["Solver"]
        
        self.HF_ratio = N_HF * 100/N_HF[0]
        self.ML_ratio = N_ML * 100/N_HF[0]
    #-------------------------------------------------------
    #             Visualize DA Output
    #------------------------------------------------------- 

    # ------------- Plot ------------------
    def plot_Jcost(self,linestyle='-o', k = 0) : 
        J_cost = self.Jcost_all[:,k]
        J_Jo   = J_cost/J_cost[0]
        N_iter = [i for i in range(len(J_cost))]

        label = self.label
        color = self.color
        if len(label)>0 :
            plt.plot(N_iter,J_Jo, linestyle,label = label,color=color)
            plt.legend(fontsize=18)
        else : 
            plt.plot(N_iter,J_Jo,linestyle)
        new_list = range(math.floor(min(N_iter)), math.ceil(max(N_iter))+1,2)
        plt.xticks(new_list, fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel('DA-Iterations',fontsize=15)
        plt.ylabel('J / Jo ',fontsize=15)
        plt.yscale('log')
        
        plt.grid()

    def plot_all_mu(self,linestyle='-o'):
        if self.CFG.Use_SWAG > 0 and self.CFG.Use_DL :
            All_mu = self.all_mu
            N_iter = np.array([i for i in range(len(All_mu))])
            N_iter = N_iter+ self.CFG.DA_Max_Iter - len(All_mu)
            
            label = self.label
            color = self.color
            if len(label)>0 :
                plt.plot(N_iter,All_mu, linestyle,label = label,color=color)
                plt.legend(fontsize=18)
            else : 
                plt.plot(N_iter,All_mu,linestyle)
            new_list = range(math.floor(min(N_iter)), math.ceil(max(N_iter))+1,2)
            plt.xticks(new_list, fontsize=15)
            plt.yticks(fontsize=15)
            plt.xlabel('DA-Iterations',fontsize=15)
            plt.ylabel('mu',fontsize=15)
        else :
            pass
    
    def plot_Param_error(self,typ = "moy",label ='',c="k",linestyle='-o'):
        N_iter = [i for i in range(len(self.Param_error))]

        E_E0 = self.Param_error/self.Param_error[0]
        if typ =="moy":
            E_E0_moy = np.zeros(len(E_E0))
            for i in range(self.CFG.Param_Dim) :
                E_E0_moy += E_E0[:,i]
            E_E0_moy = 1/self.CFG.Param_Dim * E_E0_moy
            if len(label)>0 :
                plt.plot(N_iter,E_E0_moy, linestyle,label = label,c=c)
                plt.legend(fontsize=18)
            else : 
                plt.plot(N_iter,E_E0_moy,linestyle,c=c)
            plt.xlabel('N_iter',fontsize=15)
            plt.ylabel(r' $\gamma / \gamma_0$',fontsize=15)
            plt.yscale('log')
            plt.grid()

    def plot_HF_ratio(self,linestyle='--o') :
        n_iter = len(self.J_cost) -1
        N_iter = np.arange(n_iter)+1

        plt.plot(N_iter,self.HF_ratio,linestyle,color=self.color,label=self.label)
        # plt.scatter(N_iter,self.HF_ratio,color=self.color,s=100)

        new_list = range(math.floor(min(N_iter)), math.ceil(max(N_iter))+1,2)
        plt.xticks(new_list, fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel("DA-Iterations",fontsize = 18)
        plt.ylabel(r"$HF_{ratio}$",fontsize = 18)
        plt.grid()

    def plot_obs_ref_est_opt(self):
        
         
        for i,name_obs in enumerate(self.CFG.Obs_Name) :
            plt.subplot(self.CFG.Number_Observation, 1 , i+1)
            plt.plot(self.time_obs, self.observations[:,i], 'o',color='r',label='Observations')
            plt.plot(self.time,self.observables_est[:,i],'--',color='grey',label='Initialisation' ,linewidth=2)
            plt.plot(self.time,self.observables_ref[:,i],'-',color='k',label='Reference',linewidth=4)
            plt.plot(self.time,self.observables_opti[:,i],color='c',label=self.label,linewidth=2 )
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
    
     

    def show_Ensembles_Train_Pred(self):
        df_samples_bar  = copy.copy(self.df_Ensembles)
        df_Ensembles    = copy.copy(self.df_Ensembles)

        for i,parm_nm in enumerate(self.CFG.Param_Name): 
            df_samples_bar[parm_nm] =( 2*df_Ensembles[parm_nm] - self.CFG.Param_Bounds[i][0] -self.CFG.Param_Bounds[i][1])\
                            / (self.CFG.Param_Bounds[i][1]-self.CFG.Param_Bounds[i][0])


        with sns.axes_style("darkgrid"):
            g=sns.FacetGrid(df_samples_bar, col="DA_Iter", height=2.5, col_wrap=3)
            g.map_dataframe(sns.scatterplot, x=self.CFG.Param_Name[0], y=self.CFG.Param_Name[1],style ="Solver",palette="coolwarm")


            
            k = 1
            if self.CFG.Use_DL :
                i_start_train = self.i_start_ML -1
            for ax in g.axes.flatten() :
                
                ## ---- all old samples :
                if k>1 :
                    df_samples_old_bar = df_samples_bar.loc[ (df_samples_bar["DA_Iter"] <k)].copy()
                else :
                    df_samples_old_bar = df_samples_bar.loc[ (df_samples_bar["DA_Iter"] ==k)].copy()
                sns.scatterplot(data=df_samples_old_bar, x=self.CFG.Param_Name[0], ax=ax,y=self.CFG.Param_Name[1],style="Solver", alpha=0.1,color ="grey")
                

                ## ----- new samples (Current)
                df_samples_new_bar = df_samples_bar.loc[ (df_samples_bar["DA_Iter"] ==k)].copy()
                
                ## update i start train 
                if self.CFG.DL_Model not in df_samples_new_bar["Solver"].values :
                    i_start_train = k
                ## ---- Current train sample  
                if k>= self.i_start_ML and self.CFG.Use_DL:
                    df_sample_train_bar = df_samples_bar.loc[ (df_samples_bar["DA_Iter"] >=i_start_train) & (df_samples_bar["DA_Iter"] <k)].copy()   
                    sns.scatterplot(data=df_sample_train_bar, x=self.CFG.Param_Name[0], ax=ax,y=self.CFG.Param_Name[1],style="Solver", alpha=0.3,color="g")
                
                sns.scatterplot(data=df_samples_new_bar, x=self.CFG.Param_Name[0], ax=ax,y=self.CFG.Param_Name[1],style="Solver", alpha=1.,color="darkorange")
                
                ## ----plot the prediction zone 
                if k>= self.i_start_ML and self.CFG.Use_DL:
                    # compute center 
                    Samples_train_bar = df_sample_train_bar[self.CFG.Param_Name].values
                    if len(Samples_train_bar)>0:
                        center = np.zeros(self.CFG.Param_Dim)
                        for prm in Samples_train_bar :
                            center+=prm
                        center = (1/len(Samples_train_bar))*center
                        # rayon
                        r =  self.CFG.Rayon_Cluster

                        theta = np.linspace(0, 2*np.pi, 100)
                        x1 = r*np.cos(theta) + center[0]
                        x2 = r*np.sin(theta) + center[1]
                        ax.plot(x1,x2)
                        ax.scatter([center[0]],[center[1]],marker = "s",color="k",s=1)
                
                k+=1
                ax.legend().remove() 
            # g.axes.flatten()[:].legend().remove()    
            g.fig.set_dpi(120)
            g.add_legend()
            
            # fix the legends
    # ------------------- Print ---------------  
    
    def show_param_ref_opti_est(self, PRINT = False):
        
        df_param = pd.DataFrame([self.CFG.Param_Ref, self.CFG.Param_Est, self.Param_Opti],
                                index=["Reference", "Initialisation" ,"Optimized"],
                                columns=self.CFG.Param_Name,
                                )
        if PRINT : 
            print(tabulate(df_param.T, tablefmt='grid',headers='keys'))
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

# ----------------------------------- Other utils ---------------------------------

    def plot_train_loss(self,i_start = None, i_end= None, figsize = (20,4), n_col = 3):
        plt.figure(figsize=figsize)
        if i_start == None :
            i_start = self.i_start_ML
        if  i_end == None :
            i_end = len(self.J_cost)-1
        k =1
        for iter in range(i_start, i_end) :
            plt.subplot(1,n_col, k)

            prfx = "_iter_"+str(iter)
            folder_Iter =self.CFG.Path_Ensemble+'Iter_'+str(iter)+'/' 
            
            J_loss  = np.loadtxt(folder_Iter+"loss_training"+prfx+self.CFG.ext)
            N_epochs = np.arange(len(J_loss))
            
            plt.plot(N_epochs, J_loss,'--', color ="k", label=" DA-Iter= "+str(iter))
            if self.CFG.Use_SWAG > 0 :
                plt.axvline(self.CFG.N_SWAG, label= "N_SWAG", color="r")
                plt.axvline(self.CFG.L_SWAG, label= "L_SWAG", color="b")
            
            plt.xlabel('N_epoch',fontsize = 18)
            plt.ylabel('MSE',fontsize = 18)
            plt.grid()
            
            plt.legend(bbox_to_anchor=(0,1.02,1,0.2), loc="lower left",
                        mode="expand", borderaxespad=0, ncol=3,fontsize=16)  
            if k ==n_col :
                plt.show()
                plt.figure(figsize=figsize)
                k=1
            else :
                k+=1



def Get_FNN_Samples(Param_old_bar, Param_new_bar, N_cluster_sample, rayon_cluster,DL_max_ratio):
    
    HF_param_bar = []; FNN_param_bar =[];  R_list = []
    N_param = len(Param_new_bar)
    
    DL_N_param_max = int( N_param * DL_max_ratio)
    
    for prm_new in Param_new_bar :
        n_cluster = 0
        rmin_lst  = []
        for prm_old in Param_old_bar :
            
            rayon  = np.sqrt( np.sum((prm_new-prm_old)**2) )
            if rayon <= rayon_cluster :
                n_cluster +=1
                rmin_lst = np.append(rmin_lst,rayon)
        
        
        if n_cluster >= N_cluster_sample :
            if len(FNN_param_bar)<1 :
                FNN_param_bar = np.array([prm_new]) 
                r_min   = np.min(rmin_lst)
                R_list = np.array([r_min])
            else :
                FNN_param_bar = np.vstack([FNN_param_bar,prm_new])
                r_min   = np.min(rmin_lst)
                R_list = np.append(R_list, r_min)
        else :
            if len(HF_param_bar)<1 :
                HF_param_bar = np.array([prm_new])  
            else :
                HF_param_bar = np.vstack([HF_param_bar,prm_new])

    if len(FNN_param_bar) >=1 :
        if len(FNN_param_bar) > DL_N_param_max :
            I_sort = np.argsort(R_list)
            FNN_param_bar_temp = copy.copy(FNN_param_bar[I_sort])
            FNN_param_bar      =  FNN_param_bar_temp[:DL_N_param_max]

            if len(HF_param_bar) >0 :
                HF_param_bar       = np.vstack([HF_param_bar,FNN_param_bar_temp[DL_N_param_max:]])
            else : 
                HF_param_bar =FNN_param_bar_temp[DL_N_param_max:]
    return HF_param_bar, FNN_param_bar

def Compute_Bary(Param) :
    bary = np.zeros(Param.shape[1])
    for prm in Param :
        bary += prm
    bary = (1/len(Param))*bary
    return bary

def Get_FNN_Samples_v2(Param_old_bar, Param_new_bar, rayon_cluster,DL_max_ratio):
    
    HF_param_bar = []; FNN_param_bar =[];  R_list = []
    N_param = len(Param_new_bar)
    
    DL_N_param_max = int( N_param * DL_max_ratio)
    
    center = Compute_Bary(Param_old_bar)
    for prm_new in Param_new_bar :
        rayon  = np.sqrt( np.sum((prm_new-center)**2) )
        
        if rayon <= rayon_cluster :
            if len(FNN_param_bar)<1 :
                FNN_param_bar = np.array([prm_new]) 
                R_list        = np.array([rayon])
            else :
                FNN_param_bar = np.vstack([FNN_param_bar,prm_new])
                R_list = np.append(R_list, rayon)
        else :
            if len(HF_param_bar)<1 :
                HF_param_bar = np.array([prm_new])  
            else :
                HF_param_bar = np.vstack([HF_param_bar,prm_new])

    if len(FNN_param_bar) >=1 :
        if len(FNN_param_bar) > DL_N_param_max :
            I_sort = np.argsort(R_list)
            FNN_param_bar_temp = copy.copy(FNN_param_bar[I_sort])
            FNN_param_bar      =  FNN_param_bar_temp[:DL_N_param_max]

            if len(HF_param_bar) >0 :
                HF_param_bar       = np.vstack([HF_param_bar,FNN_param_bar_temp[DL_N_param_max:]])
            else : 
                HF_param_bar =FNN_param_bar_temp[DL_N_param_max:]
    return HF_param_bar, FNN_param_bar

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

def All_Data_Set(PST,col_wrap=4, Zoom = False, dim_bar = True) : 
    df_samples = PST.get_DataFram()
    if dim_bar :
        df_samples =Calc_Param_bar(df_samples, PST.CFG)

    if PST.CFG.Use_DL == True :  
        with sns.axes_style("darkgrid"):
            g=sns.FacetGrid(df_samples, col="DA_Iter", height=2.5, col_wrap=col_wrap)
            g.map_dataframe(sns.scatterplot, x=PST.CFG.Param_Name[0], y=PST.CFG.Param_Name[1],style ="Solver",palette="coolwarm")



            k = 0
        
            i_start_train = PST.i_start_ML -1
            for ax in g.axes.flatten() :

                ## ---- all old samples :
                if k>0 :
                    if Zoom and k > PST.i_start_ML+2:
                        df_samples_old = df_samples.loc[ (df_samples["DA_Iter"] <k) & (df_samples["DA_Iter"] >=PST.i_start_ML)].copy()
                    else :
                        df_samples_old = df_samples.loc[ (df_samples["DA_Iter"] <k)].copy()
                else :
                    df_samples_old = df_samples.loc[ (df_samples["DA_Iter"] ==k)].copy()
                sns.scatterplot(data=df_samples_old, x=PST.CFG.Param_Name[0], ax=ax,y=PST.CFG.Param_Name[1],style="Solver", alpha=0.1,color ="grey")


                ## ----- new samples (Current)
                
                df_samples_new = df_samples.loc[ (df_samples["DA_Iter"] ==k)].copy()

                ## update i start train 
                if PST.CFG.DL_Model not in df_samples_new["Solver"].values :
                    i_start_train = k
                ## ---- Current train sample  
                if k>= PST.i_start_ML :
                    name_file =PST.CFG.Path_Ensemble+'Iter_'+str(k)+'/' +"DataFram_Param_ens_bar"+PST.CFG.ext
                    df_Param_Ens = pd.read_csv(name_file)
                    if not dim_bar :
                        df_Param_Ens     = Calc_Param(df_Param_Ens,PST.CFG)
                    df_sample_train  = df_Param_Ens.loc[df_Param_Ens["Training"]].copy()   
                    sns.scatterplot(data=df_sample_train, x=PST.CFG.Param_Name[0], ax=ax,y=PST.CFG.Param_Name[1],style="Solver", alpha=0.8,color="g")

                sns.scatterplot(data=df_samples_new, x=PST.CFG.Param_Name[0], ax=ax,y=PST.CFG.Param_Name[1],style="Solver", alpha=1.,color="darkorange")



                k+=1
                ax.legend().remove() 
        # g.axes.flatten()[:].legend().remove()    
        g.fig.set_dpi(120)
        g.add_legend()
        
        # fix the legends
    else :
        with sns.axes_style("darkgrid"):
            g=sns.FacetGrid(df_samples, col="DA_Iter", height=2.5, col_wrap=col_wrap)
            g.map_dataframe(sns.scatterplot, x=PST.CFG.Param_Name[0], y=PST.CFG.Param_Name[1],style ="Solver",palette="coolwarm")



            k = 0
            for ax in g.axes.flatten() :

                ## ---- all old samples :
                if k>0 :
                    if Zoom :
                        df_samples_old = df_samples.loc[ (df_samples["DA_Iter"] <k) & (df_samples["DA_Iter"] >=PST.i_start_ML)].copy()
                    else :
                        df_samples_old = df_samples.loc[ (df_samples["DA_Iter"] <k)].copy()
                else :
                    df_samples_old = df_samples.loc[ (df_samples["DA_Iter"] ==k)].copy()
                sns.scatterplot(data=df_samples_old, x=PST.CFG.Param_Name[0], ax=ax,y=PST.CFG.Param_Name[1],style="Solver", alpha=0.1,color ="grey")


                ## ----- new samples (Current)
                
                df_samples_new = df_samples.loc[ (df_samples["DA_Iter"] ==k)].copy()
                sns.scatterplot(data=df_samples_new, x=PST.CFG.Param_Name[0], ax=ax,y=PST.CFG.Param_Name[1],style="Solver", alpha=1.,color="darkorange")



                k+=1
                ax.legend().remove() 
        # g.axes.flatten()[:].legend().remove()    
        g.fig.set_dpi(120)
        g.add_legend()
        
        # fix the legends