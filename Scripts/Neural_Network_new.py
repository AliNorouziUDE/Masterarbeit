# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 17:12:34 2024

@author: Alireza Norouzi , 3151301
"""

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import ParameterSampler , ParameterGrid
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from itertools import product
import multiprocessing
import concurrent.futures
import joblib
import numpy as np
import matplotlib.pyplot as plt
import os 
import time
import pandas as pd

'''User Guide:
    1.Define a Dictionary to initialize the Neural Network with , use Init_Dict for basic settings
    2.Define the optimization space for Hyperparameter Optimization , use the optimization_spce for basic settings
    3.Define a Neural Network Model with Neural_Network_Model(Init_Dict)
    4.Train the Nerual Network with  .train function  (model.train() ) , the model will be automatically trained and saved at save directory
    5.to perform Hyperparameter Tuning use the .search function (model.search() ) the search results and best model will be saved at save directory

    '''

# Get the current working directory
current_path = os.getcwd()

# Get the parent directory
parent_path = os.path.dirname(current_path)    
    
#Coarse to Fine Study for HPO: Step 1
optimization_space = { 'HL_space':[3,4], #number of Hidden Layers Intervall
                      'N_space':[64,128,256], #number of Neurons per Layer Intervall
                      'AF_space':['relu'], # Activation function Intervalls
                      'Batch_space':[128,256,512], #Batch Size Intervall
                      'Alpha_space':[1e-4,1e-3], # L2 Regularisation Coeffient Intervall
                      'LR_space':[1e-4], #Learning Rate Intervall
                      'sample_size':10, #number of combinations of the given Hyperparamter Intervalls.
                      'method':'randomized_search'
    }


#Define Neural Network manually to Train
Init_Dict = {'max_epochs':300, #Maximum Epoches to train
             'activation':'relu', # Activation function per Layer
             'hidden_layer': 4, #number of Hidden Layers
             'neurons' :256,  #number of Neurons per Layer
             'batch_size':256, #Batch size
             'alpha':1e-4, # L2 Regularisation Coeffient
             'lr':1e-4, #Learning Rate
             'atol':1e-6, #Convergence Tolerance for Early stopping
             'epoch_to_converge':10, # wait this number of Epoches if no improvement occurs
             'dev_set_split_ratio' : 0.2, # Split ratio for dev-Dataset to be used in Early Stopping to calculate Validation loss
             'test_set_split_ratio':0.2, # Split ratio for Test-Dataset to be hold out for Testing the Network
             'verbose':2,
             'save_directory':os.path.join(parent_path,r'NN_Results'),
             'cache_directory':os.path.join(parent_path,r'NN_Cache'), #Cache folder that used to save Checkpoints
             'dataset_directory':os.path.join(parent_path,r"Datensätze\4-Gesamtdatensatz-zum-Training\nn_Merged_Dataset.pkl"), #Dataset Address to be used for Training
             'optimization_space':optimization_space 
             
    }
#Define Type of Operation , HPO or Training by setting the boolean to True.
Perform_Hyperparameter_Tuning = False
Perform_Training = True

class Neural_Network_Model:
    def __init__(self, init_dict):
        # Extract initialization parameters from the given dictionary
        self.max_epochs = init_dict['max_epochs']
        self.activation_function = init_dict['activation']
        self.hidden_layers = init_dict['hidden_layer']
        self.neurons = init_dict['neurons']
        self.batch_size = init_dict['batch_size']
        self.alpha = init_dict['alpha']
        self.learning_rate = init_dict['lr']
        self.atol = init_dict['atol']
        self.epoch_to_converge = init_dict['epoch_to_converge']
        self.save_frequency = init_dict.get('save_frequency',10)
        self.report_frequency =init_dict.get('report_frequency',10)
        self.dev_set_split_ratio = init_dict['dev_set_split_ratio']
        self.test_set_split_ratio = init_dict['test_set_split_ratio']
        self.verbose = init_dict['verbose']
        self.save_directory = init_dict['save_directory']
        self.cache_directory = init_dict['cache_directory']
        self.dataset_directory = init_dict['dataset_directory']
        
        if 'drop_inputs' in init_dict.keys():
            self.dataset_drop_inputs_columns=init_dict['drop_inputs']
        else: self.dataset_drop_inputs_columns=[]
            
        if 'drop_targets' in init_dict.keys():
            self.dataset_drop_targets_columns=init_dict['drop_targets']
        else: self.dataset_drop_targets_columns=[]    
            
        self.optimization_space = init_dict.get('optimization_space',{ 'HL_space':[2,3,4],
                              'N_space':[64,128,256],
                              'AF_space':['relu'],
                              'Batch_space':[128,256,512],
                              'Alpha_space':[1e-4],
                              'LR_space':[1e-4],
                              'sample_size':10,
                              'method':'randomized_search'})

        #Create the directories:
        for address in [self.save_directory,self.cache_directory]:
            if not os.path.exists(address): os.makedirs(address)
        
        # Initialize attributes
        self.model = None
        self.scaler_inputs = None
        self.scaler_target = None
        self.inputs_train = None
        self.targets_train = None
        self.inputs_test = None
        self.targets_test = None
        self.loss_list = []
        self.vloss_list = []
        self.network_cache = []
        self.early_stopper_counter = 0
        self.vloss_min = 10
        self.save_name = None
        self.Grad_Width = 5
        if self.verbose > 1:
            self.sk_verbose = True
        else: self.sk_verbose = False
        
        #Initialize the Model
        self.initialize_model()

        #Setup the Dataset
        self.dataset_inputs ,self.dataset_targets = self.load_dataset(self.dataset_directory)
        self.setup_nn_datasets(self.dataset_inputs, self.dataset_targets)
        
        
    def initialize_model(self):
        '''Defines the MLPRegressor model'''
        self.model = MLPRegressor(
            hidden_layer_sizes=(self.neurons,) * self.hidden_layers,
            activation=self.activation_function,
            learning_rate_init=self.learning_rate,
            solver='adam',
            max_iter=self.max_epochs,
            random_state=42,
            batch_size=self.batch_size,
            alpha=self.alpha,
            early_stopping=False,
            tol=self.atol,
            verbose=False,
            validation_fraction=self.dev_set_split_ratio,
            n_iter_no_change=10
        )
        
    def load_dataset(self,Dataset_Address):
        '''loads the Dataset , Drops the defined columns in Init_Dict'''
        if not os.path.exists(Dataset_Address): raise NameError("Dataset Doesn't exist .")    
        else:
            Dataset = joblib.load(Dataset_Address)
            if len(self.dataset_drop_inputs_columns)>0:
                Dataset['inputs'] = Dataset['inputs'].drop(self.dataset_drop_inputs_columns,axis=1)
            if len(self.dataset_drop_targets_columns)>0:
                Dataset['targets'] = Dataset['targets'].drop(self.dataset_drop_targets_columns,axis=1)
            self.dataset=Dataset
            inputs=Dataset['inputs'].to_numpy()
            targets=Dataset['targets'].to_numpy()
            return inputs,targets

        
        
    def setup_nn_datasets(self, inputs, targets):
        # Split data into training and test sets
        self.inputs_train, self.inputs_test, self.targets_train, self.targets_test = train_test_split(
            inputs, targets, test_size=self.test_set_split_ratio, shuffle=True
        )

        # Perform Min-Max normalization on the training set
        self.scaler_inputs = MinMaxScaler()
        self.scaler_target = MinMaxScaler()

        self.inputs_train = self.scaler_inputs.fit_transform(self.inputs_train)
        self.targets_train = self.scaler_target.fit_transform(self.targets_train)

        # Perform Min-Max normalization on the test set using the training set's scaler
        self.inputs_test = self.scaler_inputs.transform(self.inputs_test)
        self.targets_test = self.scaler_target.transform(self.targets_test)
    
    def Calculate_Gradients(self,X,width):
        """
        Calculates the Gradient of Loss and Vloss to evaluate the training process in Early stopping Module

        Parameters
        ----------
        X : Numpy Array
            Calculate Gradient for this Array.
        width : int
            calculate the gradient within this epoch range

        Returns
        -------
        Y : float
            Mean Gradient of X .

        """
        if len(X) > width:
            Y=np.mean(np.gradient(X)[-width:])
        else:Y= -1.0
        return Y
    
    def generate_hidden_layer_sizes(self,HL,N):
        """
        generates hidden_layer_sizes of the MLP model based on the given 
        number of Hidden layers and Number of neurons per layer.

        Parameters
        ----------
        HL : int
            Number of Hidden Layers.
        N : int
            Number of Neurons per Layers.

        Returns
        -------
        Tuple
            Architecture Size of Network.

        """
        
        l=list()
        for i in range(HL):
            l.append(N)
        return tuple(l)
    
    def Validate_model (self,model,test_set_input,test_set_target):
         '''Calculates the RMSE of Model with Testset'''
         return mean_squared_error(model.predict(test_set_input),test_set_target)**(0.5)
    
     
    def generate_hidden_layer_sizes_set(self,neuron_list, layer_list):
        """
        Generate a set of hidden_layer_sizes based on lists of neurons and layers.

        Parameters:
        - neuron_list (list): List of numbers of neurons per layer.
        - layer_list (list): List of numbers of hidden layers.

        Returns:
        - list: List of tuples representing different hidden_layer_sizes configurations.
        """
        return [self.generate_hidden_layer_sizes(layers, neurons) for layers, neurons in product(layer_list, neuron_list)]

    def save_model(self,Save_path,scaler,scaler_Target,model,inputs_test,Targets_test,Loss_curves,time):
                Checkpoint_dict = {
                    'scaler_inputs': scaler,
                    'scaler_Target': scaler_Target,
                    'model': model,
                    'test_inputs':inputs_test,
                    'test_targets':Targets_test,
                    'loss_curve':Loss_curves['loss_curve'],
                    'vloss_curve':Loss_curves['vloss_curve'],
                    'time_elapsed':time
                }

                joblib.dump(Checkpoint_dict, Save_path)
                
    def save_name_from_model(self,model):
        '''selects a name for the trained model to be saved  based on the models hyperparameters '''
        if not isinstance(model, MLPRegressor):
            raise ValueError("The input model should be an instance of MLPRegressor.")

        # Extract model hyperparameters
        num_layers = len(model.hidden_layer_sizes)
        num_neurons = model.hidden_layer_sizes[0]
        activation_func = model.activation
        batch_size = model.batch_size
        alpha = model.alpha


        # Create save name
        save_name = f"nn_L{num_layers}_N{num_neurons}_{activation_func}_B{batch_size}_a{alpha}.pkl"

        return save_name
    
    def plot_learning_curves(self):
        '''Plots Learning Curves'''
        time = self.training_time
        
        # Plot the Learning Curves
        loss_list = self.learning_Curve['loss_curve']
        vloss_list = self.learning_Curve['vloss_curve']
        fig = plt.figure(figsize=[12, 8])
        x = np.linspace(1, len(loss_list), len(loss_list))
        
        plt.plot(x, loss_list, '-b', linewidth=3, label='Training Loss')
        plt.plot(x, vloss_list, '-r', linewidth=3, alpha=0.9, label='Validation Loss')
        plt.xlabel('Epochen', fontsize=20)
        plt.ylabel('RMSE Verlust', fontsize=20)
        plt.legend(fontsize=20)
        plt.title('Lernkurven', fontsize=24)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        #plt.yscale('log')  # Use log scale for the y-axis
        plt.tick_params(labelsize=22)
        plt.grid(visible=True, which='both', linestyle='--', linewidth=2)
        # plt.vlines(21,0.002,0.004,linestyles='--',linewidth=2)
        # Add text annotation for training time
        time_text = f'Trainingszeit: {time/60:.2f} min'+'\n$vloss_{gradient}:$'+f'{np.mean(np.gradient(vloss_list[-10:])):.2e}'
        plt.annotate(time_text, xy=(0.8, 0.7), xycoords='axes fraction',
                     ha='center', va='bottom', fontsize=20, color='black')
        for spine in plt.gca().spines.values():
            spine.set_linewidth(2.5)
            
        plt.tight_layout()
        plt.show()

        return fig
    
    
    def train_model(self,inputs_train_set, Targets_train_set, model, scaler, scaler_Target, Save_path, Grad_Width, tolerance, Converge_Wait_Epochs, max_epochs,validation_ratio):
        '''Trains a Neural Network Model using Batch Learning method
        Input Arguments:
            inputs_train: Normalized scaled Train segment of the Input Dataset.
            Targets_train: Normalized scaled Train segment of the Targets Dataset.
            scaler : Scaler that is being used to normalize the Inputs Dataset.
            scaler_Target: Scaler that is being used to normalize the Targets Dataset
            save_path : a Directory to save the training Neural Network , (this path should end with name of model and format .pkl like: Model_1.pkl )
            Epoch_to_save : save the Neural Network priodically after this number of Epochs.
            Grad_with: Calculate Gradients over these number of Epochs for Early Stopper Module
            tolerance : Converging Toleratnce for the Gradient of Validation loss (d_Vloss)
            Converge_Wait_Epochs : wait for this number of Epochs after the Converging Tolerance is met.
            max_epochs : Maximum number of training Epochs (iterations)
            
        Returns :None
        '''
        loss_list = []
        vloss_list = []
        self.network_cache = []
        Early_Stopper_Counter = 0
        vloss_min = 10
        t1=time.time()
        #Generate Dev dataset :Split the Training Dataset for calculating the validation loss
        inputs_train, inputs_Dev, Targets_train, Targets_Dev = train_test_split(inputs_train_set,Targets_train_set,test_size=validation_ratio ,shuffle=True)
        file_name = self.save_name_from_model(model) #Generate the a Unique name for network based on the networks Hyperparameters
        Save_path2 = os.path.join(Save_path, file_name)
        for epoch in range(max_epochs):
            model.partial_fit(inputs_train, Targets_train)

            Output_Train = model.predict(inputs_train)
            Output_validation = model.predict(inputs_Dev)
            #Calculate the RMSE Loss
            loss = (mean_squared_error(Output_Train, Targets_train))**(0.5)
            vloss = (mean_squared_error(Output_validation, Targets_Dev))**(0.5)
            
            loss_list.append(loss)
            vloss_list.append(vloss)
            #Calculate the Gradients

            vloss_grad = self.Calculate_Gradients(vloss_list, Grad_Width)
            
            #Earlystopper Module
            if vloss_grad > 0:
                Early_Stopper_Counter += 1
            if np.abs(vloss_grad) < tolerance and vloss_grad < 0:
                Early_Stopper_Counter += 1

            if Early_Stopper_Counter == Converge_Wait_Epochs:
                print(f'Epoch: {epoch}, Loss: {loss:.4e}, Vloss: {vloss:.4e}, D_vloss: {vloss_grad:.5e}')
                print('Training is Converged')
                break
            
            if vloss < vloss_min:
                vloss_min = vloss
                
            t2 = time.time()
            self.learning_Curve = {'vloss_curve':vloss_list , 'loss_curve':loss_list}
            self.network_cache.append(model)
            self.model=model
            #Report the state
            if epoch % self.report_frequency == 0 and self.verbose > 1:
                print(f'Epoch: {epoch}, Loss: {loss:.4e}, Vloss: {vloss:.4e}, D_vloss: {vloss_grad:.5e}')
                
            if epoch % self.save_frequency == 0:
                if vloss <= vloss_min : 

                    self.save_model(Save_path2,scaler,scaler_Target,model,self.inputs_test,self.targets_test,self.learning_Curve,t2-t1)
                
        
        #Plot the Learning Curves
        self.training_time = t2-t1
        fig=self.plot_learning_curves()
        #fig.show()
        save_fig_path = f"{Save_path.split('nn_')[0]}learning_Curve.png"
        fig.savefig(save_fig_path)
        Evaluation_RMSE = self.Validate_model(model,self.inputs_test,self.targets_test) 
        print(f'Model R2 : {model.score(self.inputs_test,self.targets_test)} \nModel RMSE :{Evaluation_RMSE}')
        
        print(f'Time Elapsed:{self.training_time}')
        
    def style_dataframe(self,df,subset):
        '''colorize the dataframe with a cmap'''
        styles = [
            dict(selector="th", props=[("border", "2px solid black")]),  # Header Borders
            dict(selector="td", props=[("border", "1px solid black")]),  # Data Borders
            dict(selector="th, td", props=[("padding", "8px")]),  # Padding for Cells
            dict(selector=".col_heading", props=[("text-align", "center")]),  # Centering Header Text
            dict(selector=".blank", props=[("border", "none")]),  # Remove border from blank cells
        ]
        
        # Apply a color scale to the mean_test_score column
        df_style = df.style.background_gradient(subset=subset, cmap='jet')
        
        # Set the column widths
        df_style = df_style.set_table_styles(styles)
        
        return df_style  
        
    def hyperparameter_tuning(self,model, inputs_train, targets_train,inputs_test,targets_test,scaler_inputs,scaler_Target ,param_grid, num_combinations,cache_path, save_path,method='randomized_search',score_method='neg_root_mean_squared_error'):
        '''
        Performs a hyperparameter optimazition on the given parameter grid using Cross Validation method to obtain the best parameter for current dataset.
        CV with Score: neg_root_mean_squared_error

        model : untrained predefined Neural Network
        inputs_train :(array) inputs of the training dataset
        targets_train :(array) targets of the training dataset
        param_grid :(dict) a dictionary that contains the space of hyperparameters parameters, like batch_size = [64, 128]
        num_combinations :(int) size of the sample that being used to get random combinations from the grid.
        save_path:(string) folder to save intermediate results
        method :(string),  randomized_search , grid_search
        '''
        t1=time.time()    
        save_param_path = os.path.join(save_path, 'best_params_RS.pkl')
        save_model_path = os.path.join(save_path, 'best_model.pkl')
        save_results_path = os.path.join(save_path, 'cv_results.xlsx')
        Dataset_path = os.path.join(cache_path, 'RS_Dataset.pkl')
        param_sampler_address = os.path.join(cache_path, 'param_sampler.pkl')
        # Create combinations manually
        
        if  os.path.exists(Dataset_path):
            inputs_train , targets_train ,  inputs_test , targets_test ,scaler_inputs,scaler_Target = joblib.load(Dataset_path)
            #np.split(inputs_train)
        else: joblib.dump([inputs_train , targets_train ,  inputs_test , targets_test ,scaler_inputs,scaler_Target ],Dataset_path)    
        
        if not os.path.exists( param_sampler_address ): 
         print(' Parameter Samples are being created : ')
         if method == 'randomized_search' :
             print('Randomized Grid Search')
             param_sampler = list(ParameterSampler(param_grid, n_iter=num_combinations, random_state=42))
         else:
             print('Grid Search')
             param_sampler = list(ParameterGrid(param_grid))
             
         joblib.dump(param_sampler, param_sampler_address)
        else :
            print(' Parameter Samples are being loaded : ')
            param_sampler = joblib.load(param_sampler_address)
        
        best_model = None
        best_score = float('inf')
        results_cv_list = []
        cv=3
        for i,params in enumerate(param_sampler):
            DF = dict() #temporary Dictionary to save Cross validation results
            cache = dict()
            print('---------------Parameter------------------------')
            print(f"Training model {i+1}/{len(param_sampler)} ")
            print(f'current Parameter :{params}')
            save_cache_path = os.path.join(cache_path, f'cache_{i}.pkl')
            if not os.path.exists(save_cache_path):
                # Update the model with new hyperparameters
                model.set_params(**params)
                # Cross Validation the Current Parameter
                cv_results = cross_validate(model, inputs_train, targets_train, cv=cv,n_jobs=-1 ,scoring=score_method,return_estimator=True)
        
                DF['HL'] = len(params['hidden_layer_sizes'])
                DF['N'] = params['hidden_layer_sizes'][0]
                DF['AF'] = params['activation']
                DF['batch_size'] = params['batch_size']
                DF['lr'] = params['learning_rate_init']
                DF['alpha'] = params['alpha']
                DF['mean_fit_time'] = np.mean(cv_results['fit_time'])
                estimators = cv_results['estimator']
                for j,model_ in enumerate(estimators):
                    outputs_test = model_.predict(inputs_test)
                    DF[f'score_cv_{j}'] = -cv_results['test_score'][j]
                    #Validate the Trained Network with Test Set
                    validation_score = abs(mean_squared_error(targets_test, outputs_test))**0.5 #Calculate RMSE Error between Test set and Predictions
                    DF[f'validation_score_cv_{j}'] = validation_score
                mean_score = -np.mean(cv_results['test_score'])
                DF['mean_score_cv'] = mean_score
                print('-----------------Score----------------------')
                print(f"Validation Score: {mean_score}")
                #save the results as caches
                cache['cv_result'] = DF
                cache['model'] = model
                joblib.dump(cache,save_cache_path)
            else:
                cache= joblib.load(save_cache_path)
                DF = cache['cv_result']
                model = cache['model']
                mean_score = DF['mean_score_cv']
            results_cv_list.append(DF)
            
            # Save the model and parameters if it's the best so far
            if mean_score <= best_score:
                best_score = mean_score
                best_model = model
                best_params = params
                print(best_params)
                joblib.dump(params,save_param_path )
        print('==================Results========================')        
        t2=time.time()
        self.search_time=t2-t1
        self.cv_results = DF
        file_name = self.save_name_from_model(best_model)
        save_path_ = os.path.join(save_path, file_name)
        if not os.path.exists(save_path_): 
            print('Training the best model ...') 
            self.train_model(inputs_train, targets_train, best_model, scaler_inputs, scaler_Target, save_path_, self.Grad_Width, self.atol, self.epoch_to_converge, self.max_epochs ,self.dev_set_split_ratio)       
        else: print('the Best model is already trained .. skip')
        #best_model.fit(inputs_train,Targets_train)
        #save_model(save_model_path,scaler_inputs,scaler_Target,best_model,inputs_test,Targets_test)
        #joblib.dump(best_model,save_model_path)


        DF = pd.DataFrame(results_cv_list)
        mean_validation_score = DF[f'validation_score_cv_{0}']*0
        for cvi in range(cv):
            mean_validation_score=mean_validation_score + DF[f'validation_score_cv_{cvi}']
            
        mean_validation_score = (mean_validation_score)/cv 
        DF['mean_validation_score'] = mean_validation_score
        DF = DF.sort_values(by='mean_score_cv')
        DF = self.style_dataframe(DF,['mean_score_cv','mean_validation_score'])
        
        DF.to_excel(save_results_path,index=False)  

        print('------------------Best Model ---------------------')        
        print(f'Best Model:{best_model} \nBest Score:{best_score}')        
        self.best_model = best_model
        self.best_score = best_score



    def train(self):
        '''Trains a Neural Network based on initiated Model and save results at save directory'''
        self.train_model(self.inputs_train, self.targets_train, self.model, self.scaler_inputs, self.scaler_target, self.save_directory , self.Grad_Width, self.atol, self.epoch_to_converge, self.max_epochs,self.dev_set_split_ratio)

    def search(self):
        '''Performs a hyperparameter optimization with Cross validation and searches for the best hyperparameters, based on parameter space and save results at save directory'''

        self.parameter_space = param_dist = {
        'hidden_layer_sizes':self.generate_hidden_layer_sizes_set(self.optimization_space['N_space'], self.optimization_space['HL_space']),
        'activation': self.optimization_space['AF_space'],
        'alpha': self.optimization_space['Alpha_space'],
        'learning_rate_init': self.optimization_space['LR_space'],
        'learning_rate':['constant'],
        'batch_size':self.optimization_space['Batch_space']
        }
        sample_size = self.optimization_space['sample_size']
        method = self.optimization_space['method']
        self.hyperparameter_tuning(self.model, self.inputs_train, self.targets_train,self.inputs_test,self.targets_test,self.scaler_inputs,self.scaler_target ,self.parameter_space, sample_size,self.cache_directory, self.save_directory,method=method,score_method='neg_root_mean_squared_error')
    def visulize_Dataset(self):
        # Plotting the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(model.dataset['targets']['h_w']/1000, bins=30, edgecolor='k')
        plt.xlabel('$h_w (KJ/kg)$')
        plt.ylabel('Frequency')
        plt.title('Distribution of Target Values')
        plt.show()
        
        # Plotting the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(model.dataset['targets']['h_s']/1000, bins=30, edgecolor='k')
        plt.xlabel('$h_s (KJ/kg)$')
        plt.ylabel('Frequency')
        plt.title('Distribution of Target Values')
        plt.show()
        
        # Plotting the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(model.dataset['targets']['p_w']/1000, bins=30, edgecolor='k')
        plt.xlabel('$p_w (kpa)$')
        plt.ylabel('Frequency')
        plt.title('Distribution of Target Values')
        plt.show()
        
        # Plotting the histogram
        plt.figure(figsize=(10, 6))
        plt.hist(model.dataset['targets']['p_s']/1000, bins=30, edgecolor='k')
        plt.xlabel('$p_s (kpa)$')
        plt.ylabel('Frequency')
        plt.title('Distribution of Target Values')
        plt.show()


model = Neural_Network_Model(Init_Dict)
if Perform_Hyperparameter_Tuning == True:
    print('Hyperparameter Tuning and Training the best Network:')
    model.search()
if Perform_Training == True:
    print('Training a Network based on the given Hyperparameters:')    
    model.train()

