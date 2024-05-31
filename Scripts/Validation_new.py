# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 20:57:54 2024

@author: Alireza Norouzi , 3151301
"""

import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error , r2_score
import sklearn.metrics as sm
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import time
import os
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
import re
import Create_new #Create Sript from Masterarbeit scripts
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from sklearn.inspection import permutation_importance
from SALib.sample import sobol
# Global Settings
os.environ['RPPREFIX'] = r'C:/Program Files (x86)/REFPROP'
RP = REFPROPFunctionLibrary(os.environ['RPPREFIX'])
RP.SETPATHdll(os.environ['RPPREFIX'])
MASS_BASE_SI = RP.GETENUMdll(0, "MASS BASE SI").iEnum
SI = RP.GETENUMdll(0, "SI").iEnum
path=os.getcwd() #Current Path
parent_path = os.path.dirname(path) #Parent path

#Wähle die Art von Analyse und start das Programm.
Feature_Importance_Analysis = False # wenn aktiviert , wird Feature importance analyse durchgeführt
Plot_Curve_Diagrams = False #wenn , aktiviert , wird Interpolation und extrapolation durchgeführt
Size_Analysis = False #wenn , aktiviert , wird Analyse auf Datensatzgröße durchgeführt
Perform_Validation = False # Wenn aktiviert , wird das modell mit testdatensatz validiert 
Plot_Dataset_Distribution = False  # Wenn aktiviert wird die Datensatverteilung dargestellt
TSQ_index = 10 #Anzahl der Datenpunkte zur Berechnung der Temperatur und Entropie mit Refprop

#Adresse feststellen:
Dataset_Address=os.path.join(parent_path,r"Datensätze\4-Gesamtdatensatz-zum-Training\nn_Merged_Dataset.pkl") #Das ist die Adresse von Gesamtdatensatz (pkl datei) 
MLP_Modell_Address=os.path.join(parent_path,r"Beste Modelle\x_c_in_Features\nn_L4_N256_relu_B256_a1e-4.pkl") # Das ist die Adresse von trainierte MLP Model (pkl datei)

Dataset_Size_Analysis_Address=os.path.join(parent_path,r"Zwischen_Ergebnisse_Network_mit_Xc\Datengröße-Analyse")

Init_Dict2={'Network_Address':MLP_Modell_Address,
           'test_Dataset_Address' :'validation',
            'AF': 'Propane*Isobutane*Pentane',
            'SF': 'water',
            'save_Address': os.path.join(parent_path,r'Masterarbeit\Results\Validation')
    }



class Network_Postprocessing():
      def __init__(self,Init_Dict):
          self.Network_Address = Init_Dict['Network_Address'] 
          self.Test_Address = Init_Dict['test_Dataset_Address']
          self.loaded_file = joblib.load(self.Network_Address)
          self.model = self.loaded_file ['model']
          self.scaler_inputs = self.loaded_file['scaler_inputs']
          self.scaler_targets = self.loaded_file['scaler_Target']
          self.AF = Init_Dict['AF']
          self.SF = Init_Dict['SF']
          
          self.column_targets = ['h_w', 'p_w', 'h_s', 'p_s']
          #Setup normalized Validation Dataset   
          self.validation_dataset_targets_nn = self.loaded_file['test_targets']
          self.validation_dataset_inputs_nn = self.loaded_file['test_inputs']
          
          if self.validation_dataset_inputs_nn.shape[1] == 12:
              self.columns_inputs = ['dx', 'h_w', 'p_w', 'm_w', 'h_s', 'p_s', 'm_s', 'X_A', 'X_B', 'X_C',
                     'D_in', 'D_out']
          else:
              self.columns_inputs = ['dx', 'h_w', 'p_w', 'm_w', 'h_s', 'p_s', 'm_s', 'X_A', 'X_B',
                     'D_in', 'D_out']
              
          #inverse transform the Validation Dataset
          revert_validation_targets = self.scaler_targets.inverse_transform(self.validation_dataset_targets_nn)
          revert_validation_inputs = self.scaler_inputs.inverse_transform(self.validation_dataset_inputs_nn)
          #add the validation Dataset in Dataframe:
          self.validation_dataset_targets = pd.DataFrame(revert_validation_targets,columns = self.column_targets )
          print(f'DataTarget:{self.validation_dataset_targets_nn.shape[1]}')
          print(f'DataTarget:{self.validation_dataset_inputs_nn.shape[1]}')
          self.validation_dataset_inputs = pd.DataFrame(revert_validation_inputs,columns = self.columns_inputs)
          #setup the Test Dataset if it's existing
          if os.path.exists(self.Test_Address):    
              self.test_dataset = joblib.load(self.Test_Address)
              self.test_dataset_inputs = self.test_dataset['inputs']
              self.test_dataset_targets =self.test_dataset['targets']
              self.test_dataset_inputs_nn = self.scaler_inputs.transform(self.test_dataset_inputs)
              self.test_dataset_targets_nn = self.scaler_targets.transform(self.test_dataset_targets)
          else: self.test_dataset = None
          
      def Calculate_Tsq_from_ph(self, Inputs, Targets, Outputs=None):
        '''Calculate Temperatures, specific Entropy, and Quality of Working fluid and Secondary fluid 
        for Dataset.
        
        Parameters:
        --------    
        - Inputs: Inputs dataframe
        - Targets: Targets dataframe
        - Outputs (optional): Outputs dataframe
        --------
        Returns: Dataframes [T_w, s_w, q_w, T_s, s_s, q_s] 
        - Inputs TSQ
        - Targets TSQ
        - Outputs TSQ (optional)
        '''
        n = Inputs.shape[0]
        
        # Load all Samples
        if 'X_C' in Inputs.columns:
            z_w = Inputs[['X_A', 'X_B', 'X_C']].to_numpy()
        else:
            X_C = 1 - (Inputs['X_A'] + Inputs['X_B'])
            z_w = pd.concat([Inputs[['X_A', 'X_B']], X_C.rename('X_C')], axis=1).to_numpy()
    
        Inputs_PH = Inputs[['h_w', 'h_s', 'p_w', 'p_s']].to_numpy()
        Targets_PH = Targets[['h_w', 'h_s', 'p_w', 'p_s']].to_numpy()
    
        Inputs_TSQ_Matrix = np.zeros((n, 6))
        Targets_TSQ_Matrix = np.zeros((n, 6))
    
        if Outputs is not None:
            Outputs_PH = Outputs[['h_w', 'h_s', 'p_w', 'p_s']].to_numpy()
            Outputs_TSQ_Matrix = np.zeros((n, 6))
    
        AF = self.AF
        SF = self.SF
        z_s = [1.0]
    
        for i in range(n):
            # Calculate Temperature, Entropy, Quality for inputs
            rp_inputs_af = RP.REFPROPdll(
                AF, "PH", "T;s;q,PHASE", MASS_BASE_SI, 0, 0, Inputs_PH[i, 2], Inputs_PH[i, 0], z_w[i])
            Inputs_TSQ_Matrix[i, 0:3] = rp_inputs_af.Output[0:3]
    
            rp_inputs_sf = RP.REFPROPdll(
                SF, "PH", "T;s;q", MASS_BASE_SI, 0, 0, Inputs_PH[i, 3], Inputs_PH[i, 1], z_s)
            Inputs_TSQ_Matrix[i, 3:] = rp_inputs_sf.Output[0:3]
    
            # Calculate Temperature, Entropy, Quality for targets
            rp_targets_af = RP.REFPROPdll(
                AF, "PH", "T;s;q", MASS_BASE_SI, 0, 0, Targets_PH[i, 2], Targets_PH[i, 0], z_w[i])
            Targets_TSQ_Matrix[i, 0:3] = rp_targets_af.Output[0:3]
    
            rp_targets_sf = RP.REFPROPdll(
                SF, "PH", "T;s;q", MASS_BASE_SI, 0, 0, Targets_PH[i, 3], Targets_PH[i, 1], z_s)
            Targets_TSQ_Matrix[i, 3:] = rp_targets_sf.Output[0:3]
    
            if Outputs is not None:
                # Calculate Temperature, Entropy, Quality for outputs
                rp_outputs_af = RP.REFPROPdll(
                    AF, "PH", "T;s;q", MASS_BASE_SI, 0, 0, Outputs_PH[i, 2], Outputs_PH[i, 0], z_w[i])
                Outputs_TSQ_Matrix[i, 0:3] = rp_outputs_af.Output[0:3]
    
                rp_outputs_sf = RP.REFPROPdll(
                    SF, "PH", "T;s;q", MASS_BASE_SI, 0, 0, Outputs_PH[i, 3], Outputs_PH[i, 1], z_s)
                Outputs_TSQ_Matrix[i, 3:] = rp_outputs_sf.Output[0:3]
    
            print(f'calculated TSQ: {i + 1}/{n}')
    
        DF_Inputs = pd.DataFrame(Inputs_TSQ_Matrix, columns=['T_w', 's_w', 'q_w', 'T_s', 's_s', 'q_s'])
        DF_Targets = pd.DataFrame(Targets_TSQ_Matrix, columns=['T_w', 's_w', 'q_w', 'T_s', 's_s', 'q_s'])
        
        if Outputs is not None:
            DF_Outputs = pd.DataFrame(Outputs_TSQ_Matrix, columns=['T_w', 's_w', 'q_w', 'T_s', 's_s', 'q_s'])
            return DF_Inputs, DF_Targets, DF_Outputs
        else:
            return DF_Inputs, DF_Targets
                
        
      def predict(self,Dataset):
          if type(Dataset) == pd.core.frame.DataFrame:
              Dataset=Dataset.to_numpy()
          output = self.model.predict(Dataset)    
          return self.scaler_targets.inverse_transform(output)
      
      def calculate_scores_(self,output,targets):
            mse = np.mean((targets-output)**2 ,axis =0 ) 
            mae = np.mean(np.abs(targets - output) ,axis =0)
            mre = np.mean((mae/targets)*100 ,axis = 0)
            rmse = (mse)**0.5
            return {'MSE':mse , 'MAE':mae , 'rel%':mre , 'RMSE':rmse }
        
      def Plot_Scatter_XY(targets, outputs, xlabel, ylabel, title, legend_labels=None, figure_size=(10, 6), save_path=None):
          """
          Generate a scatter plot with a rotated histogram.

          Parameters:
          - targets: Array of target values.
          - outputs: Array of predicted values.
          - xlabel: Label for the x-axis.
          - ylabel: Label for the y-axis.
          - title: Title of the plot.
          - legend_labels: Labels for the legend (if applicable).
          - figure_size: Tuple specifying the figure size.
          - save_path: Path to save the plot (optional).

          Returns:
          - None (displays the plot).
          """
          # Create a 2x2 grid layout
          fig, axs = plt.subplots(1, 2, figsize=figure_size, gridspec_kw={'width_ratios': [10, 1]})
          relative_error = np.abs(((outputs - targets) / (targets ))*100)
          # Scatter plot
          axs[0].plot([np.min(targets), np.max(targets)], [np.min(targets), np.max(targets)], color='black', linestyle='-', linewidth=1)
          scatter = axs[0].scatter(outputs, targets, c=relative_error, cmap='jet', alpha=0.8, edgecolors='none', linewidths=0.25)
          axs[0].set_xlabel(xlabel, fontsize=18)
          axs[0].set_ylabel(ylabel, fontsize=18)
          axs[0].tick_params(axis='both', which='major', labelsize=14)
          #plt.gca().yaxis.set_major_formatter(PercentFormatter(1))

          axs[0].grid(True)
          # Rotated histogram on the right side
          axs[1].hist(relative_error, bins=100, density=True, log=False, orientation='horizontal', color='lightblue', edgecolor='black')
          axs[1].set_xlabel('Percentage', fontsize=12)
          axs[1].set_ylabel('Relative Error %', fontsize=16)
          axs[1].tick_params(axis='both', which='major', labelsize=12)
          axs[1].yaxis.set_major_formatter(PercentFormatter(1))
          # Increase the number of ticks for the histogram's x-axis
          #axs[1].locator_params(axis='x', nbins=5)

          # Add color bar
          cbar = plt.colorbar(scatter, ax=axs[0])
          cbar.ax.tick_params(labelsize=12)

          # Adjust layout
          plt.tight_layout(pad=1.5)
          file_name = title.replace(' ', '_')
          # Save or show the plot
          if save_path:
              #plt.savefig(f'{save_path}/Scatter_Histogram_{title.replace(" ", "_")}.png', bbox_inches='tight')
              plt.savefig(f'{save_path}/Scatter_{file_name}.png', bbox_inches='tight')
          else:
              plt.show()
              
      def cross_validation(self,Inputs,Targets,Outputs,CV=5):
            t0 = time.time()
            Inputs_list = np.split(Inputs.to_numpy()[:-1,:],CV)
            Targets_list = np.split(Targets.to_numpy()[:-1,:],CV)
            Outputs_list = np.split(Outputs.to_numpy()[:-1,:],CV)
            for i in range(5):
                 Inputs = pd.DataFrame( Inputs_list[i] , columns= self.columns_inputs)
                 Targets = pd.DataFrame(Targets_list[i] ,columns= self.column_targets)
                 Outputs = pd.DataFrame(Outputs_list[i] ,  columns= self.column_targets)
                 TSQ_inputs , TSQ_targets ,TSQ_outputs = self.Calculate_Tsq_from_ph(Inputs,Targets,Outputs)
                 t1 = time.time()
                 joblib.dump({'TSQ_Inputs':TSQ_inputs ,'TSQ_Targets':TSQ_targets ,'TSQ_Outputs':TSQ_outputs},'TSQ_CV_{i}.pkl')

def load_data_and_predict(filename, Model_Validation):
    # Extract constant values from the filename
    pattern = r"P_w\[(\d+\.\d+)\]_ms\[(\d+\.\d+)\]_mr\[(\d+\.\d+)\]_xA\[(\d+\.\d+)\]_xB\[(\d+\.\d+)\]_Di\[(\d+\.\d+)\]_Da\[(\d+\.\d+)\]"
    match = re.search(pattern, filename)
    if match:
        Test_P_w, Test_M_s, Test_M_r, Test_XA, Test_XB, Test_Di, Test_Da = map(float, match.groups())
    else:
        raise ValueError("Filename format does not match expected pattern.")

    # Load the dataset
    Test_Dataset = np.load(filename)

    # Calculate derived values
    Test_M_w = Test_M_s / Test_M_r
    Test_XC = 1 - (Test_XA + Test_XB)

    # Create columns for input data
    test_m_s_col = np.full_like(Test_Dataset[:,0], Test_M_s)
    test_m_w_col = np.full_like(Test_Dataset[:,0], Test_M_w)
    test_X_A_col = np.full_like(Test_Dataset[:,0], Test_XA)
    test_X_B_col = np.full_like(Test_Dataset[:,0], Test_XB)
    test_X_C_col = np.full_like(Test_Dataset[:,0], Test_XC)
    test_Di_col = np.full_like(Test_Dataset[:,0], Test_Di)
    test_Da_col = np.full_like(Test_Dataset[:,0], Test_Da)
    test_h_w_col = np.full_like(Test_Dataset[:,0], Test_Dataset[0,1])
    test_h_s_col = np.full_like(Test_Dataset[:,0], Test_Dataset[0,2])
    test_p_w_col = np.full_like(Test_Dataset[:,0], Test_Dataset[0,3])
    test_p_s_col = np.full_like(Test_Dataset[:,0], Test_Dataset[0,4])

    # Stack the input data
    test_input = np.vstack([Test_Dataset[:,0], test_h_w_col, test_p_w_col, test_m_w_col, test_h_s_col, test_p_s_col,
                            test_m_s_col, test_X_A_col, test_X_B_col, test_X_C_col, test_Di_col, test_Da_col]).T

    # Predict using the model
    Test_Dataset_Output = Model_Validation.predict(Model_Validation.scaler_inputs.transform(test_input))

    return Test_Dataset,Test_Dataset_Output


def Plot_Scatter_XY(targets, outputs, xlabel, ylabel, title,unit='', legend_labels=None, figure_size=(15, 12), save_path=None ,dpi=300,log=False,error_type='rel',zoom=True,Textbox_xy=[0.05, 0.95],alpha=[0,0,0,0],xlasthist=100,colorbar_enabled=True):
    """
    Generate a scatter plot with a rotated histogram and an inset zoomed region.

    Parameters:
    - targets: Array of target values.
    - outputs: Array of predicted values.
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - title: Title of the plot.
    - legend_labels: Labels for the legend (if applicable).
    - figure_size: Tuple specifying the figure size.
    - save_path: Path to save the plot (optional).

    Returns:
    - None (displays the plot).
    """
    base_fontsize= 26
    point_size = 20
    colormap = 'jet'
    # Create a 2x2 grid layout
    fig, axs = plt.subplots(1, 2, figsize=figure_size, gridspec_kw={'width_ratios': [10, 1]})
    relative_error = (np.abs(((outputs - targets) / (targets)) * 100))
    RMSE=np.sqrt(np.mean((outputs - targets)**2))
    MAE=np.mean(np.abs(outputs - targets))
    global mre, mmre
    mre = np.mean(relative_error)
    r2 = abs(r2_score(targets, outputs))
    if r2>1:
        r2 = abs(r2_score(outputs,targets))
    if mre > 100:
       rmse_text = f'RMSE: {RMSE:.2f} {unit}\nMAE: {MAE:.2f} {unit}\n$R^{2}$: {r2:.4f}'     
    else:     rmse_text = f'RMSE: {RMSE:.2f} {unit}\nMAE: {MAE:.2f} {unit}\nMRE: {mre:.2f}%\n$R^{2}$: {r2:.4f}'    
    # Display RMSE value at the top-left part of the diagram

    axs[0].text(Textbox_xy[0], Textbox_xy[1], rmse_text, transform=axs[0].transAxes, fontsize=base_fontsize, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    
    if error_type =='rel':
        hist_score = relative_error
        score_label = 'relativer Fehler %'
    elif error_type =='abs':
        hist_score = np.abs(outputs-targets)
        score_label = 'absoluter Fehler'
        
    # Scatter plot
    axs[0].set_title(title+unit, fontsize=base_fontsize+2)
    axs[0].grid(True,which='both',linewidth=2)
    axs[0].set_axisbelow(True)
    axs[0].axline(xy1=(np.min(targets), np.min(targets)),xy2=(np.max(targets), np.max(targets)), color='black', linestyle='-', linewidth=2,alpha=0.5)
    scatter = axs[0].scatter(targets,outputs, c=hist_score, cmap='jet', alpha=0.8, edgecolors='none', linewidths=2, s=point_size)  # Adjust 's' for smaller points
    if log == True:
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        
    axs[0].set_xlabel(xlabel, fontsize=base_fontsize)
    axs[0].set_ylabel(ylabel, fontsize=base_fontsize)
    axs[0].tick_params(axis='both', which='major', labelsize=base_fontsize-2,width=3,length=8,pad=5)

    if zoom == True:    
        # Inset axes for zoomed region
        ite = np.argmax(hist_score)
        x1 = targets[ite]+alpha[0]
        x2 = targets[ite]+alpha[1]
        y1 = outputs[ite]+alpha[2]
        y2 = outputs[ite]+alpha[3]
        axins = inset_axes(axs[0], width="40%", height="30%", loc='lower right',borderpad=5)
        axins.scatter( targets , outputs, c=hist_score, cmap=colormap, alpha=1, edgecolors='none', linewidths=3, s=point_size+0.35*point_size)
        axins.set_xlim([x1, x2])
        axins.set_ylim([y1, y2])
        axins.tick_params(labelsize=16,axis='both', which='major')
    
        axins.grid(True)
        axs[0].indicate_inset_zoom(axins, edgecolor="black")
        mark_inset(axs[0], axins, loc1=1, loc2=3, fc="none", ec="0.1" ,linewidth =0.2)
        #fontweight='bold'
        axins.axline(xy1=(np.min(targets), np.min(targets)),xy2=(np.max(targets), np.max(targets)), color='black', linestyle='-', linewidth=1,alpha=0.5)
    # Histogram on the right side with color-coded bars
    N, bins, patches = axs[1].hist(hist_score, bins=50, density=False,weights=100*np.ones(hist_score.shape[0])/hist_score.shape[0] , log=False, orientation='horizontal', color='lightblue', edgecolor='black')
    norm = colors.Normalize(hist_score.min(), hist_score.max())
    
    # Use the reversed 'viridis' colormap for the histogram bars
    colormap = plt.cm.get_cmap(f'{colormap}')
    for thisbin, thispatch in zip(bins, patches):
        color = colormap(norm(thisbin))
        thispatch.set_facecolor(color)
    
    axs[1].set_xlabel('Prozent', fontsize=18)
    axs[1].set_ylabel(score_label, fontsize=base_fontsize)
    axs[1].tick_params(axis='both', which='major', labelsize=20,width=3,length=8)
    #axs[1].xaxis.set_major_formatter(PercentFormatter(xmax=len(relative_error)))
    #axs[1].set_ylim(0, mre + 1)  # Adjust the y-axis limit as needed
    #axs[1].set_yticks(np.round(np.linspace(np.max(hist_score), np.min(hist_score), 10),1))
    #axs[1].set_yticks(np.round(np.logspace(0, 1, 20),2))
    axs[1].set_xticks(np.round(np.linspace(0, xlasthist, 5),1))
    axs[1].tick_params(axis='x',labelrotation=90,labelsize=16)
    axs[1].grid(True)
    for asi in ['top','bottom','left','right']:
        axs[0].spines[asi].set_linewidth(2.5)
        axs[1].spines[asi].set_linewidth(2)
    # Add color bar
    if colorbar_enabled == True:
        cbar = plt.colorbar(scatter, ax=axs[0])
        cbar.ax.tick_params(labelsize=base_fontsize,width=3,length=8,pad=5)

    # Adjust layout
    plt.tight_layout(pad=1.5)
    file_name = title.replace(' ', '_')
    # Save or show the plot
    if save_path:
        plt.savefig(f'{save_path}/Scatter_{file_name}.png', bbox_inches='tight')
    else:
        plt.show()

def Plot_histogram(hist_score, figure_size=(15, 15),title='Auswertung des ersten Haupsatzs ' ,score_label='$Δ\dot{H}_{wf}+Δ\dot{H}_{sf} $',unit='[W]', base_fontsize=28, xlasthist=1, colormap='jet',xticks=[],yticks=[],num_bins=70,xlim=[],label_rotation=90):
    fig, axs = plt.subplots(1, 1, figsize=figure_size)
    
    # Calculate the number of bins and width

    bin_width = (hist_score.max() - hist_score.min()) / num_bins
    
    # Plot the histogram with specified number of bins and bin width
    N, bins, patches = axs.hist(hist_score, bins=np.arange(hist_score.min(), hist_score.max() + bin_width, bin_width),
                                 density=False, weights=100 * np.ones(hist_score.shape[0]) / hist_score.shape[0],
                                 log=False, color='lightblue', edgecolor='black')
    
    norm = colors.Normalize(hist_score.min(), hist_score.max())
    
    # Use the reversed 'viridis' colormap for the histogram bars
    colormap = plt.cm.get_cmap(f'{colormap}')
    for thisbin, thispatch in zip(bins, patches):
        color = colormap(norm(thisbin))
        thispatch.set_facecolor(color)
    
    axs.set_xlabel(score_label+' '+unit, fontsize=base_fontsize)
    axs.set_ylabel('Prozent', fontsize=base_fontsize)
    axs.tick_params(axis='both', which='major', labelsize=20, width=3, length=8)
    for asi in ['top', 'bottom', 'left', 'right']:
        axs.spines[asi].set_linewidth(2.5)

    # Set the xticks with even spacing
    if len(xticks)>0:

        axs.set_xticks(xticks)
    if len(yticks)>0:

        axs.set_yticks(yticks)
    if len(xlim)>1:    

        axs.set_xlim(xlim[0], xlim[1])
    axs.set_title(title, fontsize=base_fontsize+2)
    plt.subplots_adjust(bottom=0.3)
    # Rotate x-axis labels if needed
    axs.tick_params(axis='x', labelrotation=label_rotation, labelsize=base_fontsize)
    axs.tick_params(axis='y', labelsize=base_fontsize)
    axs.grid(True)


def Plot_Line_XY(x,targets, outputs, xlabel, ylabel, title, legend_labels=None, figure_size=(15, 12),base_fontsize=26, save_path=None ,dpi=300,textbox_xy=[0.67,0.77],new_fig=True,unit=''):
    """
    Generate a line plot to compare the targets and MLP outputs

    Parameters:
    - targets: Array of target values (inversed Normalization required).
    - outputs: Array of predicted values (inversed Normalization required).
    - xlabel: Label for the x-axis.
    - ylabel: Label for the y-axis.
    - title: Title of the plot.
    - legend_labels: Labels for the legend (if applicable).
    - figure_size: Tuple specifying the figure size.
    - save_path: Path to save the plot ,it saves the Plot if the path exists (optional).
    - dpi : resolution of the Plot (optional)

    Returns:
    - None (displays the plot).
    """

    point_size = 20

    # Create a 2x1 grid layout
    
    fig, axs = plt.subplots(1, 1, figsize=figure_size)
    relative_error = (np.abs(((outputs - targets) / (targets)) * 100))
    global mre, mmre
    mre = np.mean(relative_error)
    # Display RMSE value at the top-left part of the diagram
    rmse_text = f'RMSE: {np.sqrt(np.mean((outputs - targets)**2)):.2f} {unit}\nMAE: {np.mean(np.abs(outputs - targets)):.2f} {unit}\nMRE: {100*np.mean((outputs - targets)/outputs):.2f}%\n$R^{2}$: {r2_score(targets, outputs):.4f}'
    axs.text(textbox_xy[0], textbox_xy[1], rmse_text, transform=axs.transAxes, fontsize=base_fontsize, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    # Line plot
    axs.set_title(title+unit, fontsize=base_fontsize+2)
    axs.plot(x,targets, linestyle='-', color='red', linewidth=3)
    axs.plot(x,outputs, linestyle=':', color='black', linewidth=4)
    axs.set_xlabel(xlabel, fontsize=base_fontsize)
    axs.set_ylabel(ylabel+unit, fontsize=base_fontsize)
    axs.tick_params(axis='both', which='major', labelsize=base_fontsize-2, width=3, length=8)
    axs.grid(True, which='both', linewidth=2)
    axs.legend(['physikalisches Modell','MLP-Modell'],loc='right', fontsize=base_fontsize-2)

    # Adjust spines
    for asi in ['top', 'bottom', 'left', 'right']:
        axs.spines[asi].set_linewidth(2.5)

    # Adjust layout
    plt.tight_layout(pad=1.5)
    file_name = title.replace(' ', '_')
    # Save or show the plot
    if save_path:
        plt.savefig(f'{save_path}/Curve_{file_name}.png', bbox_inches='tight')
    else:
        plt.show()  
        
#%% 
# Datensatgrößeanalyse        
if Size_Analysis== True:

    #=========================Eingaben====================================
    #Add list of Neuroal Networks to be compared     
    Address_list = [Dataset_Size_Analysis_Address+r"\Neural_Network_0.001\nn_L3_N128_relu_B256_a0.0001.pkl",
                    Dataset_Size_Analysis_Address+r"\Neural_Network_0.005\nn_L4_N256_relu_B256_a0.0001.pkl",
                    Dataset_Size_Analysis_Address+r"\Neural_Network_0.01\nn_L4_N256_relu_B128_a0.0001.pkl",
                    Dataset_Size_Analysis_Address+r"\Neural_Network_0.05\nn_L4_N256_relu_B128_a0.0001.pkl",
                    Dataset_Size_Analysis_Address+r"\Neural_Network_0.1\nn_L4_N256_relu_B128_a0.0001.pkl",
                    Dataset_Size_Analysis_Address+r"\Neural_Network_0.5\nn_L4_N256_relu_B128_a0.0001.pkl",
                    Dataset_Size_Analysis_Address+r"\Neural_Network_1\nn_L4_N256_relu_B128_a0.0001.pkl",
                    Dataset_Size_Analysis_Address+r"\Neural_Network_10\nn_L4_N256_relu_B128_a0.0001.pkl",
                    Dataset_Size_Analysis_Address+r"\Neural_Network_25\nn_L4_N256_relu_B128_a0.0001.pkl",
                    Dataset_Size_Analysis_Address+r"\Neural_Network_50\nn_L4_N256_relu_B256_a0.0001.pkl",
                    Dataset_Size_Analysis_Address+r"\Neural_Network_75\nn_L4_N256_relu_B128_a0.0001.pkl",
                    Dataset_Size_Analysis_Address+r"\Neural_Network_100\nn_L4_N256_relu.pkl"
                    ]
    
    
    #list of the Dataset in the Dataset
    Size_list =[0.001,0.005,0.01,0.05,0.1,0.5,1,10,25,50,75,100]
    result_list = []

    
    DF = pd.DataFrame()
    DF_ref =  pd.DataFrame()
    
    for sz , Address in enumerate(Address_list):
        Init_Dict={'Network_Address': Address,
                   'test_Dataset_Address' :'validation',
                    'AF': 'Propane*Isobutane*Pentane',
                    'SF': 'water',
                    'save_Address': r'Results\Validation\size_comparison'
            }
        
        Model_Validation =Network_Postprocessing(Init_Dict)
        #inputs_test_ref=Model_Validation.scaler_inputs.transform(inputs_test_ref)
        #Calculate the Scores
        output= Model_Validation.predict(Model_Validation.validation_dataset_inputs_nn)
        targets = Model_Validation.validation_dataset_targets.to_numpy()
        scores = Model_Validation.calculate_scores_(output,targets)
        scores_Table = pd.DataFrame(scores).T
        scores_Table.columns= ['h_w','p_w','h_s','p_s']
        #output_ref =  Model_Validation.predict(inputs_test_ref)
        #scores_ref  = Model_Validation.calculate_scores_(output_ref,Targets_test_ref)
        #scores_Table_ref = pd.DataFrame(scores_ref).T
        #scores_Table_ref.columns= ['h_w','p_w','h_s','p_s']
        #Calculate the Temperature , Quality and Entropy of validation set
        Inputs = Model_Validation.validation_dataset_inputs
        Targets = Model_Validation.validation_dataset_targets
        Outputs = pd.DataFrame(output,columns=Model_Validation.column_targets)
        #Reference DFs
       # Inputs_ref = pd.DataFrame(inputs_test_ref,columns=Inputs.columns)
       # Targets_ref = pd.DataFrame(Targets_test_ref,columns=Targets.columns)
       # Outputs_ref = pd.DataFrame(output_ref,columns=Targets.columns)
        TSQ_inputs , TSQ_targets ,TSQ_outputs = Model_Validation.Calculate_Tsq_from_ph(Inputs.iloc[0:TSQ_index],Targets[0:TSQ_index],Outputs[0:TSQ_index])
       # TSQ_inputs_ref , TSQ_targets_ref ,TSQ_outputs_ref = Model_Validation.Calculate_Tsq_from_ph(Inputs_ref.iloc[0:TSQ_index],Targets_ref.iloc[0:TSQ_index],Outputs_ref.iloc[0:TSQ_index])
        first_law_targets = np.mean(np.abs(Inputs['m_w']* (Targets['h_w']-Inputs['h_w'])  - Inputs['m_s']*(-Targets['h_s']+Inputs['h_s'])))
        first_law_outputs = np.mean(np.abs(Inputs['m_w']* (Outputs['h_w']-Inputs['h_w'])  - Inputs['m_s']*(-Outputs['h_s']+Inputs['h_s'])))
        #========
        #first_law_targets_ref = np.mean(np.abs(Inputs_ref['m_w']* (Targets_ref['h_w']-Inputs_ref['h_w'])  - Inputs_ref['m_s']*(-Targets_ref['h_s']+Inputs_ref['h_s'])))
        #first_law_outputs_ref = np.mean(np.abs(Inputs_ref['m_w']* (Outputs_ref['h_w']-Inputs_ref['h_w'])  - Inputs_ref['m_s']*(-Outputs_ref['h_s']+Inputs_ref['h_s'])))
        #second_law_targets_ref = np.min(np.abs(Inputs_ref['m_w']* (TSQ_targets_ref['s_w']-TSQ_inputs_ref['s_w'])  + Inputs_ref['m_s']*(-TSQ_targets_ref['s_s']+TSQ_inputs_ref['s_s'])))
        #second_law_outputs_ref = np.min(np.abs(Inputs_ref['m_w']* (TSQ_outputs_ref['s_w']-TSQ_inputs_ref['s_w'])  + Inputs_ref['m_s']*(-TSQ_outputs_ref['s_s']+TSQ_inputs_ref['s_s'])))
        print(f'First_law Targets:{first_law_targets} ,   First_law outputs :{first_law_outputs} ')
        second_law_targets = np.min(np.abs(Inputs['m_w']* (TSQ_targets['s_w']-TSQ_inputs['s_w'])  + Inputs['m_s']*(-TSQ_targets['s_s']+TSQ_inputs['s_s'])))
        second_law_outputs = np.min(np.abs(Inputs['m_w']* (TSQ_outputs['s_w']-TSQ_inputs['s_w'])  + Inputs['m_s']*(-TSQ_outputs['s_s']+TSQ_inputs['s_s'])))
        print(f'S_dot_irr Targets:{second_law_targets} ,   S_dot_irr outputs :{second_law_outputs} ')
        second_law_rel =100* abs(second_law_outputs-second_law_targets)/second_law_targets
        #second_law_rel_ref =100* abs(second_law_outputs_ref-second_law_targets_ref)/second_law_targets_ref
        series=[scores_Table.loc['rel%'].to_numpy(),second_law_rel,first_law_outputs]
       # series_ref=[scores_Table_ref.loc['rel%'].to_numpy(),second_law_rel_ref,first_law_outputs_ref]
        series=np.hstack(series)
        #series_ref=np.hstack(series_ref)
        DF[f'{Size_list[sz]}'] = series
       # DF_ref[f'{Size_list[sz]}'] = series_ref
        
    DF_filtered = DF.drop([4,5])
    
    plt.figure(figsize=[20, 12])
    plt_font_size = 32
    style_list = ['-.','--','-.','--']
    marker_list = ['o','*','v','^']
    for plot_i in range(DF_filtered.shape[0]):
        plt.semilogy(DF_filtered.iloc[plot_i],style_list[plot_i],marker=marker_list[plot_i], markersize=18, linewidth=4)
    plt.grid(True, which='both', linewidth=2)
    plt.xlabel('Anteil des Datensatzes%', fontsize=plt_font_size)
    plt.ylabel('MRE %', fontsize=plt_font_size)  
    plt.xticks(fontsize=plt_font_size-5)  
    plt.yticks(fontsize=plt_font_size-2) 
    plt.tick_params(axis='x', width=2, length=8,pad=5)
    plt.tick_params(axis='y', width=2, length=8)
    plt.title('Auswirkung der Datensatzgröße auf die Modellleistung', fontsize=plt_font_size+2)  
    plt.legend([r'$h_{wf}$ [J/kg]', r'$p_{wf}$ [pa]', r'$h_{sf}$ [J/kg]', r'$p_{sf}$ [pa]'], fontsize=plt_font_size)  
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2.5)
    DF_filtered_1_law = DF.iloc[5]
    DF_filtered_2_law = DF.iloc[4]
    
    
    plt.figure(figsize=[20, 12])
    plt.semilogy(DF_filtered_1_law,'-.',marker='o',color='red',markersize=18, linewidth=4)
    plt.grid(True,which='both',linewidth=2)
    plt.xlabel('Anteil des Datensatzes%',fontsize=plt_font_size)
    plt.ylabel('RMSE \n$Δ\dot{H}_{wf}+Δ\dot{H}_{sf} $ [W]',fontsize=plt_font_size)
    plt.xticks(fontsize=plt_font_size-5)  
    plt.yticks(fontsize=plt_font_size-2) 
    plt.tick_params(axis='x', width=2, length=8,pad=5)
    plt.tick_params(axis='y', width=2, length=8)
    plt.title('Auswirkung der Datensatzgröße auf die Modellleistung \nErster Hauptsatz' ,fontsize=plt_font_size)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2.5)
    
    plt.figure(figsize=[20, 12])
    plt.semilogy(DF_filtered_2_law,'-.',marker='*',color='green',markersize=18, linewidth=4)
    plt.grid(True,which='both',linewidth=2)
    plt.xlabel('Anteil des Datensatzes%',fontsize=plt_font_size)
    plt.ylabel('RMSE \n$\dot{S}_{irr}$  [W/K]',fontsize=plt_font_size)
    plt.xticks(fontsize=plt_font_size-5)  
    plt.yticks(fontsize=plt_font_size-2) 
    plt.tick_params(axis='x', width=2, length=8,pad=5)
    plt.tick_params(axis='y', width=2, length=8)
    plt.title('Auswirkung der Datensatzgröße auf die Modellleistung \nEntropieproduktionsstrom' ,fontsize=plt_font_size)
    for spine in plt.gca().spines.values():
        spine.set_linewidth(2.5)

#%%
Model_Validation =Network_Postprocessing(Init_Dict2)
Inputs = Model_Validation.validation_dataset_inputs
Targets = Model_Validation.validation_dataset_targets
t3=time.time()
output= Model_Validation.predict(Model_Validation.validation_dataset_inputs_nn)
t4=time.time()
targets = Model_Validation.validation_dataset_targets.to_numpy()
Outputs = pd.DataFrame(output,columns=Model_Validation.column_targets)

TSQ_inputs , TSQ_targets ,TSQ_outputs = Model_Validation.Calculate_Tsq_from_ph(Inputs.iloc[0:TSQ_index],Targets[0:TSQ_index],Outputs[0:TSQ_index])
#%%Feature Importance
if Feature_Importance_Analysis==True:
    print('Preforming Feature Importance Analysis on MLP-Modell')
    scoring='r2'
    #scoring='neg_root_mean_squared_error'
    per_result=permutation_importance(Model_Validation.model,Model_Validation.validation_dataset_inputs_nn,Model_Validation.validation_dataset_targets_nn,scoring=scoring,random_state=42)
    if 'X_C' in Model_Validation.validation_dataset_inputs:
        Imp_X_labels=[r'$\Delta$x',r'$h_{wf}$',r'$p_{wf}$',r'$\dot{m}_{wf}$',r'$h_{sf}$',r'$p_{sf}$',r'$\dot{m}_{sf}$',r'$x_{A}$',r'$x_{B}$',r'$x_{C}$',r'$d_{i}$',r'$d_{a}$']
    else:
        Imp_X_labels=[r'$\Delta$x',r'$h_{wf}$',r'$p_{wf}$',r'$\dot{m}_{wf}$',r'$h_{sf}$',r'$p_{sf}$',r'$\dot{m}_{sf}$',r'$x_{A}$',r'$x_{B}$',r'$d_{i}$',r'$d_{a}$']
    Importance=pd.Series(per_result['importances_mean'])
    
    fig,ax=plt.subplots(figsize=[12,10])
    Importance.plot.bar(x=Imp_X_labels,yerr=per_result['importances_std'], ax=ax,color='red')
    ax.set_xticklabels(labels=Imp_X_labels,fontsize=26)
    ax.set_title("Wichtigkeitsbewertung der Eingabewerten\ndurch Permutation",fontsize=24)
    ax.set_ylabel("Durchschnittliche Wichtigkeit\n(Steigung des dimensionlosen RMSE)",fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=20,width=3,length=8,pad=5)
    ax.set_axisbelow(True)
    plt.grid(True,linewidth=2)

    for asi in ['top', 'bottom', 'left', 'right']:
        ax.spines[asi].set_linewidth(2.5)

#%%    
if Perform_Validation == True:
    #Validation
    #Scatter Temperature
    Plot_Scatter_XY(TSQ_targets['T_w'],TSQ_outputs['T_w'],'physikalisches Modell' ,'MLP-Modell',r'Validierung: $T_{wf}$ ',unit='[K]',xlasthist=50)        
    Plot_Scatter_XY(TSQ_targets['T_s'],TSQ_outputs['T_s'],'physikalisches Modell' ,'MLP-Modell',r'Validierung: $T_{sf}$ ',unit='[K]',xlasthist=30) 
    #Scatter Entropie
    Plot_Scatter_XY(TSQ_targets['s_w']/1000,TSQ_outputs['s_w']/1000,'physikalisches Modell' ,'MLP-Modell',r'Validierung: $s_{wf}$ ',unit=r'$\left[\frac{kj}{kg \cdot k}\right]$',xlasthist=30) 
    Plot_Scatter_XY(TSQ_targets['s_s']/1000,TSQ_outputs['s_s']/1000,'physikalisches Modell' ,'MLP-Modell',r'Validierung: $s_{sf}$ ',unit=r'$\left[\frac{kj}{kg \cdot k}\right]$',xlasthist=60)
    #Scatter Enthalpie
    Plot_Scatter_XY(Targets['h_w']/1000,Outputs['h_w']/1000,'physikalisches Modell' ,'MLP-Modell',r'Validierung: $h_{wf}$ ',unit=r'$\left[\frac{kj}{kg}\right]$',alpha=[-3,10,-10,3])
    Plot_Scatter_XY(Targets['h_s']/1000,Outputs['h_s']/1000,'physikalisches Modell' ,'MLP-Modell',r'Validierung: $h_{sf}$ ',unit=r'$\left[\frac{kj}{kg}\right]$',alpha=[-3,10,-3,10])
    #Scatter Pressure
    Plot_Scatter_XY(Targets['p_w']/1000,Outputs['p_w']/1000,'physikalisches Modell' ,'MLP-Modell',r'Validierung: $p_{wf}$ ',unit='[kpa]')
    Plot_Scatter_XY(Targets['p_s']/1000,Outputs['p_s']/1000,'physikalisches Modell' ,'MLP-Modell',r'Validierung: $p_{sf}$ ',unit='[kpa]')
    #Scatter Entropieproductionrate
    S_dot_irr_targets = Inputs['m_w'][:TSQ_index]* (TSQ_targets['s_w']-TSQ_inputs['s_w'])  + Inputs['m_s'][:TSQ_index]*(-TSQ_targets['s_s']+TSQ_inputs['s_s'])
    S_dot_irr_outputs = Inputs['m_w'][:TSQ_index]* (TSQ_outputs['s_w']-TSQ_inputs['s_w'])  + Inputs['m_s'][:TSQ_index]*(-TSQ_outputs['s_s']+TSQ_inputs['s_s'])
    Plot_Scatter_XY(S_dot_irr_targets/1000,S_dot_irr_outputs/1000,'physikalisches Modell' ,'MLP-Modell',r'Validierung: $\dot{s}_{irr}$ ',unit=r'$\left[\frac{kW}{k}\right]$',xlasthist=25)
    #Scatter Energy residual
    energy_residuals_targets = Inputs['m_w']* (Targets['h_w']-Inputs['h_w'])  - Inputs['m_s']*(-Targets['h_s']+Inputs['h_s'])
    energy_residuals_outputs = Inputs['m_w']* (Outputs['h_w']-Inputs['h_w'])  - Inputs['m_s']*(-Outputs['h_s']+Inputs['h_s'])
    Plot_Scatter_XY(energy_residuals_targets,energy_residuals_outputs,'physikalisches Modell' ,'MLP-Modell',r'Validierung: $\dot{E}_{aus}-\dot{E}_{ein} $ ',unit='[W]',error_type='abs',log=True,zoom=False,Textbox_xy=[0.05,0.2],xlasthist=30)
    #plot Histogram for first Law
    Plot_histogram(energy_residuals_outputs,title='Auswertung des ersten Haupsatzes',xticks=[-200,-150, -100,-75, -50, -25 ,0,20 ,50, 75,100],yticks=[1,5,10,15,20,25],xlim=[-200, 100])

#%%
def Plot_Curve_Diagram(Diagram_Type,p_w,p_s,m_w,m_s,x_A,x_B,Di,Da,textbox_xy=[0.67,0.77],textbox_score_xy=[0.67,0.77]):
    #Draw the Curve Diagrams
    Test_P_w = p_w #pressure of working fluid
    Test_P_s = p_s # pressure of secondary fluid
    Test_M_s = m_w # Mass flow rate of secondary fluid
    Test_M_w = m_s # Mass flow rate of Working fluid
    Test_XA = x_A # Mole fraction of Propane
    Test_XB = x_B # Mole fraction of isobutane
    Test_XC = 1-(Test_XA+Test_XB) # Mole fraction of Pentane
    Test_Di = Di # inner Diameter of tube
    Test_Da = Da # inner Diamter of Shell
    
    #Prepare the Physical Model
    sample_n_size=0.1
    
    Input_params = {'WF': 'Propane*Isobutane*Pentane',
                  'SF': 'water',
                  'random_sample': 0.1,
                  'p_w': [2, 23],
                  'p_s': 1.5,
                  'D_w': Test_Di,
                  'D_s': Test_Da,
                  'm_r': [2, 10],
                  'm_w': [1e-3, 5e-1],
                  'm_s': [2.5e-3, 0.1],
                  'save_directory': rf"D:\Masterarbeit\Test\size_{sample_n_size}\Results",
                  'Save_Outputs': False,
                  'verbose': 1,
                  'L': 100,
                  't': 1.6e-3,
                  'ode_atol': 1e-2,
                  'ode_rtol': 1e-1,
                  'method': 'RK45',
                  'n_samples':sample_n_size,
                  'skip_list':[],
                  'cache_folder': rf"D:\Masterarbeit\Test\size_{sample_n_size}\cache"}
    
    #initialize the Physical model
    Physical_Model = Create_new.Heat_Exchanger_Model(Input_params)
    pressure_array = np.linspace(1,25,20)
    result_list_1 = []
    #define inputs Parameters
    Solver_Input = np.array([Test_P_w,Test_P_s,Test_M_w,Test_M_s,Test_XA,Test_XB,Test_XC,Test_Di,Test_Da])
    #solve the physical model with given Parameter
    status=Physical_Model.solve(Solver_Input)
    current_output= Physical_Model.current_output
    #generate the Dataset for MLP Model.
    Inputs_,Targets_ = Physical_Model.generate_nn_dataset(current_output.T, [Test_M_w,Test_M_s,Test_XA,Test_XB,Test_XC,Test_Di,Test_Da],no_outer_iteration=True)
    #predict the Targets with MLP Model.
    if Model_Validation.scaler_inputs.get_feature_names_out().shape[0] == 11:
        Inputs_=Inputs_.drop(labels=['X_C'],axis=1)
    nn_Targets = Targets_.to_numpy()
    nn_Outputs = Model_Validation.predict(Model_Validation.scaler_inputs.transform(Inputs_))
    nn_Inputs = Inputs_.to_numpy()
    base_font_size = 26
    #calculate Temperature and entropy of Targets , Outputs and inputs.
    #TS_In_Mat,TS_T_Mat,TS_Out_Mat = Model_Validation.Calculate_Tsq_from_ph(Inputs_,Targets_,pd.DataFrame(nn_Outputs,columns=Targets_.columns))
    box_text = '$d_i$ [mm]: {:.2f} , $d_a$ [mm]: {:.2f}\n'.format(Test_Di, Test_Da) + \
           r'$\dot{m}_{wf}$ [kg/s]:'+f' {m_w:.2f} ,'+' $\dot{m}_{sf}$ [kg/s]: '+f'{m_s:.2f}\n' + \
           '$X_A$: {:.2f} , $X_B$: {:.2f}\n'.format(x_A, x_B) + \
           r'$p_{wf}$ [bar]: '+f'{p_w:.2f} ,'+r' $p_{sf}$ [bar]: '+f'{p_s:.2f}'
    
    if Diagram_Type == 'T':
        TS_In_Mat,TS_T_Mat,TS_Out_Mat = Model_Validation.Calculate_Tsq_from_ph(Inputs_,Targets_,pd.DataFrame(nn_Outputs,columns=Targets_.columns))
        #Plot line : Temperature of working fluid
        Plot_Line_XY(nn_Inputs[:,0],TS_T_Mat['T_w']-273.15,TS_Out_Mat['T_w']-273.15,'Δx [m]' ,'T ','Temperatur des Arbeitsfluids ','Bewertung\Tempertur des Arbeitsfluids',textbox_xy=textbox_score_xy,unit='[°C]',base_fontsize=30)
        #plt.plot(nn_Inputs[:,0],TS_In_Mat['T_s']-273.15,'blue',linewidth=3)
        #plt.legend(['physikalisches Modell','MLP-Modell','Temperatur des Sekundärfluids [°C]'],fontsize=22)
        plt.text(textbox_xy[0], textbox_xy[1], box_text, fontsize=base_font_size, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.8))
        
    elif Diagram_Type == 'p':
    
        #Plot line : Pressure of working fluid 
        Plot_Line_XY(nn_Inputs[:,0],pd.DataFrame(nn_Outputs,columns=Targets_.columns)['p_w']/1000,Targets_['p_w']/1000,'Δx [m]' ,'p ','Druck des Arbeitfluids ','Bewertung\Druck des Arbeitsfluids',textbox_xy=textbox_score_xy,unit='[kPa]',base_fontsize=30)
        # plt.plot(nn_Inputs[:,0],Inputs_['p_s']/1000,'blue',linewidth=3)
        # plt.legend(['physikalisches Modell','MLP-Modell','Druck des Sekundärfluids [kpa]'],fontsize=22)
        plt.text(textbox_xy[0], textbox_xy[1], box_text, fontsize=base_font_size, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.8))
        
    elif Diagram_Type == 'h':
        #Plot line : Enthalpy of working fluid 
        Plot_Line_XY(nn_Inputs[:,0],pd.DataFrame(nn_Outputs,columns=Targets_.columns)['h_w']/1000,Targets_['h_w']/1000,'Δx [m]' ,'$h_{wf}$','spezifische Enthalpie des Arbeitsfluids','Bewertung\spezifische Enthalpie des Arbeitsfluids',textbox_xy=textbox_score_xy,unit='[kJ/kg]',base_fontsize=30)
        #plt.scatter(nn_Inputs[:,0],pd.DataFrame(nn_Outputs,columns=Targets_.columns)['h_w']/1000)
        plt.text(textbox_xy[0], textbox_xy[1], box_text, fontsize=base_font_size, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.8))
        # plt.plot(nn_Inputs[:,0],Inputs_['h_s']/1000,'blue',linewidth=3)
        # plt.legend(['physikalisches Modell','MLP-Modell','spezifische Enthalpie des Sekundärfluids [kj/kg]'],fontsize=22)

    elif Diagram_Type == 's':
        TS_In_Mat,TS_T_Mat,TS_Out_Mat = Model_Validation.Calculate_Tsq_from_ph(Inputs_,Targets_,pd.DataFrame(nn_Outputs,columns=Targets_.columns))
        #Plot line : entropy of working fluid
        Plot_Line_XY(nn_Inputs[:,0],TS_T_Mat['s_w']/1000,TS_Out_Mat['s_w']/1000,'Δx [m]' ,'s ','spezifische Entropie des Arbeitsfluids ','Bewertung des MPL-Modells\spezifische Entropie des Arbeitsfluids',textbox_xy=textbox_score_xy,unit='$[kJ/(kg \cdot k)]$',base_fontsize=30)
        plt.text(textbox_xy[0], textbox_xy[1], box_text, fontsize=base_font_size, verticalalignment='bottom', bbox=dict(facecolor='white', alpha=0.8))
        # plt.plot(nn_Inputs[:,0],TS_In_Mat['s_s']/1000,'blue',linewidth=3)
        # plt.legend(['physikalisches Modell','MLP-Modell','spezifische Entropie des Sekundärfluid [kj/kg.k]'],fontsize=22)


def Validate_by_Simulation(p_w, p_s, m_w, m_s, x_A, x_B, Di, Da):
    #Draw scatter diagram
    Test_P_w = p_w #pressure of working fluid
    Test_P_s = p_s # pressure of secondary fluid
    Test_M_s = m_w # Mass flow rate of secondary fluid
    Test_M_w = m_s # Mass flow rate of Working fluid
    Test_XA = x_A # Mole fraction of Propane
    Test_XB = x_B # Mole fraction of isobutane
    Test_XC = 1-(Test_XA+Test_XB) # Mole fraction of Pentane
    Test_Di = Di # inner Diameter of tube
    Test_Da = Da # inner Diamter of Shell
    
    #Prepare the Physical Model
    sample_n_size=0.1
    
    Input_params = {'WF': 'Propane*Isobutane*Pentane',
                  'SF': 'water',
                  'random_sample': 0.1,
                  'p_w': [2, 23],
                  'p_s': 1.5,
                  'D_w': Test_Di,
                  'D_s': Test_Da,
                  'm_r': [2, 10],
                  'm_w': [1e-3, 5e-1],
                  'm_s': [2.5e-3, 0.1],
                  'save_directory': os.path.join(parent_path,rf"Masterarbeit\Test\size_{sample_n_size}\Results"),
                  'Save_Outputs': False,
                  'verbose': 1,
                  'L': 100,
                  't': 1.6e-3,
                  'ode_atol': 1e-8,
                  'ode_rtol': 1e-5,
                  'method': 'Radau',
                  'n_samples':sample_n_size,
                  'skip_list':[],
                  'cache_folder': os.path.join(parent_path,rf"Masterarbeit\Test\size_{sample_n_size}\cache")}
    
    #initialize the Physical model
    Physical_Model = Create_new.Heat_Exchanger_Model(Input_params)
    pressure_array = np.linspace(1,25,20)
    result_list_1 = []
    #define inputs Parameters
    Solver_Input = np.array([Test_P_w,Test_P_s,Test_M_w,Test_M_s,Test_XA,Test_XB,Test_XC,Test_Di,Test_Da])
    #solve the physical model with given Parameter
    try: 
        status=Physical_Model.solve(Solver_Input)
        current_output= Physical_Model.current_output
        #generate the Dataset for MLP Model.
        Inputs_,Targets_ = Physical_Model.generate_nn_dataset(current_output.T, [Test_M_w,Test_M_s,Test_XA,Test_XB,Test_XC,Test_Di,Test_Da],no_outer_iteration=True)
        #predict the Targets with MLP Model.
        nn_Inputs = Model_Validation.scaler_inputs.transform(Inputs_.to_numpy())
        nn_Targets=Targets_.to_numpy()
        MLP_Model=Model_Validation.model
        nn_Outputs = Model_Validation.predict(nn_Inputs)
        MRE_list=[]
        for i in range(nn_Outputs.shape[1]):  
            MRE=100*np.mean((nn_Targets[:,i]-nn_Outputs[:,i])/nn_Targets[:,i])
            MRE_list.append(MRE)
        return MRE_list
    except:return np.nan

#%%
#Interpolation-Extrapolation Analyse
if Plot_Curve_Diagrams==True:
    Plot_Curve_Diagram('h',7.5,1.5,0.04696605225326494,0.01197717744705945,0.34,0.36,15.9968,25,textbox_xy=[-0.5,185],textbox_score_xy=[0.64,0.8]) #Exact h
    Plot_Curve_Diagram('T',7.5,1.5,0.04696605225326494,0.01197717744705945,0.34,0.36,15.9968,25,textbox_xy=[19,59],textbox_score_xy=[0.69,0.8]) #Exact T
    Plot_Curve_Diagram('p',7.5,1.5,0.04696605225326494,0.01197717744705945,0.34,0.36,15.9968,25,textbox_xy=[19,748.9],textbox_score_xy=[0.67,0.8]) #Exact p
    Plot_Curve_Diagram('s',7.5,1.5,0.04696605225326494,0.01197717744705945,0.34,0.36,15.9968,25,textbox_xy=[0,1],textbox_score_xy=[0.58,0.81]) #Exact s
    per_res=Validate_by_Simulation(7.5,1.5,0.04696605225326494,0.01197717744705945,0.34,0.36,15.9968,25)
    
    #Interpolation , p variation
    Plot_Curve_Diagram('h',15,1.5,0.04696605225326494,0.01197717744705945,0.34,0.36,15.9968,25,textbox_xy=[7,580],textbox_score_xy=[0.64,0.4])
    Plot_Curve_Diagram('T',15,1.5,0.04696605225326494,0.01197717744705945,0.34,0.36,15.9968,25,textbox_xy=[7.2,90],textbox_score_xy=[0.69,0.8])
    #Extrapolation , p variation
    Plot_Curve_Diagram('h',4,1.5,0.04696605225326494,0.01197717744705945,0.34,0.36,15.9968,25,textbox_xy=[0,120],textbox_score_xy=[0.64,0.8])
    Plot_Curve_Diagram('T',4,1.5,0.04696605225326494,0.01197717744705945,0.34,0.36,15.9968,25,textbox_xy=[29.1,35],textbox_score_xy=[0.69,0.8])
    #Interpolation , D variation
    Plot_Curve_Diagram('h',7.5,1.5,0.04696605225326494,0.01197717744705945,0.34,0.36,20,25,textbox_xy=[0,180],textbox_score_xy=[0.64,0.8])
    Plot_Curve_Diagram('T',7.5,1.5,0.04696605225326494,0.01197717744705945,0.34,0.36,20,25,textbox_xy=[6.90,102],textbox_score_xy=[0.69,0.4])
    #Extrapolation , D variation
    Plot_Curve_Diagram('h',7.5,1.5,0.04696605225326494,0.01197717744705945,0.34,0.36,12,25,textbox_xy=[37,511],textbox_score_xy=[0.05,0.25])
    Plot_Curve_Diagram('T',7.5,1.5,0.04696605225326494,0.01197717744705945,0.34,0.36,12,25,textbox_xy=[37,101],textbox_score_xy=[0.68,0.4])

#%%
#Datensatzverteilung Analyse
if Plot_Dataset_Distribution == True:
    Total_Dataset=joblib.load(Dataset_Address)
    Plot_histogram(Total_Dataset['inputs']['p_w']/1e5,title='Verteilung des Gasamtdatensatzes: $p_{wf}$\nEingangswerte',num_bins=25,score_label=r'$p_{wf}$ ',unit='[bar]',xticks=[2,4,6,8,10,12,15,18,20,22,25],label_rotation=0)
    Plot_histogram(Total_Dataset['inputs']['p_s']/1e5,title='Verteilung des Gesamtdatensatzes: $p_{sf}$\nEingangswerte',num_bins=25,score_label=r'$p_{sf}$ ',unit='[bar]',label_rotation=0)
    Plot_histogram(Total_Dataset['inputs']['h_w']/1e3,title='Verteilung des Gesamtdatensatzes: $h_{wf}$\nEingangswerte',num_bins=25,score_label=r'$h_{wf}$ ',unit='[kJ/kg]',label_rotation=0)
    Plot_histogram(Total_Dataset['inputs']['h_s']/1e3,title='Verteilung des Gesamtdatensatzes: $h_{sf}$\nEingangswerte',num_bins=25,score_label=r'$h_{sf}$ ',unit='[kJ/kg]',label_rotation=0)
    Plot_histogram(Total_Dataset['inputs']['m_w']*1e3,title='Verteilung des Gesamtdatensatzes: $m_{wf}$\nEingangswerte',num_bins=25,score_label=r'$m_{wf}$ ',unit='[g/s]',label_rotation=0)
    Plot_histogram(Total_Dataset['inputs']['m_s']*1e3,title='Verteilung des Gesamtdatensatzes: $m_{sf}$\nEingangswerte',num_bins=25,score_label=r'$m_{sf}$ ',unit='[g/s]',label_rotation=0)
    Plot_histogram(Total_Dataset['inputs']['X_A'],title='Verteilung des Gesamtdatensatzes: $x_{A}$\nEingangswerte',num_bins=25,score_label=r'$x_{A}$ ',unit='',label_rotation=0)
    Plot_histogram(Total_Dataset['inputs']['X_B'],title='Verteilung des Gesamtdatensatzes: $x_{B}$\nEingangswerte',num_bins=25,score_label=r'$x_{B}$ ',unit='',label_rotation=0)
    Plot_histogram(Total_Dataset['inputs']['X_C'],title='Verteilung des Gesamtdatensatzes: $x_{C}$\nEingangswerte',num_bins=25,score_label=r'$x_{C}$ ',unit='',label_rotation=0)
    Plot_histogram(Total_Dataset['targets']['h_w']/1e3,title='Verteilung des Gesamtdatensatzes: $h_{wf}$\nZielwerte',num_bins=25,score_label=r'$h_{wf}$ ',unit='[kJ/kg]',label_rotation=0)
    Plot_histogram(Total_Dataset['targets']['h_s']/1e3,title='Verteilung des Gesamtdatensatzes: $h_{sf}$\nZielwerte',num_bins=25,score_label=r'$h_{sf}$ ',unit='[kJ/kg]',label_rotation=0)


