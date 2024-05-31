# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 17:05:11 2024

@author: Alireza Norouzi , 3151301
"""
import numpy as np
from scipy.integrate import solve_ivp
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
from ht.conv_internal import Nu_conv_internal as Nusselt
from ht.condensation import Cavallini_Smith_Zecchin as cond_Cavallini
from fluids.two_phase import two_phase_dP
import fluids.friction as frict
import matplotlib.pyplot as plt
import os
import pandas as pd
from winsound import Beep
import gc
import joblib
import multiprocessing
import concurrent.futures
import psutil
import glob
import re
import seaborn as sns
import time
# Import Saltelli-SampleSet-Function
from SALib.sample import sobol

'''
User Guide:
 Set the Refprop Directory os.environ['RPPREFIX'] = r'C:/Program Files (x86)/REFPROP' to your Refprop Directory.
1.Add a Save directory to direction_list and cache_folder in Init_Dict Dictionary
2.Change the Simulation Settings in Init_Dict Dictionary
3.Define a Physical Model (Model = Heat_Exchanger_Model(Init_Dict)  ), set the cache address and saving address.
4.Solve the Physical Model with given Parameter Grid using either serial solver or parallel solver with Model.simulate()
5.use Model.Postprocessing method to generate datasets for Neural Network and check if the Datasets are thermodynamically valid (first law and second law)
6.use Model.merge_datasets method to merge the generated datasets and take random samples (recommanded Homogenous random sampling)
'''

# Global Settings
os.environ['RPPREFIX'] = r'C:/Program Files (x86)/REFPROP'
RP = REFPROPFunctionLibrary(os.environ['RPPREFIX'])
RP.SETPATHdll(os.environ['RPPREFIX'])
MASS_BASE_SI = RP.GETENUMdll(0, "MASS BASE SI").iEnum
SI = RP.GETENUMdll(0, "SI").iEnum


class Heat_Exchanger_Model():
    """Simulate a Shell and Tube Heatexchanger , calculate the Enthalpy and Pressure in all cells along the pipes

    Initialize the Module with a Dictionary that contains these keys:

    - WF:Working fluid in this format : Propane*Isobutane*Pentane

    - AF: Secondary fluid 

    - random_sample : random sample size , in Data Generation Algorithm

    - p_w: list of Pressures in bar 

    - D_w : Diameter of the Inner Pipe (Tube) in mm

    - D_s : Diameter of the Outer Pipe (Shell) in mm

    - z_w : list of Working fluid Mass fractions  

    - m_r : list of Mass flow rate ratios m_s/m_w

    - m_s : list of Mass flow rates of Secondary fluid 

    - z_number_samples : (optional) number of samples to generate random combinations of Mole fraction

    """

    def __init__(self, Init_Dict):
        self.AF = Init_Dict['WF']
        self.SF = Init_Dict['SF']
        self.Superheating_Temperature = Init_Dict.get('superheat_temp', 70)
        self.Data_Generation_dx = 0.1  # Data Generation for each length  [m]
        # Temperature difference between fluids as Stop criteria for Solver
        self.DT_tolerance = Init_Dict.get('dT', 1)
        self.ODE_tolerance = Init_Dict.get(
            'ode_atol', 1e-5)  # ODE solver absolute tolerance
        self.ODE_R_tolerance = Init_Dict.get(
            'ode_rtol', 1e-3)  # ODE solveer relative rolerance
        self.Random_Sample_Size = Init_Dict.get('random_sample', 0.01)
        self.ODE_solver_method = Init_Dict.get('solver_method', 'Radau')
        # list to generate different datasets with different inlet self.AF Pressure.
        self.p_w_list = Init_Dict['p_w']
        # list to generate data with different mass flow rate ratios
        self.m_dot_r_list = Init_Dict['m_r']
        # list to generate data with different self.SF mass flow rates
        self.m_dot_s_list = Init_Dict['m_s']
        # list to generate data with different self.AF mass flow rates
        self.m_dot_w_list = Init_Dict['m_w']
        self.D_w = Init_Dict['D_w']  # Diameter of Tube [mm]
        self.D_s = Init_Dict['D_s']  # Diameter of Shell [mm]
        self.t_Tube = Init_Dict.get('t', 1.6e-3)  # Tube`s wall Thickness [m]
        self.L = Init_Dict.get('L', 100)  # Length of the Shell and Tube [m]
        self.ep = Init_Dict.get('ep', 0.045*1e-3)  # Pipe`s Roughness [m]
        # Pipe`s Thermal Conductivity [w/m*K]
        self.K_pipe = Init_Dict.get('K', 401)
        self.dx = Init_Dict.get('dx', 0.5)  # Grid size
        self.u_s_limit = 1.5
        self.u_w_limit = 10
        
        self.cache_folder = Init_Dict.get('cache_folder', '')
        if not os.path.exists(self.cache_folder):
            os.makedirs(self.cache_folder)
            
        self.verbose_level = Init_Dict.get('verbose', 0)
        self.Save_Outputs = Init_Dict.get('Save_Outputs', False)
        self.n_samples = Init_Dict.get('n_samples', 11)
        self.P_s = Init_Dict.get('p_s', 1.0)  # Input Secondary Fluid Pressure
        self.z_s = [1.0]
        self.z_number_samples = Init_Dict.get('z_number_samples', 0)
        self.memory_limit_percent = 95.0
        self.skip_batch_list = Init_Dict.get('skip_list', [])
        self.Save_Direction =Init_Dict.get('save_directory', '')
        
        if not os.path.exists(self.Save_Direction):
            os.makedirs(self.Save_Direction)
        self.Monitoring_columns = ['x', 'T_w', 's_w', 'h_w', 'p_w', 'rho_w', 'cp_w', 'K_w', 'vis_w', 'Pr_w', 'u_w', 'Re_w', 'Nu_w', 'alpha_w', 'R_total_dx', 'dh_w', 'f_w',
                                   'q_w', 'phase_w', 'T_s', 's_s', 'h_s', 'p_s', 'rho_s', 'cp_s', 'K_s', 'vis_s', 'Pr_s', 'u_s', 'Re_s', 'Nu_s', 'alpha_s', 'R_total_dx', 'dh_s', 'f_s', 'q_s', 'phase_s']
        self.Output_columns = ['x', 'h_w', 'h_s', 'p_w', 'p_s']
        # Calculate the hoop stress based on the Given Diameter and pipe thickness
        self.hoop_stress = (
            self.p_w_list[-1]*1e5) * (self.D_w*1e-3) / (2*self.t_Tube)
        self.min_pipe_safty_factor = 2.0
        # Yield Stress of Copper 62Mpa  215*Mpa Stain-less steel .
        self.yield_strength = 62*1e6
        if self.yield_strength/self.hoop_stress < self.min_pipe_safty_factor:
            print(' Tube thickness is too low , it fails due to Pipe Pressure. ')

        if self.z_number_samples > 0:
            self.z_list = self.generate_mole_fractions(self.z_number_samples)

    def row_exists(array, target_row):
        """
        Check if the target_row exists in the 2D numpy array.

        Parameters:
        - array: 2D numpy array
        - target_row: 1D numpy array representing the row to check

        Returns:
        - True if the target_row exists in the array, False otherwise
        """
        return any(np.all(row == target_row) for row in array)


    def check_memory_usage(self):
        # Get the current memory usage
        memory_percent = psutil.virtual_memory().percent

        # Check if memory usage exceeds the limit
        if memory_percent > self.memory_limit_percent:
            self.Finish_sound()
            print(
                f"Memory usage is above the limit ({memory_percent}%). Shutting down...")
            raise MemoryError("Memory usage exceeded the limit.")

    def generate_param_grid(self):
        return self.Generate_Variable_List()

    def calculate_m_range(self):
        """Calculate the mass flow rate range based on the given Geometry , to have acceptable Velocity in pipes"""
        m_bound_d = 1000*(np.pi/4)*((self.D_s*1e-3)**2 - (self.D_w*1e-3)**2)
        m_bound_u = 3*1000*(np.pi/4)*((self.D_s*1e-3)**2 - (self.D_w*1e-3)**2)
        return m_bound_d, m_bound_u

    def update_Simulation_Status_Excel(self, new_simulation_inputs_list, new_simulation_inputs_header, excel_file='Simulations.xlsx'):
        """

        Parameters
        ----------
        new_simulation_inputs_list : list
            Simulation_Inputs_List  list that has Input variables and Status as Label.
        new_simulation_inputs_header : list
            List of headers for the Simulation_inputs_List.
        excel_file : TYPE, optional
            Address of the Excel file. The default is 'Simulations.xlsx'.

        Returns
        -------
        updated_data : TYPE
            Updated List of Simulation Inputs and Status

        """
        # Load existing data from Excel file, if it exists
        try:
            existing_data = pd.read_excel(excel_file)
        except FileNotFoundError:
            existing_data = pd.Datself.AFrame()

        # Create a Datself.AFrame for the new data
        new_data = pd.Datself.AFrame(
            new_simulation_inputs_list, columns=new_simulation_inputs_header)

        # Concatenate existing data with the new data
        updated_data = pd.concat([existing_data, new_data], ignore_index=True)

        # Save the updated Datself.AFrame to the Excel file
        updated_data.to_excel(excel_file, index=False)
        return updated_data

    def save_data_address(self, txt_file_path, string_to_append):
        '''Store the given String Report in to a Text file
        txt_file_path : address of the file
        '''
        # File path
        if not os.path.exists(txt_file_path):
            with open(txt_file_path, 'a') as file:
                file.write('')

        # Check if the string doesn't exist in the file
        string_exists = False

        # Read the existing content of the file
        with open(txt_file_path, 'r') as file:
            lines = file.readlines()
            if any(string_to_append in line for line in lines):
                string_exists = True

        # If the string doesn't exist, append it to the file
        if not string_exists:
            with open(txt_file_path, 'a') as file:
                file.write(string_to_append + '\n')

    def plot_ph(self):
        """Plot the Current P-x , h-x diagram"""
        self.Plot_PH_from_solution(self.current_output.T)

    def plot_T(self):
        # Input_list : [0]:Working fluid ,[1]:Secondary fluid , [1]:Mole fraction array of Working fluid ,[1]:Mole fraction array of Secondary fluid
        # Input_Array=[self.AF,self.SF,z_s,z_w,m_dot_w,m_dot_s,D_Tube_in,D_Tube_out,D_Shell,self.dx,ep,K_pipe]
        Input_list = [self.IA[0], self.IA[1], self.IA[3], self.IA[2]]
        T = self.Calculate_T_from_ph(self.current_output.T[:, 1:], Input_list)
        self.Plot_T(self.current_output[0, :], T)

#!!! Testing the Plots
    def Plot_PH_from_solution(self, Solution):
        '''Plot the Pressure and Enthalpy Profile in both pipes
        ------------
        Parameters:
        Solution : Solve_ivp output Array, [x,hw,hs,pw,ps]
        ------------
        Returns: Single figure with stacked x-h and two x-P subplots
        '''
        x = Solution[:, 0]
        h_w = Solution[:, 1]
        h_s = Solution[:, 2]
        P_w = Solution[:, 3]
        P_s = Solution[:, 4]
        p_w_margin = np.max(P_w/1000) - 0.1*np.max(P_w/1000)
        p_s_margin = np.min(P_s/1000) + 70
        # Create a single figure with stacked subplots
        fig = plt.figure(figsize=(15, 10))

        # Enthalpy subplot
        plt.subplot2grid((3, 2), (0, 0), rowspan=2, colspan=2)
        plt.plot(x, h_w / 1000, '-r')
        plt.plot(x, h_s / 1000, '-b')
        plt.xlabel('L [m]', fontsize=16)
        plt.ylabel('h [kJ/kg]', fontsize=16)
        plt.title('Enthalpieverläufe in der Wärmeübertrager', fontsize=24)
        plt.grid(True)
        plt.legend(['Arbeitsfluide', 'Sekundärfluide'])

        # Pressure subplot 1
        plt.subplot2grid((3, 2), (2, 0))
        plt.semilogy(x, P_w / 1000, '-r')
        plt.xlabel('L [m]', fontsize=16)
        plt.ylabel('P [kPa]', fontsize=16)
        plt.title('Druckverlauf in Arbeitfluide', fontsize=18)
        plt.hlines(p_w_margin, np.min(x), np.max(x),colors='black',linestyles='dotted')
        plt.legend(['Arbeitsfluide', 'Grenze'])
        plt.grid(True)

        # Pressure subplot 2
        plt.subplot2grid((3, 2), (2, 1))
        plt.semilogy(x, P_s / 1000, '-b')
        plt.xlabel('L [m]', fontsize=16)
        plt.ylabel('P [kPa]', fontsize=16)
        plt.title('Druckverlauf in Sekundärfluide', fontsize=18)
        plt.grid(True)
        plt.hlines(p_s_margin, np.min(x), np.max(x),colors='black',linestyles='dotted')
        plt.legend(['Sekundärfluide', 'Grenze'])
        # Adjust layout to prevent clipping of titles
        plt.tight_layout()

        return fig

    def Plot_T(self, x, T):
        '''Plots the Temperature Profile in both pipes
        ------------
        Parameters:
        -x: Array of spatial Data
        -T: Array of Temperatures
        ------------
        Returns: x-T Diagram
        '''
        middle_pos = x.shape[0] // 2  
    
        fig, ax = plt.subplots(figsize=[10, 10])
        #Plot the Temperatures
        ax.plot(x, T[:, 0]-273.15, '-r', label='Arbeitsfluid')
        ax.plot(x, T[:, 1]-273.15, '-b', label='Sekundärfluid')
        
        # Add arrows
        x_tail = x[middle_pos]
        y_tail = T[middle_pos, 0]-273.15 
        x_head = x[middle_pos+3] - x_tail
        y_head = T[middle_pos+3, 0]-273.15  - y_tail
        x_tail_sf = x[middle_pos]
        y_tail_sf = T[middle_pos, 1]-273.15 
        x_head_sf = x[middle_pos-3] - x_tail_sf
        y_head_sf = (T[middle_pos-3, 1]-273.15  - y_tail_sf )
        

        ax.arrow(x_tail, y_tail, x_head, y_head,
                 shape='full', lw=0, length_includes_head=True,overhang=0.3,head_starts_at_zero=True,head_width=1.0,color='red')
        

        ax.arrow(x_tail_sf, y_tail_sf, x_head_sf, y_head_sf,
                 shape='full', lw=0, length_includes_head=True,overhang=0.3,head_starts_at_zero=True ,head_width=1.0,color='blue')
    
        ax.set_xlabel('L [m]', fontsize=16)
        ax.set_ylabel('T [°C]', fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.set_title('Temperaturverlauf in der Wärmeübertrager', fontsize=24)
        ax.grid(True)
        ax.legend()
    
        return fig

    def generate_nn_dataset(self, Solution, Input_Array, Random_Sample_size=None,no_outer_iteration = False):
        '''
        Generate the Dataset for Neural Network.

        Parameters:
        - Solution: Output array from Solve_ivp [x, h_w, h_s, p_w, p_s]
        - Input_Array: Input array [m_w[kg/s], m_s[kg/s], X_A, X_B, X_C, Di[mm], Da[mm]]
        - Random_Sample_size: Random sample size to take random samples each iteration.

        Returns: DataFrame
        - Input: Features for Neural Network training.
        - Target: Targets for Neural Network training.
        '''
        dx = self.dx
        num_samples = Solution.shape[0]

        # Create arrays with repeated values for m_dot_w, m_dot_s, and Constants
        m_dot_w = np.full((num_samples, 1), Input_Array[0])
        m_dot_s = np.full((num_samples, 1), Input_Array[1])

        # Create dataset parts for working fluid and secondary fluid
        Dataset_part_w = np.column_stack(
            [Solution[:, 1], Solution[:, 3], m_dot_w])
        Dataset_part_s = np.column_stack(
            [Solution[:, 2], Solution[:, 4], m_dot_s])

        # Create dataset part for constants
        Dataset_part_c = np.full((num_samples, 5), [
                                 Input_Array[2], Input_Array[3],Input_Array[4] ,Input_Array[5], Input_Array[6]])

        # Create dataset parts for working fluid and secondary fluid (line parts)
        Dataset_part_l_w = np.column_stack([Solution[:, 1], Solution[:, 3]])
        Dataset_part_l_s = np.column_stack([Solution[:, 2], Solution[:, 4]])

        result_a = []  # final list to store the Input data.
        result_b = []  # final list to store the Target data.
        if no_outer_iteration == False:
            Num_Outer_Iteration = num_samples
        else:
            Num_Outer_Iteration = 1
            
        # j: is the outter iteration parameter to set the Pipe`s segment inlet
        for j in range(Num_Outer_Iteration):
            # Temporary List to store Input values in each starting point to end.
            alist = []
            # Temporary List to store Target values in each starting point to end.
            blist = []

            # i: is the inner iteration parameter to set the Pipe segment outlet
            for i in range(num_samples - j):
                L = i * dx  # Update the relative length of the Pipe Segment
                # Properties of Segments Inlet as Inputs of Working fluid
                input_w = Dataset_part_w[j]
                # Properties of Segments Outlet as Inputs of Secondary Fluid
                input_s = Dataset_part_s[i + j]
                input_c = Dataset_part_c[j]  # Append Constant Values to Inputs
                input_line = np.hstack([L, input_w, input_s, input_c])
                # Properties of Segments Outlet as Targets of Working Fluid
                output_w = Dataset_part_l_w[i + j]
                # Properties of Inlet of Segment as Targets of Secondary Fluid
                output_s = Dataset_part_l_s[j]
                output_line = np.hstack([output_w, output_s])
                # Update the Inner loop Lists with generated Data
                alist.append(input_line)
                blist.append(output_line)

            Inner_Iteration_result_Input = np.vstack(alist)
            Inner_Iteration_result_Target = np.vstack(blist)

            result_a.append(Inner_Iteration_result_Input)
            result_b.append(Inner_Iteration_result_Target)
        

        Input = np.vstack(result_a)
        Target = np.vstack(result_b)

        # If Random_Sample_size is provided, take random samples from the dataset
        if Random_Sample_size:
            Input,Target = self.take_random_sample(Input, Target,Random_Sample_size)
        # Pack Inputs and Targets into DataFrames
        Input = pd.DataFrame(Input, columns=[
                             'dx', 'h_w', 'p_w', 'm_w', 'h_s', 'p_s', 'm_s', 'X_A', 'X_B','X_C', 'D_in', 'D_out'])
        Target = pd.DataFrame(Target, columns=['h_w', 'p_w', 'h_s', 'p_s'])

        return Input, Target

    def take_random_sample(self, Input_dataset, Target_dataset, random_sample_size=0.10):
        """
        take a random sampled subset of a numpy array.
        ------------
        Parameters:
        - Input_dataset: The Inputs numpy array.
        - Input_dataset: The Targets numpy array.
        - random_sample_size: The proportion of the dataset to sample in range [0,1] or it can be an Integer to get random samples with fixed size. 
        ------------
        Returns:
        - reduced_dataset: The randomly sampled subset of the input dataset.
        """
        if random_sample_size > 1 :random_size = int(random_sample_size)
        elif random_sample_size < 1 and random_sample_size > 0 : random_size = int(random_sample_size * Input_dataset.shape[0])
        else: return Input_dataset, Target_dataset
        
        if random_sample_size > Input_dataset.shape[0]: return Input_dataset, Target_dataset
        else:
            random_idx = np.random.choice(
                Input_dataset.shape[0], size=random_size, replace=False)
            reduced_dataset_I = Input_dataset[random_idx]
            reduced_dataset_T = Target_dataset[random_idx]
            return reduced_dataset_I, reduced_dataset_T
    
    
    def Calculate_Tsq_from_ph(self, Solution, Input_list):
        '''Calculate Temperatures , specific Entropy and Quality of Working fluid and Secondary fluid 
        Parameters:
        --------    
        -Solution : Output of the Solver
        -Input_list : [0]:Working fluid ,[1]:Secondary fluid , [1]:Mole fraction array of Working fluid ,[1]:Mole fraction array of Secondary fluid
        --------
        Returns: Array TSQ  [T_w, s_w, q_w, T_s, s_s, q_s]
        '''
        Matrix = np.zeros([Solution.shape[0], 6])
        AF = Input_list[0]
        SF = Input_list[1]
        z_w = Input_list[2]
        z_s = Input_list[3]

        for i in range(Solution.shape[0]):
            Matrix[i, 0:3] = RP.REFPROPdll(
                AF, "PH", "T;s;QMASS", MASS_BASE_SI, 0, 0, Solution[i, 2], Solution[i, 0], z_w).Output[0:3]
            Matrix[i, 3:] = RP.REFPROPdll(
                SF, "PH", "T;s;QMASS", MASS_BASE_SI, 0, 0, Solution[i, 3], Solution[i, 1], z_s).Output[0:3]

        T = np.vstack(Matrix)
        return T

    def Calculate_T_from_ph(self, Solution, Input_list):
        '''Calculate Temperatures curves of Working fluid and Secondary fluid 
        Parameters:
        ------    
        Solution : Output of the Solver
        Input_list : [0]:Working fluid ,[1]:Secondary fluid , [1]:Mole fraction array of Working fluid ,[1]:Mole fraction array of Secondary fluid
        ------
        Returns:Array of Temperatures [T_w , T_s]
        '''
        T_w = np.zeros(Solution.shape[0])
        T_s = np.zeros(Solution.shape[0])
        AF = Input_list[0]
        SF = Input_list[1]
        z_w = Input_list[2]
        z_s = Input_list[3]

        for i in range(Solution.shape[0]):
            T_w[i] = RP.REFPROPdll(
                AF, "PH", "T", MASS_BASE_SI, 0, 0, Solution[i, 2], Solution[i, 0], z_w).Output[0]
            T_s[i] = RP.REFPROPdll(
                SF, "PH", "T", MASS_BASE_SI, 0, 0, Solution[i, 3], Solution[i, 1], z_s).Output[0]

        T = np.transpose(np.vstack([T_w, T_s]))
        return T

    def generate_mole_fractions(self, n, alpha=[1, 1, 1]):
        """
        Generates a randomized grid of unique different mole fractions.
        Parameters
        ----------
        n : int
            Number of generating samples.
        alpha : list
            Destribution Weight
        Returns
        -------
        unique_y : Array
            random grid of different mole fractions.

        """

        # Generate random array with Dirichlet distribution
        mass_fractions = np.random.dirichlet(alpha, n)

        # Round to one digit
        rounded_y = np.round(mass_fractions, 1)
        rounded_y[:, 2] = 1-(rounded_y[:, 1]+rounded_y[:, 0])
        # Ensure uniqueness
        unique_y = np.unique(rounded_y, axis=0)

        return unique_y

    def get_phase(self, Solution, Input_list):
        '''Calculate Temperatures curves of Working fluid and Secondary fluid
        Solution : Output of the Solver
        Input_list : [0]:Working fluid , [1]:Mass fraction array of Working fluid
        '''
        phase_w = []
        AF = Input_list[0]
        z_w = Input_list[1]
        for i in range(Solution.shape[0]):
            phase_w.append(RP.REFPROPdll(AF, "PH", "PHASE", MASS_BASE_SI,
                           0, 0, Solution[i, 2], Solution[i, 0], z_w).hUnits)

        Phase = np.transpose(np.vstack([phase_w]))
        return Phase

    def Refprop_PH(self, material, mass_fraction, p, h):
        """
        gets Fluid Properties at given Pressure and Enthalpy

        Parameters
        ----------
        material : string
            Fluid.
        mass_fraction : array
            Mass fraction of the Fluid.
        p : float
            Pressure of Node [pa].
        h : TYPE
            specific Enthalpy of Node [j/kg].

        Returns
        -------
        TYPE
            Density[kg/m3] , Cp[j/kg*k]  , K [w/mk] , Pr[-] , T[k] , s[j/kg*k].

        """

        return RP.REFPROPdll(material, "PH", "D;CP;VIS;TCX;PRANDTL;T;s", MASS_BASE_SI, 0, 0, p, h, mass_fraction).Output[0:7]

    def Finish_sound(self):
        duration = 500  # milliseconds
        freq = 349  # Hz
        Beep(freq, duration)
        freq = 440  # Hz
        Beep(freq, duration)

    def Get_File_Name(self, variable_parameters2):
        """
        Converts the Variable Parameters to a Unique file name
        Parameters
        ----------
        variable_parameters2 : TYPE
            DESCRIPTION.

        Returns
        -------
        str
            Unique File name.

        """
        # pw 0 , ps 1 ,mw 2 , ms 3 ,  xa 4 , xb 5 , xc 6, din 7 , da 8
        return f'P_w[{variable_parameters2[0]:.1f}]_ms[{variable_parameters2[3]}]_mr[{variable_parameters2[3]/variable_parameters2[2]}]_xA[{variable_parameters2[4]}]_xB[{variable_parameters2[5]}]_Di[{variable_parameters2[7]}]_Da[{variable_parameters2[8]}]'

    def Save_Solution(self, Output, Filename, method='numpy'):
        Base_Dir = self.Save_Direction
        Folder_1 = 'Raw_Data'
        Folder_2 = os.path.join(Base_Dir, Folder_1)
        if not os.path.exists(Folder_2):
            os.makedirs(Folder_2)
        if method == 'numpy':
            file_name = f'{Filename}.npy'
            np.save(f'{Folder_2}\{file_name}', Output)
        elif method == 'excel':
            file_name = f'{Filename}.xlsx'
            DF = pd.DataFrame(Output, columns=self.Output_columns)
            DF.to_excel(f'{Folder_2}\{file_name}', index=False)

    def Generate_Parameter_Grid(self):
        '''Generate a Grid of Simulation Parameters Grid for Data Generation Process.
        Output:
           Array [ PW , PS , m_w , m_s , X_A , X_B, X_C , D_in , D_aus ] 
        '''
        problem = {
            'num_vars': 6,
            'names': ['X_A', 'X_B', 'X_C', 'm_r', 'm_s', 'p_w'],
            'bounds': [[0, 1],
                       [0, 1],
                       [0, 1],
                       self.m_dot_r_list,
                       self.m_dot_s_list,
                       self.p_w_list,
                       ],
            'dists': ['unif', 'unif', 'unif', 'unif', 'unif', 'unif']
        }
        p_sf_A = self.P_s
        D_in = self.D_w
        D_aus = self.D_s

        # Number of samples (input) created for S.A.; -> N * (D + 2)
        N_sample = 2**self.n_samples

        # Totale Saltelli-Sample-Verteilung
        N_total = N_sample * (problem['num_vars'] + 2)

        # Erstelle eine Stichprobe von Eingangsparametern für die Analyse der globalen Sensitivität eines mathematischen Modells. Das Saltelli-Sampling ist eine Methode der quasi-Monte-Carlo-Stichprobenziehung
        sample_set = sobol.sample(
            problem=problem, N=N_sample, calc_second_order=False)
        sample_set[:, :3] /= np.sum(sample_set[:, :3], axis=1, keepdims=True)
        sample_set[:, :3] = np.round(sample_set[:, :3], decimals=2)

        sample_set[:, 2] = 1 - (sample_set[:, 0] + sample_set[:, 1])
        # =======================Filters ============================
        # Drop the Parameters with X_C > 0.65
        sample_set = sample_set[sample_set[:, 2] <= 0.65]
        # ===========================================================
        # Create the Result Matrix
        PS_array = (
            np.zeros(sample_set.shape[0]) + p_sf_A).reshape([sample_set.shape[0], 1])
        D_in_array = (
            np.zeros(sample_set.shape[0]) + D_in).reshape([sample_set.shape[0], 1])
        D_aus_array = (
            np.zeros(sample_set.shape[0]) + D_aus).reshape([sample_set.shape[0], 1])
        m_r_array = sample_set[:, 3].reshape([sample_set.shape[0], 1])
        m_s_array = sample_set[:, 4].reshape([sample_set.shape[0], 1])
        m_w_array = m_s_array / m_r_array
        PW_array = sample_set[:, 5].reshape([sample_set.shape[0], 1])
        # Concatenate all arrays
        Result = np.concatenate((PW_array, PS_array, m_w_array, m_s_array,
                                sample_set[:, :3], D_in_array, D_aus_array), axis=1)
        DF_Result = pd.DataFrame(Result, columns=[
                                 'p_w', 'p_s', 'm_w', 'm_s', 'X_A', 'X_B', 'X_C', 'D_in', 'D_out'])
        return DF_Result


    def solve(self, variable_parameters):
        ''' solves the Heat and Pressure Drop ODEs with given Parameters.
        --------------------------------------------------------    
        Variable_parameter should be an Array containing these paramters with correct order:
        p_w [bar] , p_s[bar] , m_w[kg/s] , m_s[kg/s] , X_A[mol/mol] , X_B[mol/mol] , X_C[mol/mol] , Di[mm] , Da[mm]
        '''
        t0=time.time()
        # Load the Inputs and Convert the Units to SI
        # Inlet Pressure of Working Fluid [pa]
        P_w_in = variable_parameters[0] * 1e5
        # Outlet Pressure of Secondary Fluid [pa]
        P_s_out = variable_parameters[1] * 1e5
        # Mass flow rate of Working Fluid [kg/s]
        m_dot_w = variable_parameters[2]
        # Mass flow rate of Secondary Fluid [kg/s]
        m_dot_s = variable_parameters[3]
        x_A = variable_parameters[4]  # Mole Fraction A
        x_B = variable_parameters[5]  # Mole Fraction B
        x_C = variable_parameters[6]  # Mole Fraction C
        D_Tube_in = variable_parameters[7] * 1e-3  # Diameter of inner Pipe
        D_Shell = variable_parameters[8] * 1e-3  # Diameter of outter Pipe
        # if np.allclose(variable_parameters,[6.07756611e+00 ,1.50000000e+00 ,1.77579334e-03, 1.36019194e-02,
        #  3.30000000e-01 ,5.40000000e-01, 1.30000000e-01, 2.99968000e+01,
        #  5.00000000e+01]   ) :
         # return ([variable_parameters].append('Infinite Loop'))
        #Assign the Constants
        z_s = [1]
        z_w = np.array([x_A, x_B, x_C])
        t_Tube = self.t_Tube
        K_pipe = self.K_pipe
        ep = self.ep
        # get the species from fluid composition string
        species = self.AF.split('*')
        self.num_species = len(species)
        x = np.linspace(0, self.L, int(self.L/self.dx))  # Create the Grid
        D_Tube_out = D_Tube_in+2*t_Tube  # outer Diameter of Tube
        State_list = []  # a List to gather information in Solve_ivp process
        Simulation_Setup_and_Status = list(variable_parameters)
        
        A_q_in = 0.25*np.pi * D_Tube_in**2    # Cross Section of Tube
        A_q_out = 0.25*np.pi * (D_Shell**2 - D_Tube_out**2)  # Cross Section of Shell
        
        # Calculate the inlet Enthalpy of AF and Secondarty Enthaly of SF
        T_w_sat_V = RP.REFPROPdll(
            self.AF, "PQ", "T", MASS_BASE_SI, 0, 0, P_w_in, 1.0, z_w).Output[0]

        h_s_sat_V = RP.REFPROPdll(
            self.SF, "PQ", "h", MASS_BASE_SI, 0, 0, P_s_out, 1.0, z_s).Output[0]

        T_w_in = T_w_sat_V + self.Superheating_Temperature
        T_s_out = T_w_sat_V

        h_w_in, s_w_in , rho_w ,cp_w = RP.REFPROPdll(
            self.AF, "PT", "h,s,D,CP", MASS_BASE_SI, 0, 0, P_w_in, T_w_in, z_w).Output[0:4]
        h_s_out, s_s_out ,rho_s,cp_s = RP.REFPROPdll(
            self.SF, "PT", "h,s,D,CP", MASS_BASE_SI, 0, 0, P_s_out, T_s_out, z_s).Output[0:4]
        #print(f'TW_in={T_w_in}     , TS_aus = {T_s_out} , {z_w}\nT_W_in_th={T_w_in_test}    ,T_s_out_th={T_s_out_test}')
        # print(f'h_w_in:{h_w_in},h_s_out:{h_s_out},P_w_in:{P_w_in},P_s_out:{P_s_out}')
        c_r = (cp_w*m_dot_w)/(cp_s*m_dot_s)
        #calculate Velocity in both pipes
        u_s = m_dot_s/(rho_s * A_q_out)
        u_w = m_dot_w/(rho_w * A_q_in)
        if abs(rho_s) > 1e4 or abs(rho_w) > 1e4:
            raise ValueError('Density is diverged rho_s:{rho_s}   rho_w:{rho_w}')
        if u_s > self.u_s_limit :
            t1=time.time()
            time_elapsed = t1-t0
            Status = 'Velocity of secondary fluid exceeds limit'
            
            Simulation_Setup_and_Status.extend([u_w,u_s,0.0,c_r,0.0,0.0,Status,time_elapsed])
            return Simulation_Setup_and_Status

        if u_w > self.u_w_limit :
            t1=time.time()
            time_elapsed = t1-t0
            Status = 'Velocity of working fluid exceeds limit'
            Simulation_Setup_and_Status.extend([u_w,u_s,0.0,c_r,0.0,0.0,Status,time_elapsed])
            return Simulation_Setup_and_Status
        
        boundary_condition = [h_w_in, h_s_out, P_w_in, P_s_out]
        # Assign Input Variables  of ODE solver function
        Input_Array = [self.AF, self.SF, z_s, z_w, m_dot_w, m_dot_s,
                       D_Tube_in, D_Tube_out, D_Shell, self.dx, ep, K_pipe ,A_q_in , A_q_out ]
        self.IA = Input_Array
        # Define Events and their Inputs:

        E1_Input_list = [Input_Array[0], Input_Array[1],
                         Input_Array[2], Input_Array[3], self.DT_tolerance]

        
        # Event to convergence when the Temperature difference is smaller than 1k
        def E1(t, x): return self.Event_DT(t, x, E1_Input_list)
        def E3(t, x): return self.Event_Phase_Change(t, x, h_s_sat_V)
        # Event to limit the pressure difference of working fluid
        def E4(t, x): return self.Event_dp_w(t, x, P_w_in)
        # Event to limit the pressure difference of secondary fluid
        def E5(t, x): return self.Event_dp_s(t, x, P_s_out)

        E1.terminal = True
        E3.terminal = True
        E4.terminal = True
        E5.terminal = True
        max_num_iterations = 2000
        max_step_size = 1.0

        try:
            if self.verbose_level > 1 :
                print(f'current Paramter :{variable_parameters} ')
            solution = solve_ivp(
                lambda x, y: self.HeatExchangerModel(
                    x, y, State_list, Input_Array),
                (0, self.L),
                boundary_condition,
                t_eval=x,
                method=self.ODE_solver_method,
                atol=self.ODE_tolerance,
                rtol=self.ODE_R_tolerance,
                events=(E1, self.Event_Freeze, E3,
                        E4, E5),
                max_step=max_step_size,
                dense_output=False,
            )

            solution_message = solution.message

            # Events status
            Conv_Event = solution.t_events[0].size  # DT Event , Converged!
            Div_Event_2 = solution.t_events[1].size  # Freeze
            Div_Event_3 = solution.t_events[2].size  # Phase change in SF
            Div_Event_4 = solution.t_events[3].size  # Pressure loss is too high in WF
            Div_Event_5 = solution.t_events[4].size  # Pressure loss is too high in SF
            
            # Stack the Solution in an Array
            output = np.vstack([solution.t, solution.y])
            #Postprocessing 
            self.current_output = output
            self.current_state_list = State_list
            _,_,dp_w,dp_s = self.Post_pressure_drop(output)
            Div_Event_6,Div_Event_7,self.first_law,Q_h = self.Post_First_law(output,[m_dot_w,m_dot_s])
            Div_Event_8,self.second_law = self.Post_Second_law(output,[s_w_in,s_s_out],[m_dot_w,m_dot_s],[z_w,z_s])
            t1=time.time()
            if self.verbose_level == 1:
                print(solution_message)
            elif self.verbose_level == 2:
                print(solution_message)
                print(f'Triggered Events ({t1-t0:.2e} s) : DT:{Conv_Event}  , neg_h:{Div_Event_2} , phase_change_s:{Div_Event_3} ,Q_h:{Div_Event_7}\n , dp_w:{Div_Event_4} , dp_s:{Div_Event_5} ,1.Hs:{Div_Event_6} , 2.Hs:{Div_Event_8}')
                
            Event_Results = np.array([Conv_Event, Div_Event_2, Div_Event_3, Div_Event_4, Div_Event_5,
                                     Div_Event_6, Div_Event_7, Div_Event_8])  # Stack the Events in an Array
            Phase_w = self.Phase_Analyse(State_list)
            Status = self.Evaluate_Results(Event_Results, Phase_w)
            t1=time.time()
            time_elapsed = t1-t0
            # Update the Result list with Status
            #'p_w', 'p_s', 'm_w', 'm_s', 'X_A', 'X_B', 'X_C', 'D_in', 'D_out','u_w','u_s','Q_h','c_r','dp_w','dp_s', 'Result'
            Simulation_Setup_and_Status.extend([u_w,u_s,Q_h,c_r,dp_w,dp_s,Status,time_elapsed])
            
            # Save the Output results
            if self.Save_Outputs == True and Status == 'Valid solution - All phases present' or Status == 'Valid solution - No subcooled liquid':
                raw_file_name = self.Get_File_Name(variable_parameters)
                self.Save_Solution(output.T, raw_file_name)

            # return output.T , State_list ,Simulation_Setup_and_Status
            return Simulation_Setup_and_Status

        except Exception as e:
            #print(f"Error in simulation: {e}")
            self.current_output = np.nan
            import traceback
            traceback.print_exc()
            raise

    def Post_First_law(self, solution, m_dot):
        '''Thermodynamical Analysis of the System
        
        First law of thermodynamics , conservation of Energy
        
        Parameters: 
            Solution : ODE Solution Output 
            m_dot : Mass flow rate list , [m_dot_w , m_dot_s]
            
        Returns:
            Evaluation_result_Firstlaw :int, is First Law of thermodynamics violated ?
            Evaluation_result_Qh :int, has Heat transfer rate exceeded the boundary?
            residual : residuals of Energy balance
            Q_dot_w : Total Heat flow
        '''
        _,h_w,h_s,p_w,p_s = solution
        
        m_dot_w = m_dot[0]
        m_dot_s = m_dot[1]

        h_w_in = h_w[0]
        h_s_aus = h_s[0]

        h_w_aus = h_w[-1]
        h_s_in =h_s[-1]
        # Calculate Enthalpy flow of both pipes
        Q_dot_w = abs(m_dot_w * (h_w_aus - h_w_in))
        Q_dot_s = abs(m_dot_s*(h_s_aus - h_s_in))
        residual =  (Q_dot_s-Q_dot_w)
        
        #Evaluate Conservation of Energy in system
        if abs(residual) < 1e-6:
            Evaluation_result_Firstlaw = 0 
        else :  Evaluation_result_Firstlaw =1
        
        #Evaluate the total Heat flow in Heat Exchanger
        if Q_dot_w >= 1000 and Q_dot_w <= 10000 :
            Evaluation_result_Qh = 0
        else : Evaluation_result_Qh = 1
        
        return Evaluation_result_Firstlaw,Evaluation_result_Qh,residual,Q_dot_w
    
    def Post_Second_law(self, solution, s_in, m_dot, z):
        '''Thermodynamical Analysis of the System
        
        Second law of thermodynamics ,net Entropy production rate should be positive
        
        Parameters : 
            Solution : ODE Solution Output
            S_in : list [s_w_in , s_w_out]
            m_dot : list [m_dot_w , m_dot_s]
            z  : list [z_w , z_s]
            
        Returns:
           Evaluation_result : is S_dot_irr negative? 
           S_dot_irr : net entropy production rate
        '''
        _,h_w,h_s,p_w,p_s = solution
        
        m_dot_w = m_dot[0]
        m_dot_s = m_dot[1]
        z_w = z[0]
        z_s = z[1]

        s_w_aus = RP.REFPROPdll(
            self.AF, "Ph", "s", MASS_BASE_SI, 0, 0,p_w[-1] , h_w[-1], z_w).Output[0]
        s_s_in = RP.REFPROPdll(
            self.SF, "Ph", "s", MASS_BASE_SI, 0, 0,p_s[-1] , h_s[-1], z_s).Output[0]

        s_w_in = s_in[0]
        s_s_aus = s_in[1]

        # Calculate Entropy Production rate of both pipes
        S_dot_w = (m_dot_w * (s_w_aus - s_w_in))
        S_dot_s = (m_dot_s*(s_s_aus - s_s_in))
        # Calculate Net Entropy Production rate of system
        S_dot_irr = S_dot_s + S_dot_w
        # Evaluate the Second law ,Invalid for negative Net Entropy Production rate
        if S_dot_irr > 0: Evaluation_result = 0 
        else:Evaluation_result = 0 
        return Evaluation_result,S_dot_irr 
    
    def Post_pressure_drop(self,solution):
        '''Pressure drop in Working Fluid  should be smaller than 0.1*P0 
        
           Pressure drop in Secondary Fluid should be smaller than 70kpa
           
           Parameter: Solution output of ODE
           
           return :
             Evaluation_dpw : has Pressure Drop in AF Exceeded the limit ?
             Evaluation_dps : has Pressure Drop in SF Exceeded the limit ?
             dp_w : Pressure Drop in AF along pipe
             dp_s : Pressure Drop in SF along pipe
        '''
        _,h_w,h_s,p_w,p_s = solution
        
        #Calculate Pressure Drop along the Pipe
        dp_w = abs(p_w[-1] - p_w[0])
        dp_s = abs(p_s[-1] - p_s[0])
        
        #Define the Limits based on the Standard Design
        dp_w_limit = p_w[0]*0.1
        dp_s_limit = 70e3
        
        #Check if the Pressure drop is under the desirable range
        if dp_w_limit - dp_w > 0 :
            Evaluation_dpw = 0
        else: Evaluation_dpw = 1
        
        #Check if the Pressure drop is under the desirable range
        if dp_s_limit - dp_s > 0 :
            Evaluation_dps = 0
        else: Evaluation_dps = 1
        
        return  Evaluation_dpw, Evaluation_dps, dp_w,   dp_s
    
    def Evaluate_Results(self, Event_Results, Phase_w):
        '''Evaluates the Simulation results and returns the Status of the Simulation
        Possible Events: Convergence_DT[0], negative_h_s[1], phase_change_s[3], u_w[4], Q_h[5], dp_w[6], dp_s[7], u_s[8], First law [9], Second law [10]
        '''
        #[dt , fr , -h , dp_w , dp_s , 1.hs , Q_h , 2.hs]
        result_info = {
            0: 'Converged successfully',
            1: 'Negative enthalpy for secondary fluid',
            2: 'Phase change in secondary fluid',
            3: 'Pressure drop in working fluid exceeds limit',
            4: 'Pressure drop in secondary fluid exceeds limit',
            5: 'First law violation',
            6: 'Heat transfer rate out of desired range',
            7: 'Second law violation'
        }
        sum_events = np.sum(Event_Results)
        if Event_Results.size == 0 or np.all(Event_Results == 0):
            result = 'Not converged - in L'
        elif Event_Results[0] == 1 and sum_events == 1:
            if Phase_w[0] == 1 and Phase_w[1] == 1 and Phase_w[2] == 1:
                result = 'Valid solution - All phases present'
            elif Phase_w[0] == 1 and Phase_w[1] == 1 and Phase_w[2] == 0:
                result = 'Valid solution - No subcooled liquid'
            else:
                result = 'Valid solution - Only superheated gas'
        else:
            Event_Results[0] = 0
            first_error_index = np.where(Event_Results == 1)[0][0]
            result = f'Not converged - {result_info[first_error_index]}'

        return result

     #!!! Events

    def Event_First_law(self, x, y, h_in, m_dot):
        '''Thermodynamical Analysis of the System
        Termination Condiation when the First law of thermodynamics is being violated by small tolerance

        '''
        m_dot_w = m_dot[0]
        m_dot_s = m_dot[1]

        h_w_in = h_in[0]
        h_s_aus = h_in[1]

        h_w_aus = y[0]
        h_s_in = y[1]
        # Calculate Enthalpy flow of both pipes
        Q_dot_w = abs(m_dot_w * (h_w_aus - h_w_in))
        Q_dot_s = abs(m_dot_s*(h_s_aus - h_s_in))
        residual = 1e-10 - (Q_dot_s-Q_dot_w)
        #print(f'Q_h - Q_c :{residual}')
        return residual

    def Event_Second_law(self, x, h, s_in, m_dot, z):
        '''Thermodynamical Analysis of the System
        Termination Condiation when net Entropy production rate is negative

        '''
        m_dot_w = m_dot[0]
        m_dot_s = m_dot[1]
        z_w = z[0]
        z_s = z[1]

        s_w_aus = RP.REFPROPdll(
            self.AF, "Ph", "s", MASS_BASE_SI, 0, 0, h[0], h[2], z_w).Output[0]
        s_s_in = RP.REFPROPdll(
            self.SF, "Ph", "s", MASS_BASE_SI, 0, 0, h[3], h[1], z_s).Output[0]

        s_w_in = s_in[0]
        s_s_aus = s_in[1]

        # Calculate Entropy Production rate of both pipes
        S_dot_w = (m_dot_w * (s_w_aus - s_w_in))
        S_dot_s = (m_dot_s*(s_s_aus - s_s_in))
        # Calculate Net Entropy Production rate of system
        S_dot_irr = S_dot_s + S_dot_w
        # print(f'S_dot_irr:{S_dot_irr}')
        # Stop by negative Net Entropy Production rate
        return -S_dot_irr + 1e-10

    def Event_Q_h(self, x, y, h_w_in, m_dot_w):
        '''Divergence Condition
        Termination Condiation when Heat flow is greater than 10kw or smaller than 1kw
        '''
        # Calculate the Heat transfer rate :Q_dot = m_dot*dh
        Q_dot = abs(m_dot_w * (y[0]-h_w_in))
        if x > 1.0:
            #print(f'Q_dot:{Q_dot} at {x}')
            if Q_dot < 1000:
                return -Q_dot  # Negative

            elif Q_dot > 10000:
                return -Q_dot  # Negative
            else:
                return 10000 - Q_dot  # Positive

        else:
            return Q_dot + 1e-1   # Positive

    def Event_DT(self, x, h, Input_list):
        '''Convergence Condition
        Termination Condition when Temperature difference is 0.5 K '''
        # h=[dh_w,dh_s,dp_w,dp_s]
        AF = Input_list[0]
        SF = Input_list[1]
        z_s = Input_list[2]
        z_w = Input_list[3]
        DT_tolerance = Input_list[4]

        T_s = RP.REFPROPdll(SF, "PH", "T", MASS_BASE_SI, 0,
                            0, h[3], h[1], z_s).Output[0]
        T_w = RP.REFPROPdll(AF, "PH", "T", MASS_BASE_SI, 0,
                            0, h[2], h[0], z_w).Output[0]
        temperature_difference = abs(T_w - T_s)
        return temperature_difference - DT_tolerance

    def Event_Freeze(self, x, h):
        '''Divergence Condition
        Termination Condition when Enthalpy of Secondary Fluid is negative
        When This Termination Event is triggered the Solution is disqualified and not being used
        '''
        # h=[dh_w,dh_s,dp_w,dp_s]
        return h[1]

    def Event_Phase_Change(self, x, h, h_s_sat_V):
        '''Divergence Condition
        Disqualify the Simulation due to phase change in Secondary fluid , phase change occurs when Enthalpy of the Secondary fluid is higher than Enthalpy of Saturated of Vapor .
        '''

        # h=[dh_w,dh_s,dp_w,dp_s]
        h_s = h[1]  # Enthalpy of secondary fluid

        if x > 0:
            return h_s_sat_V - h_s
        else:
            return h_s

    def Event_dp_w(self, x, y, p0):
        '''Divergence Condition
        Pressure Loss in Working Fluid should be smaller than 0.1*[P0]bar'''
        # y=[dh_w,dh_s,dp_w,dp_s]
        dp = abs(y[2] - p0)
        dp_limit = p0*0.1
        #print(f'dp:{dp}   dp_limit:{dp_limit}')
        return dp_limit - dp

    #!! Problematic

    def Event_U_s(self, x, y, Input_list):
        '''Divergence Condition
        Disqualify the Simulation when the velocity of Secondary in Shell is larger than 1.5 m/s
        #Recommanded Maximum Velocity in shell for Water 1.5m/s
        #R.K. Sinnott, Coulson & Richardson’s Chemical Engineering Design, Volume 6, 3rd Edition, Butterworth-Heinemann
        '''
        # y=[dh_w,dh_s,dp_w,dp_s]
        SF = Input_list[0]  # Secondary fluid
        z_s = Input_list[1]  # get mole fractions of secondary fluid
        D_in = Input_list[2]  # inner Diameter of Ring
        D_a = Input_list[3]  # outer Diameter of Ring
        m_dot_s = Input_list[4]  # Mass flow rate of Secondary fluid
        # Calculate the Cross section of Ring
        A_q = (np.pi / 4) * (D_a**2 - D_in**2)

        # calculate density of secondary fluid
        rho = RP.REFPROPdll(SF, "PH", "D", MASS_BASE_SI, 0,
                            0, y[3], y[1], z_s).Output[0]
        u_s = m_dot_s / (rho * A_q)
        # print(u_s)
        if x > 0:
            # Force Sign change if the u_s is greater than 1.5 to call this Event
            return 1.5 - u_s
        else:
            # Return an arbitary positive value to prevent a faulty start.
            return 0.1

    def Event_U_w(self, x, h, Input_list):
        '''Divergence Condition
        Disqualify the Simulation when the velocity of working fluid is above 10m/s
        #Recommanded Maximum Velocity in Tube for high pressure Gases 10m/s
        #R.K. Sinnott, Coulson & Richardson’s Chemical Engineering Design, Volume 6, 3rd Edition, Butterworth-Heinemann
        '''
        # h=[dh_w,dh_s,dp_w,dp_s]
        AF = Input_list[0]

        z_w = Input_list[3]
        D_in = Input_list[4]
        m_dot_w = Input_list[5]
        A_q = (np.pi/4)*D_in**2

        rho = RP.REFPROPdll(AF, "PH", "D", MASS_BASE_SI, 0,
                            0, h[2], h[0], z_w).Output[0]
        u_w = m_dot_w/(rho*A_q)
        if x > 0:
            # Force sign change if the u_w is greater than 10 to call this event.
            # print(u_w)
            return 10 - u_w
        # Return an arbitary positive value to prevent a faulty start.
        else:
            return 0.1

    def Event_dp_s(self, x, y, p0):
        '''Divergence Condition
        Pressure Loss in Working Fluid should be smaller than 0.1bar
        #Recommanded Pressure Drop for Water 70kpa  [0.7 bar]
        #R.K. Sinnott, Coulson & Richardson’s Chemical Engineering Design, Volume 6, 3rd Edition, Butterworth-Heinemann
        '''
        # y=[dh_w,dh_s,dp_w,dp_s]
        dp = abs(y[3] - p0)
        return 70e3 - dp

    Event_DT.terminal = True
    Event_Freeze.terminal = True
    Event_Phase_Change.terminal = True
    #!!! ODE

    def HeatExchangerModel(self, x, y, State_list, Input_Array):
        ''' function for the the Diffrential Equation of Heat and Pressure loss of Heat Exchanger '''
        # ============Integration from 0 to L=================
        # Get the Data from
        P_w = y[2]
        P_s = y[3]
        h_w = y[0]
        h_s = y[1]
        Q_s = -1
        Q_w = -1
        AF = Input_Array[0]
        SF = Input_Array[1]
        z_s = Input_Array[2]
        z_w = Input_Array[3]
        m_dot_w = Input_Array[4]
        m_dot_s = Input_Array[5]
        D_Tube_in = Input_Array[6]
        D_Tube_out = Input_Array[7]
        D_Shell = Input_Array[8]
        #print(f"P_w: {P_w/1000:.4f}, P_s: {P_s/1000:.4f}, h_w: {h_w/1000:.4f}, h_s: {h_s/1000:.4f}, AF: {AF}, SF: {SF}, z_s: {z_s}, z_w: {z_w}, m_dot_w: {m_dot_w}, m_dot_s: {m_dot_s}, D_Tube_in: {D_Tube_in}, D_Tube_out: {D_Tube_out}, D_Shell: {D_Shell}")

        Dh = D_Shell - D_Tube_out

        dx = Input_Array[9]
        ep = Input_Array[10]
        K_pipe = Input_Array[11]
        A_q_in = Input_Array[12]
        A_q_out = Input_Array[13]
        # Determine the Phase of the Matters
        phase_w = RP.REFPROPdll(
            AF, "PH", "PHASE", MASS_BASE_SI, 0, 0, P_w, h_w, z_w).hUnits
        phase_s = RP.REFPROPdll(
            SF, "PH", "PHASE", MASS_BASE_SI, 0, 0, P_s, h_s, z_s).hUnits
        # Determine the Quality of the Matter if it's in two phase state and read Properties from Refprop with Quality and Pressure]
        if phase_s == 'Two-phase':  # Phase Change in Secendary Fluid
            # Mix Phase State
            # calculate the Quality of Fluid
            Q_s = RP.REFPROPdll(SF, "PH", "QMASS", MASS_BASE_SI,
                                0, 0, P_s, h_s, z_s).Output[0]
            #print(f'{self.SF} , {phase_s} , Q:{Q_s}')
            # Read the Fluid Properties at current Quality and Pressure
            rho_s, rho_L_s, rho_V_s, cp_L_s, cp_V_s, K_L_s, K_V_s, vis_L_s, vis_V_s, Pr_L_s, Pr_V_s, T_s, h_s, STN_s, s_s = RP.REFPROPdll(
                SF, "PQ", "D,DLIQ,DVAP,CPLIQ,CPVAP,TCXLIQ,TCXVAP,VISLIQ,VISVAP,PRANDTLLIQ,PRANDTLVAP,T,h,STNLIQ,s", MASS_BASE_SI, 0, 0, P_s, Q_s, z_s).Output[0:15]
            # Calculate the Properties with Quality
            cp_s = (cp_V_s - cp_L_s)*Q_w + cp_L_s
            K_s = (K_V_s-K_L_s)*Q_s + K_L_s
            Pr_s = (Pr_V_s-Pr_L_s)*Q_s + Pr_L_s
            vis_s = (vis_V_s-vis_L_s)*Q_s + vis_L_s
            u_s = m_dot_s/(rho_s*A_q_out)
            Re_s = u_s*rho_s*Dh/vis_s
            # Calculate the Heat tranself.SFer Coefficient from Cavallini Correlation
            alpha_s = cond_Cavallini(
                m_dot_s, Q_s, Dh, rho_L_s, rho_V_s, vis_L_s, vis_V_s, K_L_s, cp_L_s)
            Nu_s = alpha_s * Dh / K_s
            f_s = 0
            # Calculate the Pressure drop in Pipe using the Friedel Correlation
            dp_s = two_phase_dP(m=m_dot_s, x=Q_s, rhol=rho_L_s, D=Dh, L=dx,
                                 rhog=rho_V_s, mul=vis_L_s, mug=vis_V_s, sigma=STN_s, Method='Friedel')
        else:  # Subcooled Liquid or Sueprheated Phase
            # print(f'{phase_s}')
            # Read the Fluid Properties at current Enthalpy and Pressure
            rho_s, cp_s, vis_s, K_s, Pr_s, T_s, s_s = self.Refprop_PH(
                SF, z_s, P_s, h_s)  # Secondary Fluid Material Properties
            # Calculate Velocity of Fluid
            u_s = m_dot_s/(rho_s*A_q_out)
            # Calculate the Reynold Number of Fluid
            Re_s = u_s*rho_s*Dh/vis_s
            # Calculate the Nusselt Number based on Reynold number using Gnielinski Correlation for Turbulent flows
            if Re_s < 3000:
                f_s = 64/Re_s
                Nu_s = 3.657+1.2*(D_Shell / D_Tube_out)**(-0.8)
            else:
                # Calculate Darcy friction factor using Colebrook White Equation for Turbulent flows
                f_s = frict.Colebrook(Re_s, ep/Dh)
                Nu_s = Nusselt(Re_s, Pr_s, fd=f_s, Method="Gnielinski")
                Nu_s = Nu_s * 0.86 * (D_Shell / D_Tube_out)**(-0.16)

            # Calculate the Heat tranself.SFer Coefficient from Nussselt number
            alpha_s = Nu_s * K_s / Dh
            # Calculate the Pressure drop in Pipe using Darcy Weisbach Equation
            dp_s = f_s*rho_s*0.5*(1/Dh)*u_s**2

        if phase_w == 'Two-phase':  # Phase Change in Work fluid
            # calculate the Quality of Fluid
            Q_w = RP.REFPROPdll(AF, "PH", "QMASS", MASS_BASE_SI,
                                0, 0, P_w, h_w, z_w).Output[0]
            #print(f'{self.AF} , {phase_w} , Q:{Q_w}')
            # Read the Fluid Properties at current Quality and Pressure
            rho_w, rho_L_w, rho_V_w, cp_L_w, cp_V_w, K_L_w, K_V_w, vis_L_w, vis_V_w, Pr_L_w, Pr_V_w, T_w, h_w, STN_w, s_w = RP.REFPROPdll(
                AF, "PQ", "D,DLIQ,DVAP,CPLIQ,CPVAP,TCXLIQ,TCXVAP,VISLIQ,VISVAP,PRANDTLLIQ,PRANDTLVAP,T,h,STNLIQ,s", MASS_BASE_SI, 0, 0, P_w, Q_w, z_w).Output[0:15]
            # Calculate the Heat tranself.SFer Coefficient from Cavallini Correlation
            alpha_w = cond_Cavallini(
                m_dot_w, Q_w, Dh, rho_L_w, rho_V_w, vis_L_w, vis_V_w, K_L_w, cp_L_w)
            # Calculate the Pressure drop in Pipe using the Friedel Correlation
            dp_w = -two_phase_dP(m=m_dot_w, x=Q_w, rhol=rho_L_w, D=D_Tube_in, L=dx,
                                 rhog=rho_V_w, mul=vis_L_w, mug=vis_V_w, sigma=STN_w, Method='Friedel')
            # Calculate the Properties with Quality
            cp_w = (cp_V_w - cp_L_w)*Q_w + cp_L_w
            K_w = (K_V_w-K_L_w)*Q_w + K_L_w
            Pr_w = (Pr_V_w-Pr_L_w)*Q_w + Pr_L_w
            vis_w = (vis_V_w-vis_L_w)*Q_w + vis_L_w
            u_w = m_dot_w/(rho_w*A_q_in)
            Re_w = u_w*rho_w*D_Tube_in/vis_w
            Nu_w = alpha_w * D_Tube_in / K_w
            f_w = 0
        else:  # Subcooled Liquid or Sueprheated Phase
            # Read the Fluid Properties at current Enthalpy and Pressure
            rho_w, cp_w, vis_w, K_w, Pr_w, T_w, s_w = self.Refprop_PH(
                AF, z_w, P_w, h_w)  # Workfluid Fluid Material Properties
            # Calculate Velocity of Fluid
            u_w = m_dot_w/(rho_w*A_q_in)
            # Calculate the Reynold Number of Fluid
            Re_w = u_w*rho_w*D_Tube_in/vis_w
            # Calculate the Nusselt Number based on Reynold number using Gnielinski Correlation for Turbulent flows
            if Re_w < 3000:
                f_w = 64/Re_w
                Nu_w = 3.657
            else:
                # Calculate Darcy friction factor using Colebrook White Equation for Turbulent flows
                f_w = frict.Colebrook(Re_w, ep/D_Tube_in)
                Nu_w = Nusselt(Re_w, Pr_w, fd=f_w, Method="Gnielinski")
            # Calculate the Heat tranself.SFer Coefficient from Nussselt number
            alpha_w = Nu_w * K_w / D_Tube_in
            # Calculate the Pressure drop in Pipe using Darcy Weisbach Equation
            dp_w = -f_w*rho_w*0.5*(1/D_Tube_in)*u_w**2
        #print(f'x:{x:.3},z_w:{z_w} , T_w:{T_w-273.15 :.3}   ,T_s:{T_s-273.15 :.3f} , h_w:{h_w/1000 :.3f} ,h_s:{h_s/1000 :.3f}  , p_w:{P_w/1000 :.3f} ,p_s:{P_s/1000 :.3f} ' )
        # Calculate the total Heat Resistance * dx
        R_total_dx = (alpha_w*np.pi*D_Tube_in)**-1 + np.log((D_Tube_out) /
                                                            D_Tube_in) / (2*np.pi*K_pipe) + (alpha_s*np.pi*(D_Tube_out))**-1
        # Calculate the Enthalpy Change to next Node

        dh_w = -(T_w-T_s)/(m_dot_w * R_total_dx)  # Heat ODE for Working Fluid
        dh_s = -(T_w-T_s)/(m_dot_s * R_total_dx)  # Heat ODE for Secondary Fluid
        if self.verbose_level == 3:
            print(f'x:{x} : dh_w:{dh_w} , dh_s:{dh_s} , dp_w:{dp_w} , dp_s:{dp_s}')
        State_list.append([x, T_w-273.15, s_w, h_w, P_w, rho_w, cp_w, K_w, vis_w, Pr_w, u_w, Re_w, Nu_w, alpha_w, R_total_dx, dh_w, f_w, Q_w,
                          phase_w, T_s-273.15, s_s, h_s, P_s, rho_s, cp_s, K_s, vis_s, Pr_s, u_s, Re_s, Nu_s, alpha_s, R_total_dx, dh_s, f_s, Q_s, phase_s])
        
        return [dh_w, dh_s, dp_w, dp_s]

        
    def Phase_Analyse(self, Monitoring_Process):
        '''Checks the Monitoring List and returns the existing phases

        return :Array [Superheated Gas , Two-Phase , Subcooled Liquid]
        '''
        J = np.vstack(Monitoring_Process)
        Phase_list = J[:, 18]
        phase_Matrix = np.zeros(3)
        if 'Superheated gas' in Phase_list:
            phase_Matrix[0] = 1
        else:
            phase_Matrix[0] = 0
        if 'Two-phase' in Phase_list:
            phase_Matrix[1] = 1
        else:
            phase_Matrix[1] = 0
        if 'Subcooled liquid' in Phase_list:
            phase_Matrix[2] = 1
        else:
            phase_Matrix[2] = 0
        return phase_Matrix

    def get_state_list(self):
        '''Get the Detailed State of Simulation that contains all the Transport and Thermodynamic Properties of Working and Secondary Fluids

        return : Dataframe of State list
        '''
        state_list = pd.DataFrame(self.current_state_list, columns=['x', 'T_w', 's_w', 'h_w', 'p_w', 'rho_w', 'cp_w', 'k_w', 'vis_w', 'Pr_w', 'u_w', 'Re_w', 'Nu_w', 'alpha_w', 'R_total',
                                  'dh_w', 'f_w', 'q_w', 'phase_w', 'T_s', 's_s', 'h_s', 'p_s', 'rho_s', 'cp_s', 'k_s', 'vis_s', 'Pr_s', 'u_s', 'Re_s', 'Nu_s', 'alpha_s', 'R_total', 'dh_s', 'f_s', 'q_s', 'phase_s'])
        state_list = state_list.sort_values(by='x')
        state_list = state_list.drop_duplicates(subset='x', keep='last')
        return state_list

    def solve_parallel(self, param_values):
        num_processes = multiprocessing.cpu_count()
        chunk_size = len(param_values) // num_processes
        self.cache_df = pd.DataFrame(columns=[
                                     'D_in ', 'D_aus', 'm_w', 'm_s', 'T_w', 'T_s', 'P_w', 'P_s', 'x_A', 'x_B', 'z_s', 't', 'k', 'ep', 'RS'])

        if chunk_size > 1:
            param_chunks = [param_values[i:i + chunk_size]
                            for i in range(0, len(param_values), chunk_size)]
        else:
            param_chunks = [param_values]

        total_simulations = len(param_values)
        simulations_completed = 0

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_processes) as executor:
            all_results = []

            for chunk in param_chunks:
                futures = []

                for params in chunk:
                    self.check_memory_usage()
                    future = executor.submit(self.solve, params)
                    futures.append(future)
                    gc.collect()

                # Wait for all tasks in the current batch to complete with a timeout
                try:
                    concurrent.futures.wait(
                        futures, timeout=360)  # Timeout set to 6 minutes
                except concurrent.futures.TimeoutError:
                    print("Simulation exceeded the time limit and was terminated.")
                    return []
                
                # Get results from completed tasks
                for future, params in zip(futures, chunk):
                    try:
                        result = future.result()
                        all_results.append(result)
                        simulations_completed += 1
                        progress_percentage = (
                            simulations_completed / total_simulations) * 100
                        gc.collect()
                        if self.verbose_level > 0:
                            print(
                                f"Progress: {progress_percentage:.1f}% completed")

                    except Exception as e:
                        print(f"Error in simulation: {e}")
                        gc.collect()

        if self.verbose_level > 0:
            print("Progress: 100.00% completed\nSimulation is finished!")

        return all_results

    def solve_serial(self, Parameter_list):
        '''Solves the Physical Model with Serial Processing (recommanded to be used with small batches , unreliable use it for debugging)'''
        results = []
        # Convert the DataFrame to numpy array
        if type(Parameter_list) == pd.core.frame.DataFrame:
            Parameter_list = Parameter_list.to_numpy()

        # Serial Processing:
        for i in range(Parameter_list.shape[0]):
            self.current_parameter = Parameter_list[i]
            setup_and_status = self.solve(Parameter_list[i])
            # Abort the Processing to prevent Memory Overload
            self.check_memory_usage()
            # update the results list
            results.append(setup_and_status)

        return results

    def simulate(self, num_batches=1000, computing_method='parallel'):
        '''performs a Simulation with automatic Parameter Grid Generation by solving batches of Parameter Grid.

           Parameters:
               num_batches : Number of Batches
               computing_method : Parallel computing or Serial computing 


           Returns: None 
           Saves an Excel Report of Simulation Results 
        '''
        Save_folder = self.Save_Direction
        Prameter_Batches_Path = os.path.join(
            self.cache_folder, 'Parameter_Batches.pkl')
        # Generate the Batches
        if not os.path.exists(Prameter_Batches_Path):
            Parameter_Grids = self.Generate_Parameter_Grid()
            Parameter_Batches = np.array_split(
                Parameter_Grids.to_numpy(), num_batches)
            joblib.dump(Parameter_Batches, Prameter_Batches_Path)
        else:
            Parameter_Batches = joblib.load(Prameter_Batches_Path)

        # Run the Simulation for every Batches and Save results as Cache files
        for batch_number, Parameters in enumerate(Parameter_Batches):
            Save_Batches_Path = os.path.join(
                self.cache_folder, f'Results_{batch_number}.pkl')
            if batch_number in self.skip_batch_list:
                continue
            if not os.path.exists(Save_Batches_Path):
                overall_progress = np.round(100*(batch_number/num_batches), 1)
                print('--------------------------------------------------------------')
                print(
                    f'Simulating the Batch:  {batch_number} / {num_batches}  : Overall Progress: {overall_progress} %')
                print('--------------------------------------------------------------')
                if computing_method == 'parallel':
                    # Solve via Parallel Solver
                    Result = self.solve_parallel(Parameters)
                else:
                    # Solve via Serial Solver
                    Result = self.solve_serial(Parameters)

                joblib.dump(Result, Save_Batches_Path)
            else:
                print(f"Skipping batch {batch_number} (already processed).")
                print('----------------------------------------------------------------')
                continue

        # ========================= Post Processing   =================================================
        dfs = []
        # Merge all batch results
        for i in range(num_batches):
            file_path = os.path.join(
                self.cache_folder , f'Results_{i}.pkl')

            # Check if the file exists
            if os.path.exists(file_path):
                # Load the DataFrame from the batch
                result_batch = joblib.load(file_path)

                # Append the single lists  to the larger list
                for res in result_batch:
                    if type(res) == list :
                        dfs.append(res)

        final_result = pd.DataFrame(dfs, columns=[
                                    'p_w', 'p_s', 'm_w', 'm_s', 'X_A', 'X_B', 'X_C', 'D_in', 'D_out','u_w','u_s','Q_h','c_r','dp_w','dp_s', 'Result','time_elapsed'])
        final_result = final_result.sort_values(by='Result', ascending=False)
        valid_result = final_result.query(
            "Result == 'Valid solution - No subcooled liquid' | Result == 'Valid solution - All phases present' ").sort_values(by='Result', ascending=True)
        valid_all_phases = valid_result.query(
            "Result == 'Valid solution - All phases present' ")
        self.valid_m_dot_w = pd.DataFrame(
            np.round(valid_all_phases['m_w'].to_numpy(), 3))
        self.valid_m_dot_s = pd.DataFrame(
            np.round(valid_all_phases['m_s'].to_numpy(), 3))
        self.valid_p_w = pd.DataFrame(
            np.round(valid_all_phases['p_w'].to_numpy(), 3)).sort_values(by=0)
        self.valid_results = valid_result
        self.valid_results_all_phases = valid_all_phases
        self.results_all = final_result
        self.total_time = np.sum(final_result['time_elapsed'].to_numpy())
        self.mean_time = np.mean(final_result['time_elapsed'].to_numpy())
        try: num_samp = Parameter_Grids.shape[0]
        except :  num_samp = 0  
        time_table = {'total_time':self.total_time , 'mean_time':self.mean_time,'num_samples':num_samp}
        joblib.dump(time_table,os.path.join(Save_folder, 'time_elapsed.pkl'))
        # Save Results
        # Make the file Name Unique to prevent overwritting the Result files
        result_file_name = 'Results_Merged_1.xlsx'
        
        result_file_path = os.path.join(Save_folder, result_file_name)
        print(f'save_folder:{type(Save_folder)}    , result_file_name:{result_file_name}')
        while os.path.exists(result_file_path):
            step1 = result_file_name.split('.xlsx')[0]
            step2 = step1.split('Results_Merged_')
            result_file_name = f'Results_Merged_{int(step2[1])+1}.xlsx'
            result_file_path = os.path.join(Save_folder, result_file_name)
        # Save final merged Results
        final_result.to_excel(result_file_path, index=False)
        
        # Delete all caches and leftover result files
        os.remove(Prameter_Batches_Path)
        for batch_number, Parameters in enumerate(Parameter_Batches):
            Save_Batches_Path = os.path.join(
                self.cache_folder , f'Results_{batch_number}.pkl')
            if os.path.exists(Save_Batches_Path):
                os.remove(Save_Batches_Path)
        gc.collect()       
    
    def get_parameter_from_file(self,filename):
            '''get the Paramters from simulated the file name
            -------------------
            Returns:
               [m_w,m_s,xA,xB,xC,Di,Da] 
            '''
            pattern = re.compile(r"\[([\d.]+)\]")
            matches = pattern.findall(filename)

            # Assign values to variables
            P_w, ms, mr, xA, xB, Di, Da = map(float, matches)
            mw = ms/mr
            return np.array([mw,ms,xA,xB,1-(xA+xB),Di,Da])
        
    def nn_Dataset_Thermodynamic_Analysis(self,nn_Dataset_Input,nn_Dataset_Target,Input_array):
        '''Analyses the generated neural network Datasets thermodynamically ,
        ----------------
        Returns : bool
        ----------------
        boolean that states whether Dataset is Thermodynamically Valid or not
        '''
        #nn_Dataset_Input,nn_Dataset_Target=Accurate_Model.generate_nn_dataset(G, Input_array,)
        
        Input = nn_Dataset_Input[['h_w','h_s','p_w','p_s']].to_numpy()
        Target = nn_Dataset_Target[['h_w','h_s','p_w','p_s']].to_numpy()
        z_w = Input_array[2:5]
        TSQ_I=self.Calculate_Tsq_from_ph(Input, [self.AF,self.SF,z_w,[1.0]])
        TSQ_T=self.Calculate_Tsq_from_ph(Target, [self.AF,self.SF,z_w,[1.0]])

        S_w_in = TSQ_I[:,1]
        S_s_in = TSQ_I[:,4]
        S_w_out = TSQ_T[:,1]
        S_s_out = TSQ_T[:,4]
        test_first_law =np.max( np.abs(((nn_Dataset_Input['m_w']*(nn_Dataset_Target['h_w']-nn_Dataset_Input['h_w'])) - (nn_Dataset_Input['m_s']*(-nn_Dataset_Target['h_s']+nn_Dataset_Input['h_s']))).to_numpy()))
        test_second_law = np.min((nn_Dataset_Input['m_w'].to_numpy()[0]*( S_w_out - S_w_in)) + (nn_Dataset_Input['m_s'].to_numpy()[0]*(S_s_out - S_s_in)))
        if test_first_law > 1e-8 or test_second_law < 0 :
            print(f'Dataset Thermodynamic Validation : Failed ! , First Law:{test_first_law}   ,  Second Law:{test_second_law} ')
            Test_Passed = False
        else :
            #print(f'Dataset Thermodynamic Validation:  Passed! , First Law:{test_first_law}  ,  Second Law:{test_second_law} ') 
            Test_Passed = True
        return Test_Passed
    
    def Postprocessing(self,Thermodynamic_Analysis = False):
        '''
        Important Function: Generates Dataset for the Neural Network 
        following Datasets will be created :
        --------------------------------    
        Training Datasets to be merged : in save direction / nn_dataset_to_merge  folder , using Random Sampling
        Dataset for testing : in save direction / nn_dataset_to_test folder , without Random Sampling
        --------------------------------
        Dataset Structure : access datasets with these keys:[inputs,targets]
        
        Returns
        -------
        None.

        '''

        # Get all files in the directory
        npy_files = glob.glob(os.path.join(
            fr'{self.Save_Direction}\Raw_Data', '*.npy'))

        # Iterate over each .npy file
        for npy_file in npy_files:
            # Load the data from the .npy file
            data = np.load(npy_file)

            #!!! TO dO , Implement the NN Data Generation.
            Input_array=self.get_parameter_from_file(npy_file)
            
            nn_Dataset_Input,nn_Dataset_Target = self.generate_nn_dataset(data, Input_array,Random_Sample_size=self.Random_Sample_Size)
            if Thermodynamic_Analysis == False:
                sanity_check = True 
            else:
                sanity_check=self.nn_Dataset_Thermodynamic_Analysis(nn_Dataset_Input,nn_Dataset_Target,Input_array)
            
            if sanity_check == True :
                post_filename=f'nn_P_w[{data[0,1]:.1f}]_ms[{Input_array[1]}]_mw[{Input_array[0]}]_xA[{Input_array[2]}]_xB[{Input_array[3]}]_Di[{Input_array[5]}]_Da[{Input_array[6]}]'
                base_folder = os.path.join(self.Save_Direction,'nn_dataset_to_merge')
                if not os.path.exists(base_folder):
                    os.makedirs(base_folder)
                    
                test_folder =   os.path.join(self.Save_Direction,'nn_dataset_to_test')
                if not os.path.exists(test_folder):
                    os.makedirs(test_folder)
                    
                if self.Random_Sample_Size == None:    
                    joblib.dump({'inputs':nn_Dataset_Input,'targets':nn_Dataset_Target}, os.path.join(base_folder,f'{post_filename}_rs[{self.Random_Sample_Size}].pkl') )
                    joblib.dump({'inputs':nn_Dataset_Input,'targets':nn_Dataset_Target}, os.path.join(test_folder,f'{post_filename}_rs[{self.Random_Sample_Size}].pkl') )           
                    print(f'Dataset Created : {post_filename}')
                else:
                    nn_Dataset_Input_test,nn_Dataset_Target_test = self.generate_nn_dataset(data, Input_array,Random_Sample_size=None)
                    joblib.dump({'inputs':nn_Dataset_Input,'targets':nn_Dataset_Target}, os.path.join(base_folder, f'{post_filename}_rs[{self.Random_Sample_Size}].pkl') )
                    joblib.dump({'inputs':nn_Dataset_Input_test,'targets':nn_Dataset_Target_test}, os.path.join(test_folder,f'{post_filename}_rs[None].pkl') )           
                    print(f'Dataset Created : {post_filename}')


    def merge_datasets(self,random_sampling=True,Homogenous=True,save=True):
        '''Important Function
        Merge the Datasets from nn_dataset_to_test folder.
        it has also option to downsize the datasets via Random Sampling method.
        
        --------------
        random_sampling : bool , take Random Sample using the predefined Random Sample size 
        Homogenous : bool , take Homogenous Samples based on the mean size of datasets , if the Dataset has more samples than mean size of all datasets 
        a random sample with same size is being taken 
        save: bool saves the Merged Dataset as Dictionary , to access the contents use following keys: [inputs,targets]
        --------------
        return :
            merged_Inputs Dataframe , merged_outputs Dataframe
        '''
        list_inputs = []
        list_targets = []
        size_list = []
        folder_path=os.path.join(self.Save_Direction,'nn_dataset_to_test')
        # Get a list of .pkl files in the specified folder
        pkl_files = glob.glob(os.path.join(folder_path, '*.pkl'))
    
        # Iterate over each .pkl file
        for file_i in pkl_files:
            # Load the data from the .pkl file
            data = joblib.load(file_i)
    
            Inputs = data['inputs']
            Targets = data['targets']
    
            # Store the Inputs and Targets in lists
            list_inputs.append(Inputs)
            list_targets.append(Targets)
            size_list.append(Inputs.shape[0])
    
        # Step Homogenous Random Sampling
        if random_sampling == True:
            if Homogenous == True:
                size_array = int(np.mean(np.vstack(size_list)))  # Calculate the mean size of the arrays
            else:size_array = self.Random_Sample_Size  # Calculate the mean size of the arrays
        else:  size_array = 0  
        # Get columns
        input_col = Inputs.columns
        target_col = Targets.columns
    
        list_inputs_rs = []
        list_targets_rs = []
    
        # Random Sampling
        for i in range(len(list_inputs)):
            # Take a random sample
            inputs, targets = self.take_random_sample(
                list_inputs[i].to_numpy(), list_targets[i].to_numpy(), size_array)
            list_inputs_rs.append(inputs)
            list_targets_rs.append(targets)
    
        list_inputs_rs = np.vstack(list_inputs_rs)
        list_targets_rs = np.vstack(list_targets_rs)
    
        # Merge the datasets
        df_inputs = pd.DataFrame(list_inputs_rs, columns=input_col)
        df_targets = pd.DataFrame(list_targets_rs, columns=target_col)
        if save == True:
            joblib.dump({'inputs':df_inputs,'targets':df_targets},os.path.join(self.Save_Direction,'nn_Merged_Dataset.pkl'))
        return df_inputs, df_targets
    
    def  plot_correlation_matrix(self,DF_inputs):
          correlation_matrix = DF_inputs.corr()
          fig=plt.figure(figsize=(10, 8))
          sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
          plt.title('Correlation Heatmap')
          plt.show()
          return fig
      
    def plot_distribution_histogram(self,DF_inputs):
        # #['dx', 'h_w', 'p_w', 'm_w', 'h_s', 'p_s', 'm_s', 'X_A', 'X_B', 'X_C','D_in', 'D_out']
        figure_list=[]
        #Create a folder for Plots
        save_folder = os.path.join(self.Save_Direction,'Plots')
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        #Plot histograms for each     
        for col in DF_inputs.columns:
            fig=plt.figure(figsize=(8, 6))
            
            fig.savefig(rf'D:\Masterarbeit\Results\distribution_diagram_{col}.png')
            sns.histplot(DF_inputs[col], bins=25, kde=True)
            plt.title(f'Histogram for {col}',fontsize=24)
            plt.xlabel(f'{col}',fontsize=16)
            plt.ylabel('Count',fontsize=16)
            plt.tick_params(fontsize=13)
            fig.savefig(f'{save_folder}\distribution_diagram_{col}.png')
            #plt.show()
            
if __name__ == '__main__':

    #!!!Run the code
    # Get the current working directory
    current_path = os.getcwd()

    # Get the parent directory
    parent_path = os.path.dirname(current_path)  
    
    Input = {'WF': 'Propane*Isobutane*Pentane',
                  'SF': 'water',
                  'random_sample': 0.1,
                  'p_w': [2, 23],
                  'p_s': 1.5,
                  'D_w': 25.0 - (2*1.6e-3) ,
                  'D_s': 30.0,
                  'm_r': [2, 10],
                  'm_w': [1e-3, 5e-1],
                  'm_s': [2.5e-3, 0.1],
                  'save_directory': os.path.join(parent_path,rf"Physical_Results"),
                  'Save_Outputs': True,
                  'verbose': 1,
                  'L': 100,
                  't': 1.6e-3,
                  'ode_atol': 1e-2,
                  'ode_rtol': 1e-1,
                  'method': 'RK45',
                  'n_samples':1,
                  'skip_list':[],
                  'cache_folder': os.path.join(parent_path,rf"Physical_Cache")}
    
    Model = Heat_Exchanger_Model(Input)
    Model.simulate(computing_method='parallel')
    Model.Finish_sound()
    Model.Postprocessing()
    Model.merge_datasets()
    gc.collect()