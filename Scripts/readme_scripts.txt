1. Create_new.py Script enthält die Physikalisches Modell und die Algorithmus zur Erzeugung der Datensätze für MLP Modell.

2. Neuronal_Network_new.py enthält die MLP Modell , zur Training MLP Modell sowie Hyperparameter Optimierung.

3. Validation_new.py enthält die Validierung code zur Darstellung der Kurvendiagramme sowie Streuerdiagramme , Datengrößeanalyse und Thermodynamische Validierung der Netzwerke.
 
4.Break_Event_new enthält Code für Zeitliche Analyse der MLP Model vs Physikalisches Modell.

Anleitung für Create_new.py:

 Set the Refprop Directory os.environ['RPPREFIX'] = r'C:/Program Files (x86)/REFPROP' to your Refprop Directory.
1.Add a Save directory to direction_list and cache_folder in Init_Dict Dictionary
2.Change the Simulation Settings in Init_Dict Dictionary
3.Define a Physical Model (Model = Heat_Exchanger_Model(Init_Dict)  ), set the cache address and saving address.
4.Solve the Physical Model with given Parameter Grid using either serial solver or parallel solver with Model.simulate()
5.use Model.Postprocessing method to generate datasets for Neural Network and check if the Datasets are thermodynamically valid (first law and second law)
6.use Model.merge_datasets method to merge the generated datasets and take random samples (recommanded Homogenous random sampling)
7.run the code(everything is setup)


Neuronal_Network_new.py :
    1.Define a Dictionary to initialize the Neural Network with , use Init_Dict for basic settings
    2.Define the optimization space for Hyperparameter Optimization , use the optimization_spce for basic settings
    3.Define a Neural Network Model with Neural_Network_Model(Init_Dict)
    4.Train the Nerual Network with  .train function  (model.train() ) , the model will be automatically trained and saved at save directory
    5.to perform Hyperparameter Tuning use the .search function (model.search() ) the search results and best model will be saved at save directory
    6.run the code(everything is setup)


 