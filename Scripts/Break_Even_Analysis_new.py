# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 11:38:00 2024

@author: Ali2
"""
import numpy as np
import matplotlib.pyplot as plt
A=np.array([
17.00,
33.00,
62.00,
122.00,
237.00,
487.00,
982.00,
1986.00])


B = np.array([
33.14,
57.16,
140.48,
253.03,
612.70,
1510.01,
3257.14,
14601.11])


C=np.polyfit(A, B, 3)

NN = 385.26 + 1107.240265 + 14601.10982
xmax = 2200
A1 = A.tolist()
B1 = B.tolist()
#%%
D=np.linspace(0,xmax,8)
f=float(np.polyval(C,xmax))
E = np.abs(np.polyval(C,D))   
# A1.append(xmax) 
# B1.append(f)
plt.figure(figsize=[15,18])
plt.grid(linewidth=1.5)
plt.scatter(A1,np.array(B1)/60,color='black',linewidth=2)
plt.hlines(NN/60, 0, xmax,linewidth=3,alpha=0.90)
plt.scatter(2047,NN/60,marker='*',s=120,color='black')
plt.xlabel('Anzahl der Simulationen',fontsize=26)
plt.ylabel('Zeit [min]',fontsize=26)
plt.tick_params(axis='both',labelsize=24)
plt.vlines(2047,0,NN/60,color='black',linestyles='dotted',linewidth=2)
plt.text(2080,0,'x=2047',fontsize=20)
plt.ylim([-10,450])
plt.plot(D,np.array(E)/60,':r',linewidth=2)
plt.legend(['Rechenzeit des physikalischen Modells' ,'Gesamte verstrichene Zeit für das Training eines optimalen MLP-Modells','Break-Even-Punkt'],fontsize=20)
plt.title('Break-Even-Analyse',fontsize=26)