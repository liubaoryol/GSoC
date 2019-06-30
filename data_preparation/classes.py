import os
import numpy as np

class Person:
    def __init__(self,act=[]):
        self.activity = act

    def string2list(self,stringact):
        tmp=stringact.split("\n")[:-1]
        dim = [len(tmp),tmp[0].count(",")]
        lista = np.zeros(dim)
        for i in range(len(tmp)):
            lista[i]=np.fromstring(tmp[i],sep=",")
        return lista


    def read_activity_from_folder(self,folder):
        #np.loadtxt(folder,delimiter=",",unpack=True)
        entries = os.listdir(folder)
        for entry in entries:
            if entry.endswith(".txt") and "activity" not in entry:
                with open(os.path.join(folder,entry),'r') as df:
                    tmp=self.string2list(df.read()[:-3]) #the -3 is to erase END word in file
                    self.activity.append(tmp) 


    #Adding activity if person performs a new activity
    def add_activity(self,act):
        self.activity.append(act)

    
    
