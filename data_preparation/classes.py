import os
import numpy as np

class Person:
'''
Class person has two variables: the one that is composed of activities and the second are the label to each entry of the activity variable

We have five environments: bathroom, bedroom, kitchen, living room and office which are composed of the following activities:

bathroom
    - rinsing mouth with water
    - brushing teeth
    - wearing contact lenses
bedroom
    - talking on the phone
    - drinking water
    - opening pill container
kitchen
    - cooking (chopping)
    - cooking (stirring)
    - drinking water
    - opening pill container
livingroom
    - talking on the phone
    - drinking water
    - talking on couch
    - relaxing on couch
office 
    - talking on the phone
    - writing on whiteboard
    - drinking water
    - working on computer'
'''
    def __init__(self,act=[],label=[]):
        self.activity = act
        self.label = label

#This function is for converting the activity file into a numpy array of n rows, corresponding to frames of the activity
    def string2list(self,stringact):
        tmp=stringact.split("\n")[:-2] #the -1 is to remove the line with the END word
        dim = [len(tmp),tmp[0].count(",")]
        lista = np.zeros(dim)
        for i in range(len(tmp)):
            lista[i]=np.fromstring(tmp[i],sep=",")
        return lista

#The search function is used to find the position of an argument in a listo of lists. Consequently it is used to find the name of activity for a set of frames read from file
    def search(self,lst, item):
        for i in range(len(lst)):
            part = lst[i]
            for j in range(len(part)):
                if part[j] == item: return (i, j)
        return None

#updating activity and labels
    def read_activity_from_folder(self,folder):
        tmp = open(os.path.join(folder,"activityLabel.txt"),'r').read()
        tmp = tmp.split("\n")[:-2]
        activityList=[tmp[i].split(",") for i in range(len(tmp))]
        #np.loadtxt(folder,delimiter=",",unpack=True)
        entries = os.listdir(folder)
        for entry in entries:
            if entry.endswith(".txt") and "activity" not in entry:
                with open(os.path.join(folder,entry),'r') as df:
                    tmp=self.string2list(df.read()) 
                    self.activity.append(tmp) 
                    name = os.path.splitext(entry)[0]
                    index=self.search(activityList,name)
                    if index!=None:
                        self.label.append(activityList[index[0]][1])
                    else:
                        self.label.append("none")



    #Adding activity if person performs a new activity
    def add_activity(self,act,lab):
        self.activity.append(act)
	self.label.append(lab)

    
    
