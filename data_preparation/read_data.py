import os
import numpy as np
import classes
# set working directory
dir1 = "/home/lytica/Documents/GSoC/data/data1"
dir2 = "/home/lytica/Documents/GSoC/data/data2"
dir3 = "/home/lytica/Documents/GSoC/data/data3"
dir4 = "/home/lytica/Documents/GSoC/data/data4"

person1=classes.Person()
person1.read_activity_from_folder(dirs)



'''
There are several ways for viewing an activity.
An activity variable is an array of 12 activities for each person.
Each activity is composed of n number of frames. Each frame has 170 entries: 
Frame#,ORI(1),P(1),ORI(2),P(2),...,P(11),J(11),P(12),...,P(15)
where ORI(i)-- orientation of joint has 10 entries
P(i) position of joint has 4 entries.
We have 15 joints, 11 of them given together with orientation

Therefore it occurs to me three ways to represent the data:
1. Flat list (no confidence value)
2. List of lists composed ONLY with positions of joints (no confidence value)
3. List of lists composed with both position and orientation of joints in same/different sublists (no confidence value)
4. Same as above but together with confidence value


'''

' As fist approach we will use only positions of joints, with each joint grouped in a 3D array.
def pos(activities):
	pos_act=np.array(activities)
	cols = [] #Here we will save the positional cols
	#Finding positions for the first 11 joints
	for i in range(0,len(pos_act[0][0])-16,14):
		cols.append([i+11,i+12,i+13]) 
	cols=cols[:-1]
	#Finding positions for the last 4 joints
	for i in range(len(pos_act[0][0])-16,len(pos_act[0][0]),4):
		cols.append([i,i+1,i+2])
	for j in range(len(pos_act)):
		pos_act[j] = pos_act[j][:,cols] #extracting the needed columns
	return pos_act

#Encapsular tambien cada posicion en un array
