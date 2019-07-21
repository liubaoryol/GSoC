import os
import numpy as np
import classes
# set working directory
dir1 = "/home/lytica/Documents/GSoC/GSoC/data/data1"
dir2 = "/home/lytica/Documents/GSoC/GSoC/data/data2"
dir3 = "/home/lytica/Documents/GSoC/GSoC/data/data3"
dir4 = "/home/lytica/Documents/GSoC/GSoC/data/data4"

person1=classes.Person()
person1.read_activity_from_folder(dir1)



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

#Encapsular tambien cada posicion en un array


                                                               



