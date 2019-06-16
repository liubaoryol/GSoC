import os
import numpy as np
import classes.py
# set working directory
dir = "/home/liuba/Documents/GSoC/data/data1"

person1=Person()
person1.read_activity_from_folder()

person1.activity[0]