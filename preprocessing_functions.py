import matplotlib.pyplot as plt
import sys
sys.path.append("feature_extraction")
sys.path.append("data_preparation")
import classes
import functions


def import_data(env):
	if env == "all":
		dir1 = "/home/lytica/Documents/GSoC/GSoC/data/data1"
		dir2 = "/home/lytica/Documents/GSoC/GSoC/data/data2"
		dir3 = "/home/lytica/Documents/GSoC/GSoC/data/data3"
		dir4 = "/home/lytica/Documents/GSoC/GSoC/data/data4"
	else:
		dir1 = "/home/lytica/Documents/GSoC/GSoC/separated_data/" + env + "/data1"
		dir2 = "/home/lytica/Documents/GSoC/GSoC/separated_data/" + env + "/data2"
		dir3 = "/home/lytica/Documents/GSoC/GSoC/separated_data/" + env + "/data3"
		dir4 = "/home/lytica/Documents/GSoC/GSoC/separated_data/" + env + "/data4"
	return dir1,dir2,dir3,dir4

def create_person_instances(environment):
	dir1,dir2,dir3,dir4 = import_data(environment)

	person1=classes.Person()
	person2=classes.Person()
	person3=classes.Person()
	person4=classes.Person()

	person1.read_activity_from_folder(dir1)
	person2.read_activity_from_folder(dir2)
	person3.read_activity_from_folder(dir3)
	person4.read_activity_from_folder(dir4)

	return person1,person2,person3,person4

def standardize_person(person):
	#center torso and make person's hight the same for all persons
	torso_centered=functions.center_torso(functions.pos(person.activity))
	person.pos_activities = functions.normalize_by_height_all(torso_centered)

def clean(person1_features,person2_features,person3_features,person4_features):
	#Removing the none activity	
	person1_features.pop(0)
	person2_features.pop(-2)
	person3_features.pop(-2)
	person4_features.pop(-2)

	person1.label.pop(0)
	person2.label.pop(-2)
	person3.label.pop(-2)
	person4.label.pop(-2)

	#removing the random activity
	person1_features.pop(10)
	person2_features.pop(10)
	person3_features.pop(2)
	person4_features.pop(-5)

	person1.label.pop(10)
	person2.label.pop(10)
	person3.label.pop(2)
	person4.label.pop(-5)

def save_history_training(history,model,filename = ""):
	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('model accuracy')
	plt.ylabel('accuracy')
	plt.xlabel('epoch')
	plt.legend(['train', 'test'], loc='upper left')
	if not filename:
		filename = str(n_layers)+"-layer_"+str(n_units)+"unit"+environment+model + "Training.png"
	plt.savefig(filename)
	plt.clf()

