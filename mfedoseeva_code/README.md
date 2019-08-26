# Requirements
python3  
numpy  
scikit-learn  
matplotlib

# Use

1. Clone the repository

2. Download [cad-60 dataset](http://pr.cs.cornell.edu/humanactivities/data.php), unzip all 4 parts to one directory

3. The dataset has to be broken down according to 5 environments. data_separation_script.py from support_operations will do that.  
for that create a folder where you would want the script to put the sorted dataset: 

```commandline
cd support_operations

python data_separation_script.py --dataset_dir source_folder --separated_dataset_dir target_folder
```
for example:
```commandline
python data_separation_script.py --dataset_dir ../../cad60dataset --separated_dataset_dir ../../separated_cad60
```

4. The next step is to read the skeletons and save them to python data structures.
```commandline
cd feeder

python cad_gendata.py --data_path your-path-to-cad-60-dataset(separated!)
```

for example
```commandline
python cad_gendata.py --data_path ../../cad60_separated
```

5. Now you can run the main. the default way to run main.py is:

```commandline
python main.py
```
by default it assumes that data (.npy and .pkl files) is saved into data0 folder at the root of the project folder where the main.py exists.
The default run only delivers the accuracy for the cross-validation without confusion matrices or saving the model.

the full command to run main.py with all parameters is:

```commandline
python main.py --dataset-dir data_folder --dataset_name dataset_name --evalution evaluation_type
```

the default arguments are:

```commandline
python main.py --dataset-dir ./data0 --dataset_name CAD-60 --evalution cv
```

if you want to produce confusion matrices and save the model, run main.py as follows:

```commandline
python main.py --evaluation full
```

# Credits

[hcn](https://github.com/huguyuehuhu/HCN-pytorch)  
[scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)



