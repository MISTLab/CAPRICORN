# CAPRICORN

This repository contains the code to reproduce the results of our paper:
Benjamin Ramtoula, Ricardo de Azambuja, Giovanni Beltrame. **Data-Efficient Decentralized Place Recognition with 3D constellations of Objects.** Submitted to the International Symposium on Multi-Robot and Multi-Agent Systems (MRS), 2019.


## Overview
The process to reproduce our results is the following:
1. Download the sequence we use. We already provide the computed outputs of the object-to-point detection system to create the constellations.
2. Simulate and compute the loop closure scores you want (ground truth, centralized, decentralized).
3. Verify and plot the results.

We provide in the `data/constellations.zip` file which contains the pre-computed outputs of the object-to-point detection system on the `rgbd_dataset_freiburg3_long_office_household` sequence of the TUM-RGBD dataset. Each file describes a constellation associated to each RGB frame of the sequence.

Each point of the constellation corresponds to one object, and is described by one line in the description file. It contains the following information:
```
<object-class> <x> <y> <z>
```

Where the object class comes from the [COCO dataset](http://cocodataset.org/) and are listed in the [data/coco.names file](data/coco.names).

Each position is expressed in the camera frame.

## Getting ready

### Python
All these scripts were tested with Python 3.7.

### Getting the constellation files
You should decompress the `data/constellations.zip` file which is provided. We suggest keeping the .txt files in a directory named `constellations`, within the `data` directory.

### Download the sequence
In order to compute the ground truth score, and to understand the constellation and verify their validity by checking the RGB frames, you need to download the `fr3/long_office_household` sequence of the TUM-RGBD dataset here: <https://vision.in.tum.de/data/datasets/rgbd-dataset/download>

Once you have downloaded the compressed file, you can extract it and place the folder in the `data` directory. You can now check and compare the provided constellation as well as the results to the RGB frames of the sequence.


## Simulate and compute loop closures
We provide a script which does 3 things:
- It computes the ground truth score.
- It computes the centralized loop closure scores, where each pairs of constellation are compared.
- It computes the decentralized loop closure scores, the robots returned from partial semantic queries for each frame (n_ret in the paper), and the full semantic descriptors of each frame. These values are used for estimating the amount of data exchanged later on. These can be done for different number of robots in order to see how the amount of data exchanged scales. 


### Choose the settings

Before running the simulations, you can control the different settings used in the simulation in the [`src/settings.py` file](src/settings.py).

### Run the script

You should now open the script used to run everything: `src/save_loop_closures.py`.
At the top of the script there are several paths that are defined. Make sure they fit your data organization.

The default folder to save outputs is `../results`. If you want to use this, you should create it yourself before running the script.

Once you have chosen the right settings and paths you can run the simulations. This will take more than a few minutes since many computations are performed, and cases simulated.

To run the script, simply call from the `src` directory:
```
python save_loop_closures.py
```

If you want to re-run certain parts of the script with different parameters, you can simply comment out the ones you do not want to run.

## Check results

In order to check your results, you can use the scripts we provide in the `plot_scripts` directory. 
The scripts make use of the numpy arrays saved previously.
By default, they use the paths and names we have predefined. You can modify them if needed when `np.load` is used.

Here are the possible scripts you can run from the `plot_scripts` directory:
```
python plot_precision_recall.py
python plot_imbalanced_data.py
python plot_confusion_matrix.py
python plot_data_comm.py
```
