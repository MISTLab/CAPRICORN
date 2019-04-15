import settings
from parse_tools import get_constel_files_in_folder
import numpy as np

######################### Give all necessary files path
# Path to the data directory, which contains the sequence folder and the constellations folder
base_dir = '../data/' 

# Path to the groundtruth.txt file containing all the camera poses in the sequence
groundtruth_file = base_dir+'rgbd_dataset_freiburg3_long_office_household/groundtruth.txt' 

# Path to the constellations folder
constellation_folder = base_dir+'/constellations'

# Path where the results are stored
save_folder = '../results/'

# Load the settings and the constellations from the files.
settings.init(base_dir+'coco.names')
constel_files = get_constel_files_in_folder(constellation_folder)

######################### Save the ground truth closures
print('Saving ground truth loop closure scores, based on distances between estimated scenes.')

from get_groundtruth_scores import get_groundtruth_scores

gt_scores, gt_distances = get_groundtruth_scores(constel_files,groundtruth_file)

np.save(save_folder+'groundtruth_scores.npy',gt_scores)
np.save(save_folder+'groundtruth_distances.npy', gt_distances)

print('Finished saving ground truth loop closure scores.\n\n')


######################### Save the decentralized closures
print('Simulating the decentralized behavior.')

from simulate_decentralized_loop_closures import get_loop_closure_scores_dec

for nb_robots in settings.nb_robots_checked:

    scores_dec, robots_to_verify, semantic_descriptors = get_loop_closure_scores_dec(
        constellation_folder,nb_robots)

    np.save(save_folder+'dec_scores_'+str(nb_robots)+'.npy', scores_dec)
    np.save(save_folder+'dec_rob2ver_'+str(nb_robots)+'.npy', robots_to_verify)
    np.save(save_folder+'dec_sem_desc_'+str(nb_robots)+'.npy', semantic_descriptors)

print('Finished saving information from decentralized loop closure.\n\n')

# ########################### Save the centralized closures
print('Simulating the centralized behavior.')

from simulate_centralized_loop_closures import get_loop_closure_scores_cen

scores_cen = get_loop_closure_scores_cen(constel_files)

np.save(save_folder+'cen_scores.npy', scores_cen)

print('Finished saving information from centralized loop closure.')
