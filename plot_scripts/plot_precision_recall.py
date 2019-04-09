
from evaluate_pr import my_pr
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc


gt_scores = np.load('../results/groundtruth_scores.npy')
gt_dist = np.load('../results/groundtruth_distances.npy')
detected_centralized_closures = np.load('../results/cen_scores.npy')
detected_decentralized_closures = np.load('../results/dec_scores.npy')

# Score over which ground truths are considered valid
groundtruth_closures = gt_dist < 2.0

# We can also consider we are not making an error when there simply aren't any objects. In that case the groundtruth should not have a loop closure when there aren't enough objects. When detected closures is -2 it means there aren't 3 objects in the scene, and when it is between -1 and 0 it means we don't have 3 same objects in a scene.
groundtruth_closures_fix_objects = np.logical_and(
    groundtruth_closures, (detected_centralized_closures >= 0.0))
groundtruth_closures_fix_objects_dec = np.logical_and(
    groundtruth_closures, (detected_decentralized_closures >= 0.0))



precision1, recall1 = my_pr(
    groundtruth_closures_fix_objects, detected_centralized_closures)
precision2, recall2 = my_pr(
    groundtruth_closures_fix_objects_dec, detected_decentralized_closures)



auc1 = auc(recall1,precision1)
auc2 = auc(recall2, precision2)

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(1, 1, 1)

ax.plot(recall1, precision1, color='r', ls='solid')
ax.plot(recall2, precision2, color='b', ls='dashed')

ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
plt.gca().legend(('Ours, centralized, AUC = ' +"{:.4f}".format(auc1), 'Ours, decentralized, AUC = '+"{:.4f}".format(auc2)),prop={'size':9})
major_ticks_x = np.arange(0, 1.01, 0.5)
minor_ticks_x = np.arange(0, 1.01, 0.1)
major_ticks_y = np.arange(0, 1.1, 0.5)
minor_ticks_y = np.arange(0, 1.1, 0.1)
ax.set_xticks(major_ticks_x)
ax.set_xticks(minor_ticks_x, minor=True)
ax.set_yticks(major_ticks_y)
ax.set_yticks(minor_ticks_y, minor=True)
ax.set_xlim([0,1])
ax.set_ylim([0, 1.05])

# And a corresponding grid
ax.grid(which='both')

# Or if you want different settings for the grids:
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

plt.show()
