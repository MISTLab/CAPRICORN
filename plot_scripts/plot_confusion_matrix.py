import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colorbar as colorbar

# Load detections
detected_centralized_closures = np.load('../results/cen_scores.npy')
detected_decentralized_closures = np.load('../results/dec_scores.npy')
gt_scores = np.load('../results/groundtruth_scores.npy')

detected_centralized_best_idx = np.argmax(
    detected_centralized_closures,axis=1)


detected_decentralized_best_idx = np.argmax(
    detected_decentralized_closures, axis=1)

x_cen = np.arange(len(detected_centralized_best_idx))
x_decen = np.arange(len(detected_decentralized_best_idx))

colors_cen = cm.jet(gt_scores[x_cen, detected_centralized_best_idx])
colors_decen = cm.jet(gt_scores[x_decen, detected_decentralized_best_idx])


# Filter out scores which are too low
score_lim = 0.25
max_val_cen = detected_centralized_closures[np.arange(
    len(detected_centralized_best_idx)), detected_centralized_best_idx]
max_val_decen = detected_decentralized_closures[np.arange(
    len(detected_decentralized_best_idx)), detected_decentralized_best_idx]

rejected_arg_cen = np.argwhere(max_val_cen < score_lim)
rejected_arg_decen = np.argwhere(max_val_decen < score_lim)

# Put them at values which won't show in the plot
x_cen[rejected_arg_cen] = -10
x_decen[rejected_arg_decen] = -10
detected_centralized_best_idx[rejected_arg_cen] = -10
detected_decentralized_best_idx[rejected_arg_decen] = -10


plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')


fig = plt.figure(figsize=(6, 3))
ax0 = fig.add_subplot(1, 2, 1)
ax1 = fig.add_subplot(1, 2, 2)

ax0.set_title('Centralized')
ax0.scatter(x_cen, detected_centralized_best_idx, s=2.0, c=colors_cen)
cax = fig.add_axes([0.92, 0.1,0.008, 0.8])
cax.set_label('test')
cb = colorbar.ColorbarBase(cax, cmap=cm.jet,label='Ground-truth score', orientation='vertical')
cb.ax.invert_yaxis()

ax1.set_title('Decentralized')
ax1.scatter(x_decen, detected_decentralized_best_idx, s=1.5, c=colors_decen)
ax0.set_xlabel('Index of the queried frame')
ax0.set_ylabel('Index of the matched frame')
ax1.set_xlabel('Index of the queried frame')
plt.gca().invert_yaxis()

plt.show()
