import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes

# Put the parameters used in simulation
nb_robots_choices = range(1,21)
nb_candidates = 4
nb_classes = 80
nb_robots_further_check = 4
semantic_descriptors = []
robots_verified = []

bow_data = np.zeros(len(nb_robots_choices))
netvlad_data = np.zeros(len(nb_robots_choices))
ours_data = np.zeros(len(nb_robots_choices))
ours_data_broadast = np.zeros(len(nb_robots_choices))
nb_robots_checked = 0

# semantic_descriptors_nb_robots_10_nb_cand_1_rob_furth_check_5_min_dist_0.1.npy
for nb_robots in nb_robots_choices:
    nb_robots_checked +=1
    semantic_desc_file_path = '../results/dec_sem_desc_'+str(nb_robots)+'.npy'
    robots_ver_file_path = '../results/dec_rob2ver_'+str(nb_robots)+'.npy'
    semantic_descriptors.append(np.load(semantic_desc_file_path))
    robots_verified.append(np.load(robots_ver_file_path))

    _,nb_frames,_ = semantic_descriptors[-1].shape
    semantic_splits_idx = np.linspace(0, nb_classes, nb_robots+1,dtype=int)
    nb_labels_received = np.zeros((nb_robots,nb_frames))
    nb_cand_received = np.zeros((nb_robots,nb_frames))
    nb_further_checks = np.zeros((nb_robots, nb_frames))
    for rob_i in range(len(semantic_splits_idx)-1):
        nb_labels_received[rob_i,:] = np.count_nonzero(semantic_descriptors[-1][:,:,semantic_splits_idx[rob_i]:semantic_splits_idx[rob_i+1]],axis=(0,2))
        cand = (robots_verified[-1][rob_i, :, :, :, 0] > -1).reshape(nb_frames, -1)
        nb_cand_received[rob_i, :] = np.count_nonzero(cand, axis=1)
        further_checks_labels =robots_verified[-1][rob_i, :, :, :, 0].reshape(nb_frames, -1)


    nb_objects = np.maximum(np.sum(semantic_descriptors[-1],axis=2),16)
    # 1.5 byte for each label nb and id, 1 byte for the robot id, 2 bytes for the frame id. For each object : 1 byte for the label, 2 bytes per position
    total_data_received_ours = 1.5*nb_labels_received + nb_cand_received * \
        (1+2) + nb_objects*nb_robots_further_check*(1+3*2)

    # Each query is 16kB per robot, to each robot. Return 1 robot id and 2 for frame id
    total_data_received_bow = np.zeros(((nb_robots, nb_frames)))+16000+(1+2)*nb_robots

    # 512 bytes for each query, 1 for robot id and 2 for frame id
    total_data_received_netvlad = np.zeros(
        ((nb_robots, nb_frames)))+512+(1+2)*nb_robots

    # Ours case naive : Send the full constellation to every other robot
    total_data_received_ours_naive = nb_objects*nb_robots*(1+3*2)

    # Divide per nb of frames and nb of robots to get the data for a single query
    bow_data[nb_robots_checked -
             1] = np.sum(total_data_received_bow)/nb_frames/nb_robots
    netvlad_data[nb_robots_checked -
                 1] = np.sum(total_data_received_netvlad)/nb_frames/nb_robots
    ours_data[nb_robots_checked -
              1] = np.sum(total_data_received_ours)/nb_frames/nb_robots
    ours_data_broadast[nb_robots_checked -
                    1] = np.sum(total_data_received_ours_naive)/nb_frames/nb_robots

# Convert to kB
bow_data/=1e3
netvlad_data/=1e3
ours_data/=1e3
ours_data_broadast/=1e3

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='x-small')
plt.rc('ytick', labelsize='x-small')

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(1, 1, 1)

# x = np.linspace(1., 8., 30)
ax.plot(np.arange(1, nb_robots_checked+1), ours_data, color='r', ls='solid')
ax.plot(np.arange(1, nb_robots_checked+1),
        ours_data_broadast, color='k', ls='dashdot')
ax.plot(np.arange(1, nb_robots_checked+1),
        netvlad_data, color='g', ls='dashed')
ax.plot(np.arange(1, nb_robots_checked+1), bow_data, color='b', ls='dotted')
ax.set_xlabel('Number of robots')
ax.set_ylabel('Size of one query [kB]')
plt.gca().legend(('Our solution', 'Broadcast constellation', 'Solution from [42]', 'Solution from [10]'), loc=6, prop={'size': 7})

major_ticks_x = np.arange(0, 20.01, 5)
minor_ticks_x = np.arange(0, 20.01, 1)


ax.set_xticks(major_ticks_x)
ax.set_xticks(minor_ticks_x, minor=True)
ax.grid(which='both')

ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

axins_args = {'yticks': np.arange(0, 20.01, 0.4)}
axins = zoomed_inset_axes(ax, 2.5, loc=7, axes_kwargs=axins_args)
axins.plot(np.arange(1, nb_robots_checked+1), ours_data, color='r', ls='solid')
axins.plot(np.arange(1, nb_robots_checked+1),
        ours_data_broadast, color='k', ls='dashdot')
axins.plot(np.arange(1, nb_robots_checked+1),
        netvlad_data, color='g', ls='dashed')
# specify the limits
axins.plot(np.arange(1, nb_robots_checked+1), bow_data, color='b', ls='dotted')
x1, x2, y1, y2 = 16.5, 19.5, 0.0,3.
axins.set_xlim(x1, x2)  # apply the x-limits
axins.set_ylim(y1, y2)  # apply the y-limits

from mpl_toolkits.axes_grid1.inset_locator import mark_inset
mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")

plt.show()
