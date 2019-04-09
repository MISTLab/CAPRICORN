import numpy as np
import matplotlib.pyplot as plt

nb_robots = 10
nb_classes = 80
semantic_descriptors = []
robots_verified = []


semantic_desc_file_path = '../results/dec_sem_desc.npy'

semantic_descriptors = np.load(semantic_desc_file_path)
semantic_descriptors_global = np.sum(semantic_descriptors,axis=(0,1))
semantic_descriptors_global /= np.sum(semantic_descriptors_global)
semantic_descriptors_global*=100

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(1, 1, 1)

# Need as many colors as robots
colors = [
	"#009e5d",
	"#f80097",
	"#023518",
	"#00006f",
	"#beb369",
	"#ba004c",
	"#00cdff",
	"#d38f62",
	"#483230",
	"#eab8c8"
]
colors = [val for val in colors for _ in (0,1,2,3,4,5,6,7,8,9)]
ax.bar(np.arange(0, 80), semantic_descriptors_global,width = 1.8,color=colors)

ax.set_xlabel('Coco class index')
ax.set_ylabel('% of appearance in the sequence')

major_ticks_x = np.arange(0, 81, 20)
minor_ticks_x = np.arange(0, 81, 5)
major_ticks_y = np.arange(0, 50, 25)
minor_ticks_y = np.arange(0, 50, 10)

ax.set_xticks(major_ticks_x)
ax.set_xticks(minor_ticks_x, minor=True)
ax.set_yticks(major_ticks_y)
ax.set_yticks(minor_ticks_y, minor=True)

ax.grid(which='both')

ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)

plt.show()
