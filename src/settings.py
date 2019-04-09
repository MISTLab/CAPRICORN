"""
File that loads experiment specific parameters.
"""


def init(classes_file):
    global classes, min_objects, loop_closure_neighbors_ignored, min_dist_rel_dist_vectors, nb_robots_checked, nb_candidates, nb_further_check

    # Used for everything
    classes = [line.rstrip('\n') for line in open(classes_file)]
    min_objects = 3 # This is necessary for our method to work. It can be higher for more precision but will ignore more frames
    loop_closure_neighbors_ignored = 100 # Will ignore these number of following and previous frames. So this number * 2 frames ignored
    min_dist_rel_dist_vectors = 0.25 # d in the paper, used for the matches between constellations

    # Used for the decentralized simulation
    nb_robots_checked = range(1,21) # Number of robots used in the simulation
    nb_candidates = 4 # n_ret in the paper
    nb_further_check = 4 # n_fq in the paper

