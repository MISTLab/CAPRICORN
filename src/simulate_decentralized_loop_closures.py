from scene import Scene
import settings
from parse_tools import get_constel_files_in_folder
import math
import numpy as np

# Split the frames to n robots
# For every frame of every robot, split the semantic descriptor into n parts
# Evaluate each part of each semantic descriptor with all current and previous frames of each robot
# For each subpart of the semantic descriptor return the best match to each robot
# For each robot decide which matches to look further into
# Get relative descriptor with the chosen matches
# Store the resulting score


def split_labels(labels, nb_robots):
    labels_per_robot = []
    nb_classes_per_robot = round(len(labels)/nb_robots)

    # Separate them among robots
    for i in range(0, nb_classes_per_robot*(nb_robots-1), nb_classes_per_robot):
        labels_per_robot.append(labels[i:i + nb_classes_per_robot])
    labels_per_robot.append(labels[nb_classes_per_robot*(nb_robots-1):])
    return labels_per_robot


def assign_labels(nb_robots):
    assignments = np.zeros(len(settings.classes))
    nb_classes_per_robot = round(len(assignments)/nb_robots)

    # Separate them among robots
    for i in range(0, nb_robots-1):
        assignments[i*nb_classes_per_robot:(i+1)*nb_classes_per_robot] = i
    assignments[nb_classes_per_robot*(nb_robots-1):] = nb_robots-1
    return assignments


def split_frames(constellations_folder, nb_robots):
    """Splits the frames of all sequences around all robots

    Arguments:
        sequences_folders {list of strings} -- Strings containing the paths of the folders of each sequence
        nb_robots {int} -- Number of robots in the experiment

    Returns:
        list of list of strings -- Contains a list with one sublist for each robot. Each sublist contains strings of paths to the constel files.
    """

    constel_files = get_constel_files_in_folder(constellations_folder)

    constel_files_per_robot = []
    nb_frames_per_robot = math.ceil(len(constel_files)/nb_robots)

    # Separate them among robots
    for i in range(0, nb_frames_per_robot*(nb_robots-1), nb_frames_per_robot):
        constel_files_per_robot.append(constel_files[i:i + nb_frames_per_robot])
    constel_files_per_robot.append(constel_files[nb_frames_per_robot*(nb_robots-1):])
    nb_constel_files = len(constel_files)
    return constel_files_per_robot,nb_constel_files


def get_loop_closure_scores_dec(constel_folder,nb_robots):

    nb_classes = len(settings.classes)
    nb_candidates = settings.nb_candidates
    nb_robots_further_check = settings.nb_further_check
    constel_files_per_robot, nb_constel_files = split_frames(constel_folder, nb_robots)
    nb_frames_per_robot = len(constel_files_per_robot[0])
    semantic_descriptors = np.zeros(
        (nb_robots, nb_frames_per_robot, len(settings.classes)))

    all_scenes = [[] for i in range(nb_robots)]

    # Register new scene and store label descriptors
    for rob_i in range(nb_robots):
        for cur_frame_i in range(len(constel_files_per_robot[rob_i])):
            new_scene = Scene(constel_files_per_robot[rob_i][cur_frame_i])
            all_scenes[rob_i].append(new_scene)
            if new_scene.ignore:
                labels_desc = np.zeros(
                    len(settings.classes)).astype(int)
            else:
                labels_desc = new_scene.descriptors.label_desc

            semantic_descriptors[rob_i, cur_frame_i, :] = labels_desc

    # If last robot got less frames, fill with empty frames to have the same number as the others.
    while (len(all_scenes[-1]) < nb_frames_per_robot):
                new_scene = Scene(0, empty=True)
                all_scenes[-1].append(new_scene)
                # No need to fill the semantic descriptors since it was initialized with zeros

    print('Finished registering scenes')
    nb_sub_semantic = nb_robots
    semantic_splits_idx = np.linspace(0, nb_classes, nb_robots+1, dtype=int)

    # We need tgree values for every candidate : one for the robot id, one for the frame id and one for the score. We subtract 1 so that by default no robot is verified (id = -1)
    robots_to_verify = np.zeros(
        (nb_robots, nb_frames_per_robot, nb_sub_semantic, nb_candidates, 3))-1

    # Process all received labels and note most similars
    for rob_i in range(nb_robots):
        # Go through every frame of rob_i
        for frame_i in range(1, nb_frames_per_robot):
            semantic_ref = semantic_descriptors[rob_i, frame_i, :]

            # Prepare to compute Jaccard score on subsets up to frame i
            common_objects = np.minimum(
                semantic_descriptors[:, :frame_i, :], semantic_ref)
            total_objects = semantic_descriptors[:,
                                                    :frame_i, :] + semantic_ref - common_objects

            # Go through each sub semantic descriptor
            for sem_i in range(len(semantic_splits_idx)-1):
                common_objects_sub = common_objects[:, :,
                                                    semantic_splits_idx[sem_i]:semantic_splits_idx[sem_i+1]]
                total_objects_sub = total_objects[:, :,
                                                    semantic_splits_idx[sem_i]:semantic_splits_idx[sem_i+1]]
                jaccard_scores_sublabel = np.sum(
                    common_objects_sub, axis=2)/(np.sum(total_objects_sub, axis=2)+1e-12)
                max_idxs = np.unravel_index(
                    np.argsort(-jaccard_scores_sublabel.flatten()), jaccard_scores_sublabel.shape)

                # Find candidates which have the best scores
                cand_found = 0
                top_i = 0
                while cand_found < nb_candidates and top_i < len(jaccard_scores_sublabel.flatten()):
                    # If the score is too low, stop looking
                    sem_score = jaccard_scores_sublabel[max_idxs[0]
                                                        [top_i], max_idxs[1][top_i]]
                    if sem_score <= 0.:
                        break

                    robots_to_verify[rob_i, frame_i, sem_i,
                                        cand_found, 0] = max_idxs[0][top_i]
                    robots_to_verify[rob_i, frame_i, sem_i,
                                        cand_found, 1] = max_idxs[1][top_i]
                    robots_to_verify[rob_i, frame_i,
                                        sem_i, cand_found, 2] = sem_score
                    cand_found += 1
                    top_i += 1
    print('Done choosing robots to verify')

    print('Number of robots : '+str(nb_robots))
    print('Number of candidates : '+str(nb_candidates))
    print('Number of robots further checked : '+str(nb_robots_further_check))
    'nb_robots_'+ str(nb_robots)+'_nb_cand_'+str(nb_candidates)+'_rob_furth_check_'+str(nb_robots_further_check)+'_min_dist_'+str(settings.min_dist_rel_dist_vectors)

    scores = np.zeros((nb_constel_files, nb_constel_files))
    # Now go through robots to verify and look for relative distance score
    for rob_i in range(nb_robots):
        # print('Looking at full scores for robot '+str(rob_i))
        for frame_i in range(1, len(constel_files_per_robot[rob_i])):
            # print('Looking at full scores during frame '+str(frame_i))
            sc1_glob_frame = rob_i*nb_frames_per_robot+frame_i
            sc1 = all_scenes[rob_i][frame_i]
            if sc1.ignore:
                scores[sc1_glob_frame, :] = -2.0
                scores[:, sc1_glob_frame ]= -2.0
                continue

            all_cand = robots_to_verify[rob_i, frame_i, :,:,:].reshape(-1,3)

            # Remove all entries full of -1 due to lack of candidates
            all_cand = all_cand[all_cand[:,0] >= 0]

            # rob_multiple_occurences = np.argwhere(np.bincount(all_cand[:, 0].astype(int)) > min_robots_matched)
            most_common_robots_ordered = np.argsort(-np.bincount(all_cand[:, 0].astype(int)))
            checked_robots = np.zeros(min(len(most_common_robots_ordered),nb_robots_further_check)+1)
            if len(most_common_robots_ordered) > 0:
                checked_robots[:min(len(most_common_robots_ordered),nb_robots_further_check)] = most_common_robots_ordered[:nb_robots_further_check]
            checked_robots[-1] = rob_i
            for rob_multi_val in checked_robots:
                cand_same_rob = all_cand[all_cand[:,0] == rob_multi_val]
                for cand_i in range(len(cand_same_rob)):
                    rob_cand = int(cand_same_rob[cand_i, 0])
                    frame_cand = int(cand_same_rob[cand_i, 1])

                    sc2 = all_scenes[rob_cand][frame_cand]
                    sc2_glob_frame = rob_cand*nb_frames_per_robot+frame_cand
                    if sc2.ignore:
                        scores[sc1_glob_frame, sc2_glob_frame] = -2.0
                        scores[sc2_glob_frame, sc1_glob_frame] = -2.0
                        continue
                    sem_score = sc1.descriptors.jaccard_score(sc2.descriptors)
                    if sem_score > 0.:
                        common_objects_labels = np.minimum(
                            sc1.descriptors.label_desc, sc2.descriptors.label_desc)
                        nb_common_objects = sum(common_objects_labels)
                        matches = sc1.descriptors.match_items(
                            sc2.descriptors)
                    else:
                        matches = []
                        nb_common_objects = np.inf
                    total_score = sem_score *(len(matches)/nb_common_objects)
                    scores[sc1_glob_frame, sc2_glob_frame] = total_score
                    scores[sc2_glob_frame, sc1_glob_frame] = total_score

    # Remove the neighboring frames
    for i in range(settings.loop_closure_neighbors_ignored+1):
        np.fill_diagonal(scores[i:], -1.0)
        np.fill_diagonal(scores[:, i:], -1.0)


    return scores, robots_to_verify, semantic_descriptors


