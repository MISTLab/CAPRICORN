import settings
import numpy as np
from scene import Scene


def get_loop_closure_scores_cen(constel_files):
    print('Now calculating loop closures detected.')
    scenes = []
    scores = np.zeros((len(constel_files), len(constel_files)))
    for file in constel_files:
        scenes.append(Scene(file))

    # For every pair of frames evaluate the score and store it in a matrix
    for i in range(len(scenes)-settings.loop_closure_neighbors_ignored):
        print('Looking at frame '+str(i+1)+'/'+str(len(scenes)) )
        # If one scene had not enough objects, expect no match
        if scenes[i].ignore:
            scores[i, :] = -2.0
            scores[:,i] = -2.0
            continue
        for j in range(i+settings.loop_closure_neighbors_ignored, len(scenes)):
            # print('Comparing frame '+str(i+1)+'/'+str(len(scenes)) +' to frame '+str(j+1)+'/'+str(len(scenes)))
            # If one scene had not enough objects, expect no match
            if scenes[j].ignore:
                scores[i, j] = -2.0
                scores[j, i] = -2.0
                continue
            sem_score = scenes[i].descriptors.jaccard_score(
                scenes[j].descriptors)
            if sem_score > 0.:
                common_objects_labels = np.minimum(
                    scenes[i].descriptors.label_desc, scenes[j].descriptors.label_desc)
                nb_common_objects = sum(common_objects_labels)
                matches = scenes[i].descriptors.match_items(scenes[j].descriptors)

            else:

                matches = []
                nb_common_objects = np.inf
            total_score = sem_score*(len(matches)/nb_common_objects)
            scores[i, j] = total_score
            scores[j, i] = total_score

    # Remove scores for the neighbors of each frame
    for i in range(settings.loop_closure_neighbors_ignored+1):
        np.fill_diagonal(scores[i:], -1.0)
        np.fill_diagonal(scores[:, i:], -1.0)
    return scores
