import numpy as np

def my_pr(groundtruth_closures,detected_scores):

    # For each frame we get the best detected matching frame score
    best_detected_matches_idx = np.argmax(detected_scores,1)
    best_detected_matches_scores = detected_scores[np.arange(detected_scores.shape[0]),best_detected_matches_idx]
    # Order the frames by highest scores and keep the indices
    sorted_best_matches_idx = np.argsort(-best_detected_matches_scores)
    tp_on_select = np.zeros(len(best_detected_matches_idx))
    fp_on_select = np.zeros(len(best_detected_matches_idx))
    # Check if the matched frames are valid
    for k in range(len(sorted_best_matches_idx)) :
        i = sorted_best_matches_idx[k]
        j =  best_detected_matches_idx[i]
        tp_on_select[k] = groundtruth_closures[i,j]
        fp_on_select[k] = not groundtruth_closures[i,j]
    tp_cs = np.cumsum(tp_on_select)
    fp_cs = np.cumsum(fp_on_select)
    # Compute the number of matches expected by checking for truths in every row of the groundtruth_closures
    relevant_matches = np.sum(np.amax(groundtruth_closures,1))
    fn_rcs = relevant_matches- tp_cs
    precision = tp_cs / (tp_cs + fp_cs + 1e-12)
    recall = tp_cs / (tp_cs + fn_rcs + 1e-12)

    return precision,recall
