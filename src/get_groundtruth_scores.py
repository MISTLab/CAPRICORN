import settings
import numpy as np
import math
from pyquaternion import Quaternion

def filter_dist_freiburg3(pose1_cam,pose2_cam,pose1_sc,pose2_sc,dist):
    """Function that takes two poses and the distance between two scenes and filers out scenes which shouldn't be recognized

    Arguments:
        pose1 {1D np array} -- Array containing pose tx ty tz qx qy qz qw for the camera taking one frame
        pose2 {1D np array} -- Array containing pose tx ty tz qx qy qz qw for the camera taking one frame
        dist {float} -- Computed distance between scenes viewed by cameras

    Returns:
        float  -- Unmodified float if not filtered out, inf if filtered out
    """
    # Desks have long side on x and face y
    # Desk 1 left side is limited around tx= 0.3623 to tx = 0.66 and ty = 1.2753.
    # Middle of two desks is ty = 0.0
    # Desk 2 right side is limited around tx= 0.3623 to tx = 0.66 and ty = -1.2753. Left side limited around tx = -2.0

    # Filter out when on different sides of the two desks
    if (pose1_cam[1] > 0.3 and pose2_cam[1] <-0.3) or (pose2_cam[1] > 0.3 and pose1_cam[1] <-0.3):
        return np.inf

    # Filter out when one pose completely on the sides and the other not, but keep possibility that the cameras are looking at the very same place
    if (np.abs(pose1_cam[1]) < 0.25 and (pose2_cam[0] > -1.8 and pose2_cam[0] < 0.5)):
        if np.abs(pose1_sc[0] - pose2_sc[0]) > 0.3:
            return np.inf

    if (np.abs(pose2_cam[1]) < 0.25 and (pose1_cam[0] > -1.8 and pose2_cam[0] < 0.5)):
        if np.abs(pose1_sc[0] - pose2_sc[0]) > 0.3:
            return np.inf
    return dist

def get_groundtruth_scores(constel_files, groundtruth_file):
    """Finds the distances between scenes based on positions and orientations information of frames. Estimates a scene position with distances to objects in it.

    Arguments:
        constel_files {list} -- List of strings representing the name of all the constellation txt files.
        groundtruth_file {string} -- String containing the path to the groundtruth txt file which has timestamps and poses

    Returns:
        2d numpy array -- Array of size nb_frames x nb_frames with the distance between two estimated scene positions.
    """
    # Get the timestamps of all the frames
    timestamps = np.array(
        [float(file.split('/')[-1].strip('.txt')) for file in constel_files])

    distances_frame = []
    # Get the average z positions (distances) of all the frames
    for constel_file in constel_files:
        z = 0.
        with open(constel_file) as f:
            point_from_objs = [line.split() for line in f]
        for line in point_from_objs:
            z += float(line[-1])
        if len(point_from_objs) > 0:
            z = z/len(point_from_objs)
        distances_frame.append(z)

    # Load all the poses
    poses_non_sync = np.loadtxt(groundtruth_file)

    # Get the poses closest to the timestamps of the frames
    poses_frames = []
    i = 0
    for time in timestamps:
        # Go through the times of the groundtruth data until we get a higher value than the one we have
        while (time > poses_non_sync[i][0]):
            i += 1
        # When we found the limit, choose the limit value or the one before depending on which is closest
        if (poses_non_sync[i][0]-time) > (time-poses_non_sync[i-1][0]):
            poses_frames.append(poses_non_sync[i-1][1:])
        else:
            poses_frames.append(poses_non_sync[i][1:])
    poses_frames = np.array(poses_frames)

    # Estimate points in front of the cone based on poses and distances
    points_in_front_cone = np.zeros((len(poses_frames),3))
    for i in range(len(poses_frames)):
        pose = poses_frames[i]
        vec_front_cam = np.array([0.,0.,distances_frame[i]])
        q = Quaternion([pose[-1],pose[3],pose[4],pose[5]])
        v_front_ori = q.rotate(vec_front_cam)
        points_in_front_cone[i] = pose[:3]+v_front_ori

    cones_distances = np.zeros((len(poses_frames),len(poses_frames)))
    for i in range(len(poses_frames)):
        for j in range(i,len(poses_frames)):
            pos1 = points_in_front_cone[i]
            pos2 = points_in_front_cone[j]
            dist = math.sqrt(sum((pos1-pos2)**2))

            # Filter the values based on manually imposed rules defined in the filter_dist_freiburg3 function
            dist = filter_dist_freiburg3(poses_frames[i],poses_frames[j],pos1,pos2,dist)

            cones_distances[i,j] = dist
            cones_distances[j,i] = dist

    # Filter ignored neighbors
    for i in range(settings.loop_closure_neighbors_ignored+1):
        np.fill_diagonal(cones_distances[i:], np.inf)
        np.fill_diagonal(cones_distances[:, i:], np.inf)

    # Put the scores between 0(= high distance) and 1 (= distance of 0)
    gt_scores = 1-(cones_distances /
                      np.max(cones_distances[cones_distances < 10000]))
    return gt_scores, cones_distances
