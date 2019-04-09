"""File containing the definition of all descriptor classes.
"""
import numpy as np
import settings
from scipy.spatial.distance import pdist,squareform,cdist

class Descriptors:
    """Descriptors are arrays like bag of words.
    """

    def __init__(self, constellation):
        self.labels = [constel.label for constel in constellation]
        self.positions = np.array([[constel.x, constel.y, constel.z] for constel in constellation])
        self.distances = pdist(self.positions)
        self.distances_mat = squareform(self.distances)
        self.__create_label_desc__()

    def __create_label_desc__(self):
        self.label_desc = np.zeros(len(settings.classes)).astype(int)

        for label in self.labels:
            # Add 1 to the bin corresponding to the label
            self.label_desc[settings.classes.index(label)] += 1

    def jaccard_score(self, desc):
        """Computes the Jaccard score between two histogram of labels

        Arguments:
            desc {descriptor object} -- descriptor of the scene which is compared to the self object

        Returns:
            float -- value of the Jaccard score
        """
        common_items = float(
            np.sum(np.minimum(self.label_desc, desc.label_desc)))
        return common_items / (np.sum(self.label_desc) + np.sum(desc.label_desc) - common_items)

    def get_obj_ids(self, obj_labels):
        """ Function that returns the identifiers of all the instances of a specific class in a scene

        Arguments:
        obj_label {list of strings representing labels}

        Returns:
        list of the IDs of the objects in the scene for each label received
        """
        ids = []
        for obj_label in obj_labels:
            ids.append([id for id, lbl in enumerate(
                self.labels) if lbl == obj_label])
        return ids

    def match_items(self, desc):
        """Function that returns the matched elements in the scene that minimize the rmse error. It associates each instance of one scene to an instance of the other if possible.

        Arguments:
            desc {Descriptor object} -- Descriptor of the scene to be compared to self

        Returns:
            list of tuples [(id11,id21),(id12,id22...)] -- List containg as many tuples as matches. The tuples contain the ID of one object of the scene associated to self and the ID of the matched object of the scene associated to desc.
        """

        # Put a zero in the label descriptor when that label doesn't appear in another scene.
        sc1_labels_to_keep = self.label_desc * \
            np.logical_and(self.label_desc, desc.label_desc)
        sc2_labels_to_keep = desc.label_desc * \
            np.logical_and(self.label_desc, desc.label_desc)
        common_objects_labels = np.minimum(sc1_labels_to_keep, sc2_labels_to_keep)
        nb_common_objects = sum(common_objects_labels)

        desc1_distances = []
        desc2_distances = []

        # Check if there are enough matches to continue
        if nb_common_objects < settings.min_objects:
            return []

        # if len(sc1_labels_to_keep.nonzero()) < settings.min_items_match
        obj_ids_sc1 = self.get_obj_ids(
            [settings.classes[i] for i in sc1_labels_to_keep.nonzero()[0]])
        obj_ids_sc2 = desc.get_obj_ids(
            [settings.classes[i] for i in sc2_labels_to_keep.nonzero()[0]])

        for objects_of_one_label in obj_ids_sc1:
            for obj in objects_of_one_label:
                obj_vector = np.zeros((len(obj_ids_sc1)))
                for label_searched_i in range(len(obj_ids_sc1)):
                    closest_dist = np.amin(
                        self.distances_mat[obj][obj_ids_sc1[label_searched_i]])

                    obj_vector[label_searched_i] = closest_dist
                desc1_distances.append(obj_vector)


        for objects_of_one_label in obj_ids_sc2:
            for obj in objects_of_one_label:
                obj_vector = np.zeros((len(obj_ids_sc2)))
                for label_searched_i in range(len(obj_ids_sc2)):
                    closest_dist = np.amin(
                        desc.distances_mat[obj][obj_ids_sc2[label_searched_i]])

                    obj_vector[label_searched_i] = closest_dist
                desc2_distances.append(obj_vector)

        desc1_distances = np.array(desc1_distances)
        desc2_distances = np.array(desc2_distances)

        distances_rel_dist_vectors = cdist(desc1_distances, desc2_distances)
        closest_points_1 = np.argmin(distances_rel_dist_vectors, axis=1)
        closest_points_2 = np.argmin(
            cdist(desc1_distances, desc2_distances), axis=0)


        flat_list_ids1 = [
            idx for idx_per_class in obj_ids_sc1 for idx in idx_per_class]
        flat_list_ids2 = [
            idx for idx_per_class in obj_ids_sc2 for idx in idx_per_class]
        matches = []


        for i in range(len(closest_points_1)):
            if closest_points_2[closest_points_1[i]] == i:
                if distances_rel_dist_vectors[i, closest_points_1[i]] < settings.min_dist_rel_dist_vectors :
                    matches.append(
                        (flat_list_ids1[i], flat_list_ids2[closest_points_1[i]]))
        return matches


