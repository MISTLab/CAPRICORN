"""File containing the definition of the scene class
"""
from constel_point import constel_point
from descriptors import Descriptors
import settings


class Scene:
    def __init__(self, constel_file,empty=False):
        # Extract time stamp from the name of the file
        # self.timestamp = float(constel_file.split('/')[-1].strip('.txt'))

        if not empty:
            # Read information from file to list of list
            with open(constel_file) as f:
                point_from_objs = [line.split() for line in f]

        else:
            point_from_objs = []

        # point_from_objs = [line.split() for line in open(constel_file)] # this file is never closed... will the garbage collector close it?
        if len(point_from_objs) < settings.min_objects:
            self.ignore = True
        else:
            self.constellation = []

            self.ignore = False
            # Create list of constellation
            self.constellation = [constel_point(
                point_from_obj) for point_from_obj in point_from_objs]

            # Get descriptors
            self.descriptors = Descriptors(self.constellation)
