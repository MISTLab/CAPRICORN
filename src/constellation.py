"""File containing the definition of the constellation class.
"""

class constel_point:
    """class that describes a point of a constellation
    """
    def __init__(self,point_from_obj):
        # Fix labels which have spaces and take several elements for the label name instead of the first one
        point_from_obj[0:len(point_from_obj) - 3] = [' '.join(point_from_obj[0:len(point_from_obj) - 3])]
        self.label = point_from_obj[0]
        self.x = float(point_from_obj[1])
        self.y = float(point_from_obj[2])
        self.z = float(point_from_obj[3])


