import numpy as np
import pyvista as pv


class GeoModel:
    def __init__(self, file_path, invert_y_axis = True):
        self.file_path = file_path
        self.invert_y_axis = invert_y_axis

    def convert_model(self):
        pass
    