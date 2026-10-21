import numpy as np
import pyvista as pv

from . import reports
from . import directories


class GeoModel:
    def __init__(self, file_path, invert_y_axis = False):
        self.file_path = file_path
        self.invert_y_axis = invert_y_axis
        self.script_dir = directories.MAIN_SCRIPT_DIR

    def convert_model(self):
        pass

    def parse_model(self):
        reports.print_initialization()
        reports.start_report(reports.banner)
        
