import numpy as np
import pyvista as pv

from . import reports
from . import directories
from . import extractors


class GeoModel:
    def __init__(self, file_path, invert_y_axis = False):
        self.file_path = file_path
        self.invert_y_axis = invert_y_axis
        self.script_dir = directories.MAIN_SCRIPT_DIR
        self.__file = extractors.read_grdecl_file(self.file_path)

    def convert_model(self):
        self.parse_model()

    def parse_model(self):
        reports.print_initialization()
        reports.start_report(reports.banner)
        keywords = extractors.find_keywords(self.file_path)
        reports.print_keywords_report(keywords)
