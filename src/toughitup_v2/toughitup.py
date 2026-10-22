import os
import numpy as np
import pyvista as pv


from . import reports
from . import directories
from . import extractors
from . import functions


class GeoModel:
    def __init__(self, file_path, invert_y_axis = False):
        self.file_path = file_path
        self.invert_y_axis = invert_y_axis
        self.script_dir = directories.MAIN_SCRIPT_DIR
        self.__file = extractors.read_grdecl_file(self.file_path)

    def convert_model(self):
        self.parse_model()
        self.create_geometry()
        self.create_geometry_file()


    def parse_model(self):
        reports.print_initialization()
        reports.start_report(reports.banner)
        self.__title_import = reports.print_chapter_report("Importing GRDECL...")
        reports.print_in_report(self.__title_import)
        self.keywords = extractors.find_keywords(self.file_path)
        reports.print_keywords_report(self.keywords)


    def create_geometry(self):

        if "SPECGRID" in self.keywords:
            self.__specgrid = extractors.extract_data_list(self.__file, "SPECGRID")
        else:
            self.__specgrid = extractors.extract_data_list(self.__file, "DIMENS")
        reports.print_in_report("\n")

        self.__nx = self.__specgrid[0]
        self.__ny = self.__specgrid[1]
        self.__nz = self.__specgrid[2]
        self.__n_cells = self.__nx * self.__ny * self.__nz
        self.__ecl_indexes = functions.get_ref_sys(self.__nx, self.__ny, self.__nz)

        self.__units = extractors.get_units(self.__file, "GRIDUNIT")
        reports.print_in_report(f" - Grid units: {self.__units}\n")

        self.__zcorn = extractors.extract_data_to_np(self.__file, "ZCORN") * -1.0
        self.__coord = extractors.extract_data_to_np(self.__file, "COORD")

        reports.print_in_report(reports.print_chapter_report("Building Geometry"))

        self.__geometry = functions.build_cells(
            self.__zcorn,
            self.__coord,
            self.__nx,
            self.__ny,
            self.__ecl_indexes,
            self.__n_cells,
            self.__units
        )

        reports.print_cell_structure()
        self.__cells_reshaped = self.__geometry.reshape(-1, 3)
        self.__unique_corners, self.__idx = np.unique(self.__cells_reshaped, axis=0, return_inverse = True)

        if self.invert_y_axis:
            self.__unique_corners[:,1] *= -1
            reports.print_in_report("Y-axis has been inverted...\n")

        hex_cells = []
        for i in range(self.__n_cells):
            cell_idx = self.__idx[i*8:(i+1)*8]
            hex_cells.append(cell_idx)
        hex_cells_int = np.array(hex_cells, dtype=int)  # shape (n_cells, 8)

        self.grid = pv.UnstructuredGrid(
            np.hstack([np.full((self.__n_cells, 1), 8), hex_cells_int]),
            np.full(self.__n_cells, pv.CellType.HEXAHEDRON),
            self.__unique_corners
        )

    def create_geometry_file(self):
        my_file_vtk = os.path.join(self.script_dir, "grid.vtk")
        reports.print_in_report(f" - ✅ File grid.vtk created at {self.script_dir}")
        self.grid.save(my_file_vtk)

        my_file_vtu = os.path.join(self.script_dir, "grid.vtu")
        reports.print_in_report(f" - ✅ File grid.vtu created at {self.script_dir}")
        self.grid.save(my_file_vtu)
        
        