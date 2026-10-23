import os
import numpy as np
import pyvista as pv


from . import reports
from . import directories
from . import extractors
from . import functions


class GeoModel:
    def __init__(self, file_path, invert_y_axis = False, format = None):
        self.file_path = file_path
        self.invert_y_axis = invert_y_axis
        self.__format = format
        self.script_dir = directories.MAIN_SCRIPT_DIR
        self.__file = extractors.read_grdecl_file(self.file_path)

    def convert_model(self):
        self.parse_model()
        self.create_geometry()
        self.extract_data()
        self.create_geometry_file(self.__format)
        reports.print_completion()


    def parse_model(self):
        reports.print_initialization()
        reports.start_report(reports.banner)
        self.__title_import = reports.print_chapter_report("Importing GRDECL...")
        reports.print_in_report(self.__title_import)
        self.__keywords = extractors.find_keywords(self.file_path)
        reports.print_keywords_report(self.__keywords)


    def create_geometry(self):
        """Build all logical cells in Eclipse order before extracting cell data."""

        if "SPECGRID" in self.__keywords:
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

        self.__zcorn = extractors.extract_data_to_np(self.__file, "ZCORN")
        self.__coord = extractors.extract_data_to_np(self.__file, "COORD")

        reports.print_in_report(f"\n{reports.block_heather_A}")
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
        self.__geometry[:, :, 2] *= -1.0

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
        hex_cells_int = np.array(hex_cells, dtype=int).reshape(self.__n_cells, 8)

        self.grid = pv.UnstructuredGrid(
            np.hstack([np.full((self.__n_cells, 1), 8), hex_cells_int]),
            np.full(self.__n_cells, pv.CellType.HEXAHEDRON),
            self.__unique_corners
        )

    
    def extract_data(self):
        """Load cell properties, then remove inactive cells and their property values."""

        for keyword in self.__keywords:
            if keyword.upper() in functions.keywords_accepted:
                if keyword == "COPY":
                    self.__copies = extractors.extract_pairs(self.__file, "COPY")
                    reports.print_in_report(f"Copying {self.__copies[0]} to {self.__copies[1]}...")
                    self.grid.cell_data[self.__copies[1]] = self.grid.cell_data[self.__copies[0]]
                    reports.print_in_report(" - ✅ COPY applied correctly.\n")

                elif keyword == "MULTIPLY":
                    self.__multiplies = extractors.extract_pairs(self.__file, "MULTIPLY")
                    reports.print_in_report(f"Multiplying {self.__multiplies[0]} by {self.__multiplies[1]}...")
                    self.grid.cell_data[self.__multiplies[0]] *= float(self.__multiplies[1])
                    reports.print_in_report(" - ✅ MULTIPLY applied correctly.\n")

                else:
                    reports.print_in_report(f"Extracting data for keyword: {keyword}...")
                    data = extractors.extract_data_to_np(self.__file, keyword)
                    if keyword.upper() == "ACTNUM":
                        if data.size != self.grid.n_cells:
                            raise ValueError(
                                f"ACTNUM must contain {self.grid.n_cells} values; "
                                f"got {data.size}."
                            )
                        if not np.all((data == 0) | (data == 1)):
                            raise ValueError(
                                "ACTNUM values must be 0 (inactive) or 1 (active)."
                            )
                    self.grid.cell_data[keyword] = data
                    reports.print_in_report(" - ✅ Imported correctly.\n")

        active_cells = np.ones(self.grid.n_cells, dtype=bool)
        for name in self.grid.cell_data:
            if name.casefold() == "actnum":
                active_cells &= self.grid.cell_data[name] != 0

        n_cells = self.grid.n_cells
        if not np.all(active_cells):
            self.grid = self.grid.extract_cells(
                active_cells, pass_cell_ids=False, pass_point_ids=False
            )
        reports.print_in_report(
            f" - Active cells: {self.grid.n_cells} of {n_cells}; "
            f"excluded {n_cells - self.grid.n_cells} inactive cells.\n"
        )

    def create_geometry_file(self, format: str | None = None):
        """Save the selected format, or both formats for None (case-insensitive)."""
        normalized = format.casefold() if format is not None else "none"
        if normalized not in ("vtk", "vtu", "none"):
            raise ValueError("Output format must be vtk, vtu, or None.")

        if normalized in ("vtk", "none"):
            my_file_vtk = os.path.join(self.script_dir, "grid.vtk")
            self.grid.save(my_file_vtk)
            reports.print_in_report(f" - ✅ File grid.vtk created at {self.script_dir}")

        if normalized in ("vtu", "none"):
            my_file_vtu = os.path.join(self.script_dir, "grid.vtu")
            self.grid.save(my_file_vtu)
            reports.print_in_report(f" - ✅ File grid.vtu created at {self.script_dir}")
        
        