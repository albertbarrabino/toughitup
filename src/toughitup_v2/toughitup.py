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
        """Build active cells in Eclipse order, or all cells when ACTNUM is absent."""

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

        active_cells = np.ones(self.__n_cells, dtype=bool)
        if "ACTNUM" in self.keywords:
            actnum = extractors.extract_data_to_np(self.__file, "ACTNUM")
            if actnum.size != self.__n_cells:
                raise ValueError(
                    f"ACTNUM must contain {self.__n_cells} values; got {actnum.size}."
                )
            if not np.all((actnum == 0) | (actnum == 1)):
                raise ValueError("ACTNUM values must be 0 (inactive) or 1 (active).")
            active_cells = actnum == 1
        n_active_cells = int(np.count_nonzero(active_cells))
        reports.print_in_report(
            f" - Active cells: {n_active_cells} of {self.__n_cells}; "
            f"excluded {self.__n_cells - n_active_cells} inactive cells.\n"
        )

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
            self.__ecl_indexes[active_cells],
            n_active_cells,
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
        for i in range(n_active_cells):
            cell_idx = self.__idx[i*8:(i+1)*8]
            hex_cells.append(cell_idx)
        hex_cells_int = np.array(hex_cells, dtype=int).reshape(n_active_cells, 8)

        self.grid = pv.UnstructuredGrid(
            np.hstack([np.full((n_active_cells, 1), 8), hex_cells_int]),
            np.full(n_active_cells, pv.CellType.HEXAHEDRON),
            self.__unique_corners
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
        
        