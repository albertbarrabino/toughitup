import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv

from toughitup_v2.functions import build_cells, get_info, get_ref_sys


class PillarInterpolationTests(unittest.TestCase):
    def setUp(self):
        self.pillars = np.array(
            [[x, y, 0, x + 10, y + 20, 100]
             for x, y in ((0, 0), (1, 0), (1, 1), (0, 1))],
            dtype=float,
        )

    def test_slanted_pillars_use_each_corner_depth(self):
        depths = np.array([10, 20, 30, 40, 60, 70, 80, 90], dtype=float)
        result = get_info(self.pillars, depths)
        starts = self.pillars[[0, 1, 2, 3, 0, 1, 2, 3], :2]
        expected_xy = starts + depths[:, None] * np.array([0.1, 0.2])
        np.testing.assert_allclose(result[:, :2], expected_xy)
        np.testing.assert_array_equal(result[:, 2], depths)

    def test_vertical_pillars_allow_equal_endpoint_depths(self):
        self.pillars[:, 3:] = self.pillars[:, :3]
        depths = np.array([25] * 4 + [75] * 4, dtype=float)
        result = get_info(self.pillars, depths)
        np.testing.assert_array_equal(
            result[:, :2], self.pillars[[0, 1, 2, 3, 0, 1, 2, 3], :2]
        )
        np.testing.assert_array_equal(result[:, 2], depths)

    def test_nonvertical_pillar_with_equal_depths_raises(self):
        self.pillars[0, 5] = self.pillars[0, 2]
        with self.assertRaisesRegex(ValueError, "equal endpoint depths"):
            get_info(self.pillars, np.ones(8))

    def test_extrapolation_and_reversed_endpoints(self):
        depths = np.array([-25] * 4 + [125] * 4, dtype=float)
        forward = get_info(self.pillars, depths)
        reversed_pillars = np.hstack((self.pillars[:, 3:], self.pillars[:, :3]))
        np.testing.assert_allclose(get_info(reversed_pillars, depths), forward)
        np.testing.assert_allclose(forward[0], [-2.5, -5, -25])
        np.testing.assert_allclose(forward[4], [12.5, 25, 125])

    def test_documentation_zcorn_order(self):
        coord = np.array(
            [[x, y, 1000, x, y, 1000]
             for y in (0, 2000, 4000) for x in (0, 1000, 2000, 3000)],
            dtype=float,
        )
        depths = np.repeat([1000., 1020., 1020., 1040.], 24)
        cells = build_cells(depths, coord, 3, 2, get_ref_sys(3, 2, 2), 12, "METRES")
        np.testing.assert_array_equal(cells[:6, :4, 2], 1000.)
        np.testing.assert_array_equal(cells[:6, 4:, 2], 1020.)
        np.testing.assert_array_equal(cells[6:, :4, 2], 1020.)
        np.testing.assert_array_equal(cells[6:, 4:, 2], 1040.)

    def test_geometry_and_vtu_preserve_interpolated_positions(self):
        from toughitup_v2 import directories

        with tempfile.TemporaryDirectory() as directory:
            directories.set_output_directory(directory)
            from toughitup_v2.toughitup import GeoModel

            coord = self.pillars[[0, 1, 3, 2]].ravel()
            source = Path(directory) / "slanted.grdecl"
            source.write_text(
                "SPECGRID\n1 1 1 1 F /\nGRIDUNIT\n'METRES' /\n"
                "COORD\n" + " ".join(map(str, coord))
                + " /\nZCORN\n4*25 4*75 /\n",
                encoding="utf-8",
            )
            expected = np.array(
                [[2.5, 5, -25], [3.5, 5, -25],
                 [3.5, 6, -25], [2.5, 6, -25],
                 [7.5, 15, -75], [8.5, 15, -75],
                 [8.5, 16, -75], [7.5, 16, -75]],
            )
            for invert_y in (False, True):
                with self.subTest(invert_y=invert_y):
                    model = GeoModel(source, invert_y_axis=invert_y)
                    model.convert_model()
                    grid = pv.read(Path(directory) / "grid.vtu")
                    target = expected.copy()
                    if invert_y:
                        target[:, 1] *= -1
                    self.assertEqual(grid.n_cells, 1)
                    np.testing.assert_allclose(grid.get_cell(0).points, target)


if __name__ == "__main__":
    unittest.main()
