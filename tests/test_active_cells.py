import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv

from toughitup_v2 import directories


class ActiveCellTests(unittest.TestCase):
    def test_keyword_with_inline_comment(self):
        with tempfile.TemporaryDirectory() as directory:
            directories.set_output_directory(directory)
            from toughitup_v2 import extractors

            source = Path(directory) / "keywords.grdecl"
            source.write_text("ACTNUM -- Generated : Petrel\n1 0 /\n")
            self.assertIn("ACTNUM", extractors.find_keywords(source))

    def convert(self, directory, actnum):
        directories.set_output_directory(directory)
        from toughitup_v2.toughitup import GeoModel

        coord = [
            value
            for y in range(3)
            for x in range(3)
            for value in (x, y, 0, x, y, 100)
        ]
        source = Path(directory) / "active.grdecl"
        source.write_text(
            "SPECGRID\n2 2 2 1 F /\nGRIDUNIT\n'METRES' /\n"
            "COORD -- Generated : Petrel\n" + " ".join(map(str, coord))
            + " /\nZCORN -- Generated : Petrel\n16*10 16*20 16*20 16*30 /\n"
            + ("" if actnum is None else f"ACTNUM -- Generated : Petrel\n{actnum} /\n"),
            encoding="utf-8",
        )
        model = GeoModel(source)
        with contextlib.redirect_stdout(io.StringIO()):
            model.convert_model()
        return pv.read(Path(directory) / "grid.vtu")

    def test_mask_uses_i_fastest_order_across_layers(self):
        with tempfile.TemporaryDirectory() as directory:
            grid = self.convert(directory, "0 1 3*0 1 2*0")
            self.assertEqual(grid.n_cells, 2)
            np.testing.assert_allclose(
                grid.cell_centers().points,
                [[1.5, 0.5, -15], [1.5, 0.5, -25]],
            )
            self.assertEqual(grid.n_points, 12)

    def test_missing_actnum_keeps_all_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.convert(directory, None).n_cells, 8)

    def test_all_active_keeps_all_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(self.convert(directory, "8*1").n_cells, 8)

    def test_all_inactive_exports_empty_grid(self):
        with tempfile.TemporaryDirectory() as directory:
            grid = self.convert(directory, "8*0")
            self.assertEqual(grid.n_cells, 0)
            self.assertEqual(grid.n_points, 0)

    def test_invalid_mask_is_rejected(self):
        for mask, message in (
            ("7*1", "must contain 8 values"),
            ("7*1 2", "must be 0"),
            ("7*1 nan", "must be 0"),
        ):
            with self.subTest(mask=mask), tempfile.TemporaryDirectory() as directory:
                with self.assertRaisesRegex(ValueError, message):
                    self.convert(directory, mask)


if __name__ == "__main__":
    unittest.main()
