import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pyvista as pv

from toughitup_v2 import directories


SOURCE = """SPECGRID
1 1 1 1 F /
GRIDUNIT
'METRES' /
COORD
0 0 0 0 0 10
1 0 0 1 0 10
0 1 0 0 1 10
1 1 0 1 1 10 /
ZCORN
4*0 4*10 /
"""


class OutputFormatTests(unittest.TestCase):
    def check_outputs(self, folder, extensions):
        self.assertEqual(
            {path.suffix for path in folder.glob("grid.*")}, set(extensions)
        )
        for extension in extensions:
            self.assertEqual(pv.read(folder / f"grid{extension}").n_cells, 1)

    def test_cli_output_formats(self):
        for value, extensions in (
            (None, (".vtk", ".vtu")),
            ("None", (".vtk", ".vtu")),
            ("nOnE", (".vtk", ".vtu")),
            ("vtk", (".vtk",)),
            ("VTK", (".vtk",)),
            ("vtu", (".vtu",)),
            ("VtU", (".vtu",)),
        ):
            with self.subTest(format=value), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                (folder / "input.grdecl").write_text(SOURCE, encoding="utf-8")
                args = ["input.grdecl", "-i", "False"]
                if value is not None:
                    args.extend(["-f", value])
                result = subprocess.run(
                    [sys.executable, "-c", "from toughitup_v2.cli import main; main()", *args],
                    cwd=folder,
                    env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.check_outputs(folder, extensions)

    def test_invalid_cli_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", "from toughitup_v2.cli import main; main()",
                 "input.grdecl", "-f", "obj"],
                cwd=directory,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("invalid choice", result.stderr)
            self.assertFalse(list(Path(directory).glob("grid.*")))

    def test_library_output_formats(self):
        for value, extensions in (
            (None, (".vtk", ".vtu")),
            ("NONE", (".vtk", ".vtu")),
            ("VTK", (".vtk",)),
            ("VTU", (".vtu",)),
        ):
            with self.subTest(format=value), tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                directories.set_output_directory(folder)
                from toughitup_v2.toughitup import GeoModel

                source = folder / "input.grdecl"
                source.write_text(SOURCE, encoding="utf-8")
                with contextlib.redirect_stdout(io.StringIO()):
                    GeoModel(source, format=value).convert_model()
                self.check_outputs(folder, extensions)

    def test_invalid_library_format_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            directories.set_output_directory(folder)
            from toughitup_v2.toughitup import GeoModel

            source = folder / "input.grdecl"
            source.write_text(SOURCE, encoding="utf-8")
            model = GeoModel(source)
            with self.assertRaisesRegex(ValueError, "Output format"):
                model.create_geometry_file("obj")
            self.assertFalse(list(folder.glob("grid.*")))


if __name__ == "__main__":
    unittest.main()
