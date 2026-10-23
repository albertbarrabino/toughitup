import argparse
from pathlib import Path


def _parse_bool(value: str) -> bool:
    normalized = value.casefold()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise argparse.ArgumentTypeError("expected True or False")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="toughitup",
        description="Convert a file in the current working directory.",
    )
    parser.add_argument(
        "filename",
        help="name of the file to convert (it must be in the current directory)",
    )
    parser.add_argument(
        "-i",
        type=_parse_bool,
        default=False,
        metavar="True|False",
        help="invert the y axis (default: False)",
    )
    parser.add_argument(
        "-f",
        type=str.casefold,
        choices=("vtu", "vtk", "none"),
        default=None,
        metavar="vtu|vtk|None",
        help="output format, case-insensitive (default: None saves both formats)",
    )
    args = parser.parse_args()

    if Path(args.filename).name != args.filename:
        parser.error("filename must not include a path")
    if not Path(args.filename).is_file():
        parser.error(f"file not found in the current directory: {args.filename}")

    from . import directories

    directories.set_output_directory(Path.cwd())

    from .toughitup import GeoModel

    GeoModel(args.filename, invert_y_axis=args.i, format=args.f).convert_model()