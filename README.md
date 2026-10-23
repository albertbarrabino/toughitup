# TOUGHITUP

TOUGHITUP converts corner-point grids from GRDECL files into VTU or VTK files. It imports the grid geometry and the following supported cell properties:

- PERMX, PERMY, and PERMZ
- PORO
- ACTNUM, FIPNNUM, SATNUM, and MULTNUM
- NTG and ZONE

The `COPY` and `MULTIPLY` keywords are also supported.

## Run TOUGHITUP

Run TOUGHITUP from the directory containing your GRDECL file. Provide the filename only, not a path:

```text
toughitup model.grdecl -i False -f vtu
```

### Command-line options

Both flags are optional, and their values are case-insensitive.

| Flag | Accepted values | Default | Description |
| ---- | --------------- | ------- | ----------- |
| `-i` | `True`, `False` | `False` | Negates the Y coordinates, mirroring the model across the XZ plane. |
| `-f` | `vtu`, `vtk`, `None` | `None` | Selects the output format. `None` saves both formats. |

### Examples

Save only VTU:

```text
toughitup model.grdecl -f vtu
```

Save only VTK and invert the Y axis:

```text
toughitup model.grdecl -i True -f VTK
```

Save both formats:

```text
toughitup model.grdecl
```

Output files are saved as `grid.vtu` and/or `grid.vtk` in the current directory. Existing output files with the same names are overwritten.
