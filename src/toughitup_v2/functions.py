import numpy as np


# =============== Conversion Factors =============== 

ft_to_m = 0.3048
md_to_sqm = 9.869233E-16

keywords_accepted = [
    "PERMX", "PERMY", "PERMZ", "PORO", "ACTNUM", "FIPNNUM",
    "SATNUM", "MULTNUM", "NTG", "ZONE", 
    ]


# ================== Cell builder ==================
  
def get_ref_sys(nx, ny, nz):
        
    i, j, k = np.meshgrid(
        np.arange(1, nx + 1),
        np.arange(1, ny + 1),
        np.arange(1, nz + 1),
        indexing = 'ij'
    )

    ref_sys = np.stack([i, j, k], axis=-1).reshape(-1, 3, order='F')
    return ref_sys


def format_coord(coord_data, nx, ny, ecl_indexes):
        
    coord = np.array(coord_data).reshape(ny + 1, nx + 1, 6)
    pillars = []

    for cell in ecl_indexes:
        ind_i_a = cell[0]-1
        ind_i_b = cell[0]
        ind_j_a = cell[1]-1
        ind_j_b = cell[1]
        p1 = coord[ind_j_a, ind_i_a]
        p2 = coord[ind_j_a, ind_i_b]
        p3 = coord[ind_j_b, ind_i_b]
        p4 = coord[ind_j_b, ind_i_a]
        pillars_cell = [p1, p2, p3, p4]
        pillars.append(pillars_cell)
        
    return np.array(pillars)


def format_zcorn(zcorn_data, nx, ny, ecl_indexes):
    zcorn = np.array(zcorn_data).reshape(-1, 2*ny, 2*nx)
    corners = []

    for cell in ecl_indexes:
        i0, j0, k0 = (2 * c - 2 for c in cell)
        i1, j1, k1 = (i0 + 1, j0 + 1, k0 + 1)

        # All corner index combinations
        # Notice that I am alterating the vertices order to match VTK's hexahedron definition!!
        corner_indices = [
            (k0, j0, i0), (k0, j0, i1),
            (k0, j1, i1), (k0, j1, i0),
            (k1, j0, i0), (k1, j0, i1),
            (k1, j1, i1), (k1, j1, i0)
        ]

        points = [zcorn[k, j, i] for k, j, i in corner_indices]
        corners.append(points)

    return np.array(corners)


def get_info(pillar, corner):
    """Interpolate corner XY positions using COORD and ZCORN in the same units."""
    pillar_indices = [0, 1, 2, 3, 0, 1, 2, 3]
    endpoints = pillar[pillar_indices]
    top = endpoints[:, :3]
    bottom = endpoints[:, 3:]
    delta_xy = bottom[:, :2] - top[:, :2]
    delta_z = bottom[:, 2] - top[:, 2]
    vertical = np.all(delta_xy == 0, axis=1)
    if np.any((delta_z == 0) & ~vertical):
        raise ValueError(
            "Cannot interpolate a nonvertical COORD pillar with equal endpoint depths."
        )

    z = corner.reshape(-1, 1)
    fraction = np.zeros(len(endpoints), dtype=float)
    np.divide(
        corner - top[:, 2],
        delta_z,
        out=fraction,
        where=~vertical,
    )
    xy = top[:, :2] + fraction[:, None] * delta_xy
    return np.hstack((xy, z))

    
def build_cells(zcorn_data, coord_data, nx, ny, ecl_indexes, n_cells, units):
    
    corners = format_zcorn(zcorn_data, nx, ny, ecl_indexes)
    pillars = format_coord(coord_data, nx, ny, ecl_indexes)
        
    if units == 'FEET':
        pillars *= ft_to_m
        corners *= ft_to_m
        print(" -> UNITS: Converted from feet to meters.")

    cells = []
    for i in range(n_cells):
        single_cell = get_info(pillars[i], corners[i])
        cells.append(single_cell)
        
    return np.array(cells).reshape(n_cells, 8, 3)