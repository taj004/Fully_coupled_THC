import numpy as np
import porepy as pp


def create_gb(mesh_args: dict, dim: int = 2, fractured: bool = False):
    """
    Create the grid for the computations:
       - An unstructured 2D grid, without fractures, or with five fractures
       - A 3D grid, either without fractures or with two that intersect at a line

    Parameters:
        mesh_args (dict): contains mesh arguments for the (unstrucured) grid
        dim: int: highest dimension of grid
        fractured (boolean, optional): Whether fractures are present.
            Default is False

    """

    if not mesh_args:
        raise ValueError("Mesh parameters must be given")

    if dim == 2:

        if fractured:
            pts = np.array(
                [
                    [0.6, 0.2],  # End pts
                    [0.2, 0.8],  # Statring pts
                    [0.6, 0.6],
                    [0.2, 0.5],
                    [1.2, 0.6],
                    [0.9, 0.8],
                    [1.7, 0.3],
                    [1.0, 0.2],
                ]
            ).T

            e = np.array([[0, 1], [2, 3], [4, 5], [6, 7]]).T

        else:
            pts, e = None, None
        # end if-else

        domain = {"xmin": 0.0, "xmax": 2, "ymin": 0.0, "ymax": 1}

        network_2d = pp.FractureNetwork2d(pts, e, domain)
        gb = network_2d.mesh(mesh_args)
    # end if-else

    elif dim == 3:

        domain = {"xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "zmin": 0, "zmax": 1}

        if fractured:
            f1 = pp.Fracture(
                np.array(
                    [
                        [0.25, 0.75, 0.25, 0.75],
                        [0.30, 0.30, 0.70, 0.70],
                        [0.30, 0.30, 0.70, 0.70],
                    ]
                )
            )
            f2 = pp.Fracture(
                np.array(
                    [
                        [0.25, 0.75, 0.25, 0.75],
                        [0.30, 0.30, 0.70, 0.70],
                        [0.70, 0.70, 0.30, 0.30],
                    ]
                )
            )
            frac_list = [f1, f2]
        else:
            frac_list = []
        # end if-else
        network_3d = pp.FractureNetwork3d(frac_list, domain=domain)
        gb = network_3d.mesh(mesh_args)
    else:
        raise ValueError("Only 2- or 3D grids")
    # end if-else

    return gb
