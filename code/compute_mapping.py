import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sps 

import porepy as pp

# Common names
non_isothremal_folder_path = "non_isothermal/to_study/2D/"
isothermal_folder_path = "isothermal/to_study/"
folder_name = isothermal_folder_path # non_isothremal_folder_path
gb_refinement_name = folder_name + "gb_refinement_level_" 

# Read in gb
def read_pickle(path):
    """Read a stored object"""
    with open(path, "rb") as f:
        raw = f.read()
        print(raw)
    return pickle.loads(raw)

# Store the gb
def write_pickle(obj, path):
    """
    Store the current grid bucket for the convergence study
    
    """
    path = Path(path)
    raw = pickle.dumps(obj)
    with open(path, "wb") as f:
        raw = f.write(raw)        
    return

max_refinement_level = 5
gb_list = []

# Read in the grid buckets
for i in range(max_refinement_level+1):
    refinement_level = str(i)
    read_gb_list = read_pickle(gb_refinement_name + refinement_level)
    gb_list.append(read_gb_list[0])
# end i-loop

ref_gb = gb_list[-1]

def grid_mapping(g, ref_g):
    """ Compute a projection matrix between a grid g and a reference grid, 
    so that s = proj * q. Here, q is a solution on g and s is q mapped onto 
    reference grid.
    
    This is for grids of dimension 0,1,2 and the refinement 
    is done in a unstructured way
    
    """
    if g.dim == 2:
        proj = pp.match_grids.match_2d(ref_g, g, tol=1e-5, scaling="averaged")
    elif g.dim == 1:
        proj = pp.match_grids.match_1d(ref_g, g, tol=1e-5, scaling="averaged")
    elif g.dim == 0:
        proj = sps.csc_matrix(np.ones(1))
    else:
        raise ValueError(f"Cannot compute projection matrix for a {g.dim}-dimensional grid")
    
    return proj 

def gb_ref(gb, ref_gb):
    """ Compute a projection matrix between the grids in a grid bucket and the
    reference grids in the reference grid bucket.
    
    """
    ref_g_list = ref_gb.get_grids()
    g_list = gb.get_grids()
    keys ="mapping_from_coarse_cells_to_fine_cells"
    
    len_g = len(g_list)
    assert len(ref_g_list) == len_g
    
    for i in range(len_g):
        grid, ref_grid = g_list[i], ref_g_list[i]   
        proj = grid_mapping(grid, ref_grid)
        d=gb.node_props(grid)
        d[keys] = proj
    # end i-loop
    
# Compute the mapping
for i in range(len(gb_list)-1):
    print(i)
    gb_ref(gb_list[i], ref_gb)
    write_pickle([gb_list[i]], gb_refinement_name + str(i) + "_with_mapping")
# end loop


