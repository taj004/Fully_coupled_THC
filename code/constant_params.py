"""
Various parameters that are constant in the calculations
"""

import numpy as np
import scipy.sparse as sps
import porepy as pp

# ------------------------------- #

# Matrices that are constants

def all_2_aquatic_mat(aq_components: np.ndarray, 
                      num_components: int,
                      num_aq_components: int,
                      gb_size: int):
    """ 
    Create a mapping between the aqueous and fixed species
    
    Input: 
        aq_componets: np.array of aqueous componets
        num_components: int, number of components
        num_aq : int, number of aqueous components
        gb_size: number of cells, faces or mortar cells in the pp.GridBucket
    """
    
    num_aq_components = aq_components.size
    cols = np.ravel(
        aq_components.reshape((-1, 1)) + (num_components * np.arange(gb_size)), order="F"
    )
    sz = num_aq_components * gb_size
    rows = np.arange(sz)
    matrix_vals = np.ones(sz)
    
    # Mapping from all to aquatic components. The reverse map is achieved by a transpose.
    all_2_aquatic = pp.ad.Matrix(
        sps.coo_matrix(
            (matrix_vals, (rows, cols)),
            shape=(num_aq_components * gb_size, num_components * gb_size),
        ).tocsr()
    )
    
    return all_2_aquatic

# def extension_mat(n,m):
#     "Extend an np.ndarray of size n to size m, with m>=n"
#     assert m>=n
#     s = sps.bmat([[sps.eye(n,n)],
#                   [sps.eye(m-n,n)]]) 
#     return s

def enlarge_mat_2(n,m):
    """
    Construct a matrix s to expand an vector v of size n, m times
    The function can be though of an variant of np.tile
    but applicable to, say, pp.Ad_arrays
    """
        
    k = m/n
    
    # Decide how the extension should be carried out 
    # (They might differ depending the the application)
    if k%1 == 0:
        k=int(k)
        e=np.ones((k,1))
        s = sps.lil_matrix((m,n)) 
        #breakpoint()
        for i in range(n):
            s[i*k:(i+1)*k,i]=e
        # end i-loop
    else:
        s = sps.bmat([[sps.eye(n,n)],
                      [sps.eye(m-n,n)]]) 
    # end if-else
    
    s = s.tocsr()
    return s

def sum_mat(n,m, s=2):
    """
    Create a matrix which upon multilpied with a vector, 
    sums its s entries
    """
    zz = []
    for i in range(n):
        x = np.zeros(m)
        x[s*i:s*(i+1)] = 1
        zz.append(x)
    # end i-loop
    
    S = sps.csr_matrix(zz)
    return S

def dimensional_mat(dim, gb):
    """
    An indicator matrix, based on dimension 
    """    
    
    if dim > 3 or dim < 0:
        raise ValueError("Not implemented for input dimension")
    
    # Adjust values according to dimension
    val = 0
    
    if dim < gb.dim_max():
        for g,_ in gb:
            if dim < g.dim:
                val += g.num_cells
            # end if
        # end g-loop
    # end if
    
    grid = gb.grids_of_dimension(dim)    
   
    x = np.zeros(gb.num_cells())
    
    for i in range(len(grid)):
        g = grid[i]
        indi = slice(val,val+g.num_cells)
        x[indi] = 1.0
        val += g.num_cells
    # en i-loop
    
    S = sps.diags(x)
    return S

# ------------------------------- #
"""
Constant and reference values
"""

    
def ref_p():
    "Reference pressure, [Pa]"
    return 1000.

def molar_mass_CaCO3():
    "Kg/mol"
    return 100.09 * 0.001

def molar_mass_CaSO4():
    "[kg/mol]"
    return  136.15 * 0.001

def density_CaCO3():
    "[Kg/m^3]"
    return 2.71e3
  
def density_CaSO4():
    "[Kg/m^3]"
    return 2.97e3

def dynamic_viscosity():
    "[Pa s]"
    return 1e-3

def open_aperture():
    "[m]"
    return 5e-3

def ref_temp():
    "[ĸ]"
    return 573.15

def length_scale():
    return 1

def scale_const():
    """
    Scalar constant, used to aid the convergence of Newton
    for the isothermal problem
    """
    return 1e8

def temp_scale():
    "[K]"
    ts = 1
    return ts

def fluid_conduction():
    "[W/(m K)]"
    return 0.6

def solid_conduction():
    "[W/(m K)]"
    return 3.0

def specific_heat_capacity_fluid():
    "[J/K]"
    return 4200

def specific_heat_capacity_solid():
    "[J/K]"
    return 790


# ------------------------------- #

def standard_enthalpy():
    """
    The standard enthalpy values for the reactions
    """
    R = 8.314 # Gas constant
    
    # Standard enthaply values
    hco3 =-691.1
    h = 0
    co3 = -676.3
    hso4 = -885.75
    so4 = -907.5
    oh = -229.94
    h2o = -285.8
    caco3 = -1206.9
    ca = -542.96
    caso4 = -1432.69
    
    # The enthalpy energy for the reactions
    delta_H = np.array([
        h   + co3 - hco3,
        h   + so4 - hso4,
        h2o - h   - oh  , 
        ca  + co3 - caco3,
        ca  + so4 - caso4 
        ]) * 1e3  # The above values have unit [kJ/mol] 
    
    return delta_H / R

def ref_equil():
    """ 
    Reference euilibrium constants, from Plummer et al.
    Take exponential, as they are given on log-scale
    Moreover, we consider the reciprocal for some of the equilibrium constants,
    as we look at reactions
    A_j <-> \sum_i xi_{ij} A_i, and
    A_j = eq_const * prod_i A_{i} ^xi_{ij}
    where A_j are the secondary species, and A_i are components (primary variables).
    To be consistenst with the definition of the equilibirum constant, which is
    eq = prod_i A_i ^xi_{ij} / A_j,
    we thus need to take reciprocal.
    """
    equil_consts_plummer = 1 / np.exp(np.array([10.339, 1.979, -13.997, -8.406, -4.362])) 
    equil_consts_plummer[3:5] = 1 / equil_consts_plummer[3:5]
    return equil_consts_plummer

    