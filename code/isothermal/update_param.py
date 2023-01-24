#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Update various parameters for the computations

@author: uw
"""

import numpy as np
import porepy as pp 
import scipy.sparse as sps

import sys
sys.path.insert(0,"..")
import constant_params

def update_perm(gb):
    """ Update the permeability in the matrix and fracture
    
    """
    
    for g,d in gb:     
        
        specific_volume = specific_vol(gb, g)        
        ref_perm =  d[pp.PARAMETERS]["reference"]["permeability"]
        K=d[pp.STATE][pp.ITERATE]["permeability"]
        kk =  K * specific_volume / constant_params.dynamic_viscosity()
        d[pp.PARAMETERS]["flow"].update({
            "permeability": kk,
            "second_order_tensor": pp.SecondOrderTensor(kk)
            }) 
        
        # We are also interested in the current permeability,
        # compared to the initial one
        d[pp.STATE].update({"ratio_perm": K/ref_perm})
        
    # end g,d-loop    
    
def update_mass_weight(gb):
    """ Update the mass weights in the solute transport equation
    The porosity is updated as "porosity = 1 - sum_m x_m",
    where x_m is the mth volume mineral fraction 
    
    """
    
    for g,d in gb:
        
        specific_volume = specific_vol(gb, g)
        
        porosity = d[pp.STATE][pp.ITERATE]["porosity"]
        d[pp.PARAMETERS]["mass"].update({
            "porosity": porosity,
            "mass_weight": porosity.copy() * specific_volume.copy()
            })
         
        d[pp.PARAMETERS]["flow"].update({
            "mass_weight": porosity.copy() * specific_volume.copy()
            })
        
        d[pp.PARAMETERS]["passive_tracer"].update({
            "mass_weight": porosity.copy() * specific_volume.copy()
            })
    # end g,d-loop   


def update_interface(gb):
    """ Update the interfacial permeability
    
    """

    for e,d in gb.edges():
        mg = d["mortar_grid"]
        gl, gh = gb.nodes_of_edge(e)
            
        data_l = gb.node_props(gl)
        aperture = data_l[pp.PARAMETERS]["mass"]["aperture"]
        
        Vl = specific_vol(gb, gl)
        Vh = specific_vol(gb, gh) 
        
        # Assume the normal and tangential permeability are equal
        # in the fracture
        ks = data_l[pp.PARAMETERS]["flow"]["permeability"]
        tr = np.abs(gh.cell_faces)
        Vj = mg.primary_to_mortar_int() * tr * Vh 
        
        # The normal diffusivity
        nd = mg.secondary_to_mortar_int() * np.divide(ks, aperture * Vl / 2) * Vj
        
        d[pp.PARAMETERS]["flow"].update({"normal_diffusivity": nd })    
        
    # end e,d-loop


def specific_vol(gb,g):
    return gb.node_props(g)[pp.PARAMETERS]["mass"]["specific_volume"]

    
def update_intersection_aperture(gb, g):
    """
    calulcate aperture intersection points (i.e. 0-d)
    gb : pp.GRID Bucket
    g : zero-dimension grid
    """
    
    if g.dim > 0:
        raise ValueError("The grid must be zero-dimensional")
    # end if
    
    parent_aperture = []
    num_parent = []
        
    # if g.dim < gb.dim_max()-1:
    for edges in gb.edges_of_node(g):
        e = edges[0]
        gh = e[0]
        
        if gh == g:
            gh = e[1]
        # end if
        
        if gh.dim == gb.dim_max()-1:
            dh = gb.node_props(gh)
            a = dh[pp.STATE][pp.ITERATE]["aperture"] # dh[pp.PARAMETERS]["mass"]["aperture"]
            #ah = np.abs(gh.cell_faces) * a
            mg = gb.edge_props(e)["mortar_grid"]
            
            # Projection operators
            projection = (
                mg.mortar_to_secondary_avg() * 
                mg.primary_to_mortar_avg() * 
                np.abs(gh.cell_faces)
                )
            #breakpoint()
            al = projection * a
   
            parent_aperture.append(al)
            num_parent.append(
                np.sum(mg.mortar_to_secondary_int().A,axis=1)
                )
        # end if
    # end edge-loop
    
    parent_aperture = np.array(parent_aperture)
    num_parent = np.sum(np.array(num_parent), axis=0)
        
    aperture = np.sum(parent_aperture, axis=0) / num_parent 
    
    return aperture
            

def rho(p):
    """
    Constitutive law between density and pressure
    """    
 
    # reference density 
    rho_f = 1e3 
    
    # Pressure values
    c = 1.0e-9 # compresibility [1/Pa]
    p_ref = constant_params.ref_p() # reference pressure [Pa]
    
    
    if isinstance(p, pp.ad.Ad_array): # The input pressue is a np.array
        density = rho_f * pp.ad.exp(c * (p - p_ref)) 
    else: 
        density = rho_f * np.exp(c * (p - p_ref)) 
    # end if-else
    return density

def aperture_state_to_param(gb):
    
    for g,d in gb:
        d_mass = d[pp.PARAMETERS]["mass"]
        aperture = np.clip(
            d[pp.STATE][pp.ITERATE]["aperture"].copy(), 1e-7, constant_params.open_aperture()
            )
        d_mass.update({"aperture": aperture.copy(),
                       "specific_volume": np.power(aperture, gb.dim_max()-g.dim)
                       })
        
        d[pp.STATE].update({
            "aperture_difference": constant_params.open_aperture()-aperture.copy() 
            })
    # end g,d-loop
    
def update(gb):
    """
    Update the aperture, porosity and permeability
    """
    
    aperture_state_to_param(gb)
        
    # Update porosity
    update_mass_weight(gb)
    
    # Update permeability
    update_perm(gb)
    
    # Update the interface properties
    update_interface(gb)
    
    
def update_concentrations(gb, dof_manager, to_iterate=False):
    """
    Update concenctations
    """
    dof_ind = np.cumsum(np.hstack((0, dof_manager.full_dof)))
    x = dof_manager.assemble_variable(from_iterate=True)
    
    for g,d in gb:
        
        if g.dim == gb.dim_max():
            data_chemistry = d[pp.PARAMETERS]["chemistry"]
        # end if

        for key, val in dof_manager.block_dof.items():
            
            if isinstance(key[0], tuple) is False: # Otherwise, we are at the interface
            
                # The primary species corresponds to the exponential of log_conc,
                # which is stored in the dictionary as "log_X"
                # Moreover, we consider dimension by dimension,
                # thus we also need the "current" dimension of the grid.
                # Lastly, we look at the grids inividually
                if g.dim == key[0].dim  and g.num_cells == key[0].num_cells:
                    if key[1] == "log_X" :
                        inds = slice(dof_ind[val], dof_ind[val+1]) 
                        primary = x[inds]
                
                    # Get the mineral concentrations, in a similar manner
                    # (they are needed for computing the change in porosity)
                    elif key[1] == "minerals":
                        inds = slice(dof_ind[val], dof_ind[val+1])
                        #print(inds)
                        mineral = x[inds]
                        #print(mineral)
                        break # When we reach this point, we have all the necessary 
                              # information return the concentrations, for the 
                              # particular grid. Hence, we may jump out of the loop 
                # end if-else
          # end key,val-loop
        
        if not to_iterate:
            S = data_chemistry["stoic_coeff_S"] 
            eq = data_chemistry["equilibrium_constants_comp"] 
            eq_const_on_grid = sps.dia_matrix(
                    (np.hstack([eq for i in range(g.num_cells)]), 0),
                    shape=(
                        S.shape[0] * g.num_cells,
                        S.shape[0] * g.num_cells,
                        ),
                    ).tocsr() 
            S_on_grid = sps.block_diag([S for i in range(g.num_cells)]).tocsr()        
            secondary_species = eq_const_on_grid * np.exp(S_on_grid * primary) 
        # end if
        
        caco3 = mineral[0::2]
        caso4 = mineral[1::2]
        
        if to_iterate:
            d[pp.STATE][pp.ITERATE].update({
                "CaCO3": caco3,
                "CaSO4": caso4
                })
        else:
  
            d[pp.STATE].update({
            "CaCO3": caco3,
            "CaSO4": caso4,
            
            # Primary
            "Ca2+": np.exp(primary[0::4]),
            "CO3":  np.exp(primary[1::4]),
            "SO4":  np.exp(primary[2::4]),
            "H+":   np.exp(primary[3::4]),
            
            # Secondary
            "HCO3": secondary_species[0::3],
            "HSO4": secondary_species[1::3],
            "OH-" : secondary_species[2::3],   
            })
        
            d[pp.PARAMETERS]["concentration"].update({ 
            # Primary
            "Ca2+": np.exp(primary[0::4]),
            "CO3":  np.exp(primary[1::4]),
            "SO4":  np.exp(primary[2::4]),
            "H+":   np.exp(primary[3::4]),
            
            # Secondary
            "HCO3": secondary_species[0::3],
            "HSO4": secondary_species[1::3],
            "OH-" : secondary_species[2::3],   
            
            # Minerals
            "CaCO3": caco3,
            "CaSO4": caso4
                })              
        # end if
          
    # end g,d-loop