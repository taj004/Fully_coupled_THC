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

def equil_state_to_param(gb):
    
    
    # Splitt the equilibrium constants for components 
    # and precipitation species
    x1 = np.zeros(gb.num_cells() * 3, dtype=int)
    x2 = np.zeros(gb.num_cells() * 2, dtype=int)
    for i in range(gb.num_cells()):
        j = i * 5
        inds1 = np.array([j, j+1, j+2]) 
        inds2 = np.array([j+3, j+4])
        
        x1[3*i:3*i+3] = inds1
        x2[2*i:2*i+2] = inds2
    # end i-loop
    
    equil=np.zeros(5*gb.num_cells())
    cell_val =0
    for g,d in gb:
        if g.dim==gb.dim_max():
            data=d
        # end if
        inds=slice(cell_val, cell_val+5*g.num_cells)
        equil[inds] = d[pp.STATE][pp.ITERATE]["equilibrium_constants"]
        cell_val += 5*g.num_cells
    # end loop
    #breakpoint()
    data["parameters"]["chemistry"].update({
        "cell_equilibrium_constants_comp": sps.diags(equil[x1]),
        "cell_equilibrium_constants_prec": sps.diags(equil[x2])
        })

def taylor_app_equil(temp):
    """
    Taylor approximation of the equililibrium constants
    """
    delta_H=constant_params.standard_enthalpy()
    dt = np.clip(temp-constant_params.ref_temp(), 
                 a_min=-constant_params.ref_temp()/6, a_max=constant_params.ref_temp()/6
                 )
    taylor = (
        1 + delta_H * dt / constant_params.ref_temp()**2 
            )
        
    return taylor

def update_perm(gb):
    """
    Update the permeability in the matrix and fracture
    """
    
    for g,d in gb:     
        
        specific_volume = specific_vol(gb, g)
        
        ref_perm =  d[pp.PARAMETERS]["reference"]["permeability"]
        K=d[pp.STATE][pp.ITERATE]["permeability"]
        
        kk = K * specific_volume / constant_params.dynamic_viscosity()
        
        d[pp.PARAMETERS]["flow"].update({
            "permeability": kk,
            "second_order_tensor":pp.SecondOrderTensor(kk)
            }) 
        
        # We are also interested in the current permeability,
        # compared to the initial one
        d[pp.STATE].update({"ratio_perm": K/ref_perm})
        
    # end g,d-loop    
    
def update_mass_weight(gb):
    """
    Update the porosity dependencies in the transport equations:
    For the soltue transport this is the porosity, while for
    temperature it is the heat capacities and conduction
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
        
        
        # Next, iterate on the heat effects
        d_temp = d[pp.PARAMETERS]["temperature"]
        specific_heat_capacity_fluid = constant_params.specific_heat_capacity_fluid() 
        specific_heat_capacity_solid = constant_params.specific_heat_capacity_solid()  
     
        solid_density = d_temp["solid_density"]         
        fluid_density = rho(d[pp.STATE][pp.ITERATE]["pressure"],
                            d[pp.STATE][pp.ITERATE]["temperature"])
        
        heat_capacity = (
            porosity.copy() * fluid_density * specific_heat_capacity_fluid + # fluid part
            (1-porosity.copy()) * solid_density * specific_heat_capacity_solid  # solid poart
            )
        conduction = (
            porosity.copy() * constant_params.fluid_conduction() + 
            (1-porosity.copy()) * constant_params.solid_conduction() 
            )

        d_temp.update({
            "mass_weight": heat_capacity * specific_volume.copy(), 
            "second_order_tensor": pp.SecondOrderTensor(
                conduction * specific_volume.copy()
                ) 
            })
    # end g,d-loop   


def update_interface(gb):
    """
    Update the elliptic interfaces
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
        q = 2 * constant_params.fluid_conduction() * Vj / (mg.secondary_to_mortar_int() * aperture)
        #breakpoint()
        d[pp.PARAMETERS]["flow"].update({"normal_diffusivity": nd})    
        d[pp.PARAMETERS]["temperature"].update({"normal_diffusivity": q})
        
    # end e,d-loop

def specific_vol(gb,g):
    return gb.node_props(g)[pp.PARAMETERS]["mass"]["specific_volume"]


def update_intersection_aperture(gb, g):
    """
    calulcate aperture intersection points (i.e. 0-d)
    gb : pp.GRID Bucket
    g : zero-dimension grid
    """
    
    if g.dim > gb.dim_max()-2:
        raise ValueError("The grid must be an intersection domain")
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
            a = dh[pp.PARAMETERS]["mass"]["aperture"]
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

def rho(p, temp):
    """
    Constitutive law between density and pressure
    """    
 
    # reference density 
    rho_f = 1e3 
    
    # Pressure values
    c = 1.0e-9 # compresibility [1/Pa]
    p_ref = constant_params.ref_p() # reference pressure [Pa]
    
    # isinstance(p,pp.ad.Ad_array) will return True here,
    # even though the "input" p is an pp.ad.opertor.merged_variable.
    # The reason is the pp.merged_variable is a wrapper of p, but once we 
    # want to evaluate it (like we do here), p becomes an pp.Ad_array
    
    beta = 4e-4 
    temp_ref = constant_params.ref_temp()
    dt = np.clip(temp-temp_ref, a_min=-temp_ref/6, a_max=temp_ref/6)
    
    # the density
    density = rho_f * (1 + c*(p-p_ref) - 0*beta*dt)     
    
    # Scale according to the presence of pressure/temperature
    # Most likely not needed
    if isinstance(temp, np.ndarray): 
        density *= (temp > 0).astype(float)
    
    return density

def aperture_state_to_param(gb):
    
    for g,d in gb:

        d_mass = d[pp.PARAMETERS]["mass"]
        
        # Clip to avoid unphysical aperture; however it in the paper simulations
        # this is not really needed
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
    
    
    return gb
    
def update_concentrations(gb, dof_manager, to_iterate=False):
    """
    Update concenctations
    """
    dof_ind = np.cumsum(np.hstack((0, dof_manager.full_dof)))
    x = dof_manager.assemble_variable(from_iterate=True)
    cell_val = 0
    for g,d in gb:
        
        if g.dim == gb.dim_max():
            data_chemistry = d[pp.PARAMETERS]["chemistry"]  
            eq_diagonal = data_chemistry["cell_equilibrium_constants_comp"].diagonal() 
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
                        mineral = x[inds]
                        break # When we reach this point, we have all the necessary 
                              # information return the concentrations, for the 
                              # particular grid. Hence, we may jump out of the loop 
                # end if-else
          # end key,val-loop
        
        if not to_iterate:
            S = data_chemistry["stoic_coeff_S"] 
            eq_inds=slice(cell_val, cell_val+ g.num_cells*S.shape[0])
            S_on_grid = sps.block_diag([S for i in range(g.num_cells)]).tocsr()        
            secondary_species = sps.diags(eq_diagonal[eq_inds]) * np.exp(S_on_grid * primary) 
            cell_val += g.num_cells * S.shape[0]
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
               
        # end if
          
    # end g,d-loop
