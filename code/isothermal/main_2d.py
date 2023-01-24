"""
Main script
@author: uw
"""

# %% Import the neseccary packages
import pickle
from pathlib import Path

import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
import porepy as pp

import update_param
import equations
from solve_non_linear import solve_eqs

import sys
sys.path.insert(0, "..")
from create_mesh import create_gb
import constant_params
 
# %% Initialize variables related to the chemistry and the domain

# Start with the chemical related variables
# i.e. equilibrium constants, stoichiometric coeffients, number of species

# Equilibrium constants. 
equil_consts = constant_params.ref_equil()

# Stochiometric matrix for the reactions between primary and secondary species.
# Between aqueous species
S = sps.csr_matrix(
    np.array(
        [
            [0, 1, 0,  1],
            [0, 0, 1,  1],
            [0, 0, 0, -1],
        ]
    )
)

# Between aqueous and preciptated species
E = sps.csr_matrix(
    np.array(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0]
        ]
    )
)

# The number of primary and secondary species can be inferred from the number of
# columns and rows, respectively, of S_W.
# The indices of which primary species are aqueous are specified below.
# Note that the aqueous secondary species should have no connection with fixed
# primary species (the column of S should be all zeros).

# Index of aquatic and fixed components, referring to the cell-wise T vector
aq_components = np.array([0, 1, 2, 3])
fixed_components = 0
num_aq_components = aq_components.size
num_fixed_components = 0
num_components = num_aq_components + num_fixed_components
num_secondary_components = S.shape[0] 

# Define fractures and a gb
ref_level = 0 # The refinement levels are 0, 1, 2, 3, 4; The finest grid uses dx=0.007
size = 0.2 / np.power(2,ref_level)

mesh_args={"mesh_size_frac" : size,
           "mesh_size_min"  : size,
           "mesh_size_bound": size}
gb = create_gb(mesh_args=mesh_args, grid_type="unstructured", fractured=True)

domain = {"xmin": 0, "xmax": gb.bounding_box()[1][0], 
          "ymin": 0, "ymax": gb.bounding_box()[1][1]} 


# Keywords
mass_kw = "mass"
chemistry_kw = "chemistry"
transport_kw = "transport"
flow_kw = "flow"
tracer_kw = "passive_tracer"

pressure = "pressure"
tot_var = "T"
log_var = "log_X"

minerals = "minerals"
min_1 = "CaCO3"
min_2 = "CaSO4"

mortar_pressure = "mortar_pressure"
mortar_transport = "mortar_transport"
mortar_tracer = "mortar_tracer"

# %% Loop over the gb, and set initial and default data

# Permeabilities
matrix_permeability = 1e-11

# Initial pressure
init_pressure = constant_params.ref_p() # [Pa]

for g, d in gb:
        
    gb.add_node_props(["is_tangential"])
    d["is_tangential"] = True

    # Initialize the primary variable dictionaries
    # For some reason, equation.py had some unknown (atleast to me) issues with keywords.
    # My solution was to initial the keyword primary variables as below
    d[pp.PRIMARY_VARIABLES] = {pressure:  {"cells": 1},
                               tot_var:   {"cells": num_components},
                               log_var:   {"cells": num_components},
                               minerals:  {"cells": 2},
                               tracer_kw: {"cells": 1},
                               "porosity": {"cells": 1},
                               "aperture": {"cells": 1},
                               "permeability": {"cells":1}
                               }

    # Initialize a state
    pp.set_state(d)

    unity = np.ones(g.num_cells)
    
    # --------------------------------- #
    
    # Set inital concentration values
    
    if gb.dim_max() == g.dim:
        
        so4 = 10 * unity
        ca = 0.9 / (equil_consts[4] * so4) * unity
        co3 = 1 / (equil_consts[3] * ca) * unity
        
        oh = 1.5e3* unity
        h = equil_consts[2] / oh * unity
        hco3 = equil_consts[0] * h * co3 * unity
        hso4 = equil_consts[1] * h * so4 * unity
        
        precipitated_init = np.zeros((2,g.num_cells))
        precipitated_init[0] = 20.0
        
        init_X = np.array([ca, co3, so4, h])
        init_X_log = np.log(init_X)
        
        alpha_init = equil_consts[0:3].reshape(([3,1])) * np.exp(S * init_X_log)
        
        init_T = init_X + S.T * alpha_init + E.T * precipitated_init
        
        # Reshape for the calulations
        init_T = init_T.reshape((init_T.size), order="F")
        
    else: 
        
        # Inherit values
        
        so4 = so4[0] * unity
        ca = ca[0] * unity
        co3 = co3[0] * unity
        
        oh = oh[0] * unity
        h = h[0] * unity
        hco3 = hco3[0] * unity
        hso4 = hso4[0] * unity
        
        precipitated_init = np.zeros((2, g.num_cells))
        precipitated_init[0] = 20.0
    
        init_X = np.array([ca, co3, so4, h])
        init_X_log = np.log(init_X)
        
        alpha_init = equil_consts[0:3].reshape(([3,1])) * np.exp(S*init_X_log)
        
        init_T = init_X + S.T * alpha_init + E.T * precipitated_init
        
        init_T = init_T.reshape((init_T.size), order="F")
    # end if-else
    
    # --------------------------------- #
    
    # Initial porosity, permeablity and aperture.
 
    # # Number of moles
    mol_CaCO3 = precipitated_init[0] * g.cell_volumes 
    mol_CaSO4 = precipitated_init[1] * g.cell_volumes
    
    # Next convert mol to kg
    mass_CaCO3 = mol_CaCO3 * constant_params.molar_mass_CaCO3()
    mass_CaSO4 = mol_CaSO4 * constant_params.molar_mass_CaSO4()

    # the mineral volumes
    mineral_vol_CaCO3 = mass_CaCO3 / constant_params.density_CaCO3() 
    mineral_vol_CaSO4 = mass_CaSO4 / constant_params.density_CaSO4() 
    
    # Reactive surface area.  
    S_0 = np.power(g.cell_volumes , 2/3)

    mineral_width_CaCO3 = mineral_vol_CaCO3 / S_0 
    mineral_width_CaSO4 = mineral_vol_CaSO4 / S_0 
    
    if g.dim == gb.dim_max():
        aperture = unity
    elif g.dim == 1:
        aperture = constant_params.open_aperture() - (
            mineral_width_CaCO3 + mineral_width_CaSO4
            ) 
    else: 
        aperture = aperture[0]*unity  #update_param.update_intersection_aperture(gb, g)
    # end if-else

    specific_volume = np.power(aperture, gb.dim_max()-g.dim)
    
    porosity = (
        1 - (mineral_vol_CaCO3 + mineral_vol_CaSO4) / g.cell_volumes  
        ) 
    
    non_reactive_volume_frac = 0.8 if g.dim == gb.dim_max() else 0
    
    if g.dim == gb.dim_max():
        porosity -= non_reactive_volume_frac        
        K = matrix_permeability * unity
    else:
        K = np.power(aperture, 2) / 12 
    # end if-else
        
    kk = K * specific_volume.copy() / constant_params.dynamic_viscosity()
    # --------------------------------- #

    # Boundary conditions
    bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]

    if bound_faces.size != 0: # Non-zero if a grid is "attached" to the global boundary, zero otherwise
        
        bound_faces_centers = g.face_centers[:, bound_faces]
        
        # Neumann faces
        neu_faces = np.array(["neu"] * bound_faces.size) 
        
        # Define the inflow and outflow in the domain
        inflow = bound_faces_centers[0] < domain["xmin"] + 1e-4
        outflow = bound_faces_centers[0] > domain["xmax"] - 1e-4

        # The BC labels for the flow problem
        labels_for_flow = neu_faces.copy()
        labels_for_flow[inflow] = "dir" 
        labels_for_flow[outflow] = "dir"
        
        # Set the BC values for flow
        bc_values_for_flow = np.zeros(g.num_faces)
        bc_values_for_flow[bound_faces[inflow]] = 7 * init_pressure
        bc_values_for_flow[bound_faces[outflow]] = init_pressure
        bound_for_flow = pp.BoundaryCondition(g, 
                                              faces=bound_faces, 
                                              cond=labels_for_flow)

        # The boundary conditions for transport. Transport here means
        # transport of solutes
        labels_for_transport = neu_faces.copy()
        labels_for_transport[inflow] = "dir"
        labels_for_transport[outflow] = "dir"
        bound_for_transport = pp.BoundaryCondition(g, 
                                                   faces=bound_faces, 
                                                   cond=labels_for_transport)

        # Set bc values. Note, at each face we have num_aq_components
        # Left side
        expanded_left = pp.fvutils.expand_indices_nd(bound_faces[inflow], num_aq_components)
        Ny = bound_faces[inflow].size
        bc_values_for_transport = np.zeros(g.num_faces * num_aq_components)
        
        bc_so4 = so4[0]
        
        bc_ca = 1 / (equil_consts[4] * bc_so4)   # Precipitation 
        bc_co3 = 0.5 / (equil_consts[3] * bc_ca) # Dissolution   
        
        bc_oh = oh[0] 
        bc_h = equil_consts[2] / bc_oh
    
        bc_hso4 = equil_consts[1] * bc_h * bc_so4
        bc_hco3 = equil_consts[0] * bc_h * bc_co3 
       
        bc_X = np.array([bc_ca, bc_co3, bc_so4, bc_h])
        bc_alpha = equil_consts[0:3] * np.exp(S*np.log(bc_X))
        bc_T = bc_X + S.T * bc_alpha 
        
        bc_values_for_transport[expanded_left] = np.tile(bc_T, Ny)
        
        # Right side
        expanded_right = pp.fvutils.expand_indices_nd(bound_faces[outflow], num_aq_components) 
        bc_T_right_side = init_X[0:4,0] + S.T * alpha_init[0:3,0]
        bc_values_for_transport[expanded_right] = np.tile(bc_T_right_side,  bound_faces[outflow].size)
        
        # Boundary condition for the passive tracer
        labels_for_tracer = neu_faces.copy()
        labels_for_tracer[np.logical_or(inflow, outflow)] = "dir"
        bound_for_tracer = pp.BoundaryCondition(g, 
                                                faces=bound_faces, 
                                                cond=labels_for_tracer)
        
        # The values
        bc_values_for_tracer = np.zeros(g.num_faces)
        bc_values_for_tracer[bound_faces[inflow]] = 3
        bc_values_for_tracer[bound_faces[outflow]] = 1

    else:

        # In the lower dimansions, no-flow (zero Neumann) conditions
        bound_for_flow = pp.BoundaryCondition(g)
        bc_values_for_flow = np.zeros(g.num_faces)
       
        bound_for_transport = pp.BoundaryCondition(g)
        bc_values_for_transport = np.zeros(g.num_faces * num_aq_components)
        
        bound_for_tracer = pp.BoundaryCondition(g)
        bc_values_for_tracer = np.zeros(g.num_faces)
    # end if-else

    # Initial guess for Darcy flux
    init_darcy_flux = np.zeros(g.num_faces)
    
    # --------------------------------- #

    # Set the values in dictionaries
    mass_data = {
        "porosity": porosity.copy(),
        "aperture": aperture.copy(),
        "specific_volume": specific_volume.copy(),
        "mass_weight": porosity.copy() * specific_volume.copy(),
        "num_components": num_components
    }

    flow_data = {
        "mass_weight": porosity * specific_volume.copy(),
        "bc_values": bc_values_for_flow,
        "bc": bound_for_flow,
        "permeability": kk,
        "second_order_tensor": pp.SecondOrderTensor(kk),
        "darcy_flux": init_darcy_flux,
    }

    transport_data = {
        "bc_values": bc_values_for_transport,
        "bc": bound_for_transport,
        "num_components": num_aq_components,
        "darcy_flux": init_darcy_flux,
    }
    
    passive_tracer_data = {
        "bc_values": bc_values_for_tracer,
        "bc": bound_for_tracer,
        "darcy_flux": init_darcy_flux,
        "mass_weight": porosity.copy() * specific_volume.copy() 
        }
    
    # The Initial values also serve as reference values
    reference_data = {
        "porosity": porosity.copy(),
        "permeability": K.copy(),
        "permeability_aperture_scaled": K.copy() * specific_volume.copy(),
        "aperture": aperture.copy(), 
        "surface_area": S_0,
        "non_reactive_volume_frac": non_reactive_volume_frac,
        "mass_weight": porosity.copy() * specific_volume.copy(),
    }
    
    # Concentration of ions
    conc_data = {
        # Primary components
        "Ca2+": init_X[0] ,
        "CO3":  init_X[1] ,
        "SO4":  init_X[2] ,
        "H+":   init_X[3] ,
        
        # Secondary species
        "HCO3": alpha_init[0] ,
        "HSO4": alpha_init[1] ,
        "OH-":  alpha_init[2] ,
        
        # Minerals
        "CaCO3": precipitated_init[0] ,
        "CaSO4": precipitated_init[1] ,   
    }
    
    # --------------------------------- #

    # Set the parameters in the dictionary
    # Treat this different to make sure parameter dictionary is constructed
    d = pp.initialize_data(g, d, mass_kw, mass_data)

    d[pp.PARAMETERS][flow_kw] = flow_data
    d[pp.DISCRETIZATION_MATRICES][flow_kw] = {}
    d[pp.PARAMETERS][transport_kw] = transport_data
    d[pp.DISCRETIZATION_MATRICES][transport_kw] = {}
    d[pp.PARAMETERS][tracer_kw] = passive_tracer_data
    d[pp.DISCRETIZATION_MATRICES][tracer_kw] = {}
    
    # The reference values
    d[pp.PARAMETERS]["reference"] = reference_data

    # The concentrations
    d[pp.PARAMETERS]["concentration"] = conc_data

    #d[pp.PARAMETERS]["iterating_concentration"] = conc_data.copy()
    
    # Set some data only in the highest dimension, in order to avoid techical issues later on
    if g.dim == gb.dim_max():

        # Make block matrices of S_W and equilibrium constants, one block per cell.
        cell_S = sps.block_diag([S for i in range(gb.num_cells())]).tocsr()
        cell_E = sps.block_diag([E for i in range(gb.num_cells())]).tocsr()     
       
        # Between species
        cell_equil_comp = sps.dia_matrix(
            (np.hstack([equil_consts[0:3] for i in range(gb.num_cells())]), 0),
            shape=(
                num_secondary_components * gb.num_cells(),
                num_secondary_components * gb.num_cells(),
            ),
        ).tocsr()
        
        # Between aquoues and precipitating species
        cell_equil_prec = sps.dia_matrix(
            (np.hstack([equil_consts[3:5] for i in range(gb.num_cells())]), 0),
            shape=(
                E.shape[0] * gb.num_cells(),
                E.shape[0] * gb.num_cells(),
                ),
            ).tocsr()
        
        d[pp.PARAMETERS][chemistry_kw] = {
            "equilibrium_constants_comp": equil_consts[0:3],
            "equilibrium_constants_prec": equil_consts[3:5],
            "cell_equilibrium_constants_comp": cell_equil_comp,
            "cell_equilibrium_constants_prec": cell_equil_prec,
            
            "stoic_coeff_S": S,
            "stoic_coeff_E": E,
            "cell_stoic_coeff_S": cell_S,
            "cell_stoic_coeff_E": cell_E
        }
        
        dt = 0.3
        d[pp.PARAMETERS][transport_kw].update({
            "time_step": dt,
            "final_time": 7000.0,
            "current_time": 0,
            "aqueous_components": aq_components
        })

        d[pp.PARAMETERS]["previous_time_step"] = {"time_step": []}
        d[pp.PARAMETERS]["previous_newton_iteration"] = {"Number_of_Newton_iterations":[]}
        d[pp.PARAMETERS]["grid_params"] = {} # Filled in later
        
    # end if

    # --------------------------------- #

    # Set initial values

    # First for the concentrations, replicated versions of the equilibrium.
    log_conc = init_X_log.reshape((init_T.size), order="F")
    
    # Minerals per cell
    mineral_per_cell = precipitated_init.reshape((2 * g.num_cells), order="F")
    
    d[pp.STATE].update({
        "dimension": g.dim * unity,
        pressure: init_pressure * unity,
        tot_var: init_T.copy(),
        log_var: log_conc,
        tracer_kw: unity,
        
        minerals: mineral_per_cell.copy(),
        min_1: precipitated_init[0].copy(),
        min_2: precipitated_init[1].copy(),
        
        "porosity": porosity.copy(),
        "aperture": aperture.copy(),
        "permeability": K.copy(),
        
        "aperture_difference": (
            constant_params.open_aperture()-aperture.copy() if gb.dim_max()>g.dim else 0 * unity
            ), 
        "ratio_perm": unity, 
        
        # Primary components
        "Ca2+": init_X[0] ,
        "CO3":  init_X[1] ,
        "SO4":  init_X[2] ,
        "H+":   init_X[3] ,
        
        # Secondary species
        "HCO3": alpha_init[0] ,
        "HSO4": alpha_init[1] ,
        "OH-":  alpha_init[2] ,
        
        # Minerals
        "CaCO3": precipitated_init[0] ,
        "CaSO4": precipitated_init[1] ,  

        pp.ITERATE: {
            pressure: init_pressure * unity,
            tot_var: init_T.copy(),
            log_var: log_conc.copy(),
            tracer_kw:  unity,
  
            minerals: mineral_per_cell.copy(),
            min_1: precipitated_init[0].copy(),
            min_2: precipitated_init[1].copy(),
            
            "porosity": porosity.copy(),
            "aperture": aperture.copy(),
            "permeability": K.copy()
        }
    })
    
    # --------------------------------- #

# end g,d-loop

#pp.plot_grid(gb,"dimension", figsize=(15,12))

#%% Loop over the edges:
    
for e, d in gb.edges():

    # Initialize the primary variables and the state in the dictionary
    d[pp.PRIMARY_VARIABLES] = {mortar_pressure: {"cells": 1},
                               mortar_transport: {"cells": num_aq_components},
                               mortar_tracer: {"cells": 1},
                                }
    pp.set_state(d)

    # Number of mortar cells
    mg_num_cells = d["mortar_grid"].num_cells
    unity = np.ones(mg_num_cells)
    
    # For transport we need the number of aqueos components and inital
    # conditions
    init_X_log_aq = init_X_log[aq_components].T[0] 
    
    # Aqueous part of the total concentration, based on the values from the fractrue(s) 
    init_T_aq = np.exp(init_X_log_aq) + S.T * alpha_init.T[0]  
    init_T_aq = np.tile(init_T_aq, mg_num_cells)
    
    d[pp.STATE].update({
        mortar_pressure: 0 * unity,
        mortar_transport: np.zeros(init_T_aq.size),
        mortar_tracer: np.zeros(unity.size),
        "mortar_dispersion": 0 * np.zeros(init_T_aq.size),

        pp.ITERATE: {
            mortar_pressure: 0 * unity,
            mortar_transport: np.zeros(init_T_aq.size),
            mortar_tracer: np.zeros(unity.size),
            "mortar_dispersion": 0 * np.zeros(init_T_aq.size),

        }
    })

    # Set the parameter dictionary
    d = pp.initialize_data(e, d, transport_kw, {"num_components": num_aq_components,
                                                "darcy_flux": unity})

    d[pp.PARAMETERS][flow_kw] = {"darcy_flux": unity}
    d[pp.DISCRETIZATION_MATRICES][flow_kw] = {}
   
    d[pp.PARAMETERS][tracer_kw] = {"darcy_flux": unity}
    d[pp.DISCRETIZATION_MATRICES][tracer_kw] = {}  

# end e,d-loop
 # For flow, we also need the the normal diffusivity
update_param.update_interface(gb)

#%% The data in various dimensions
gb_2d = gb.grids_of_dimension(2)[0]
data_2d = gb.node_props(gb_2d)

gb_0d = gb.grids_of_dimension(0)[0]
data_0d = gb.node_props(gb_0d)

# Fill in grid related parameters
grid_list = [g for g,_ in gb]
edge_list = [e for e,_ in gb.edges()]
data_2d[pp.PARAMETERS]["grid_params"].update({
    "grid_list": grid_list,
    "edge_list": edge_list,
    "mortar_projection_single": pp.ad.MortarProjections(gb, grids=grid_list, edges=edge_list, nd=1),
    "mortar_projection_several": pp.ad.MortarProjections(gb, grids=grid_list, edges=edge_list, nd=num_components),
    "trace_single": pp.ad.Trace(grids=grid_list, nd=1),
    "trace_several": pp.ad.Trace(grids=grid_list, nd=num_components),
    "divergence_single": pp.ad.Divergence(grid_list, dim=1),
    "divergence_several": pp.ad.Divergence(grid_list, dim=num_aq_components),
    
    "all_2_aquatic": constant_params.all_2_aquatic_mat(
        aq_components, num_components, num_aq_components, gb.num_cells()
        ),
    "all_2_aquatic_faces":constant_params.all_2_aquatic_mat(
        aq_components, num_components, num_aq_components, gb.num_faces()
        ),
    "all_2_aquatic_interface": constant_params.all_2_aquatic_mat(
        aq_components, num_components, num_aq_components, gb.num_mortar_cells()
        ),
    
    "sum_mat": constant_params.sum_mat(gb.num_cells(), 2*gb.num_cells()),
    "extension_mat": constant_params.enlarge_mat_2(gb_2d.num_cells,  gb.num_cells()),
    "dim_2_mat": constant_params.dimensional_mat(2, gb),
    "dim_1_mat": constant_params.dimensional_mat(1, gb),
    "dim_0_mat": constant_params.dimensional_mat(0, gb),
    
    "extend_flux": constant_params.enlarge_mat_2(gb.num_faces(), num_aq_components*gb.num_faces()),
    "extend_edge_flux": constant_params.enlarge_mat_2(gb.num_mortar_cells(), num_aq_components*gb.num_mortar_cells())
    })

#%% Conctruct an dof_manager, equation manager and the initial equations
dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)

equation_manager = equations.gather(gb,
                                    dof_manager=dof_manager,
                                    equation_manager=equation_manager)
#%% Prepere for exporting
refinement_level = "refinement_level_" + str(ref_level)
picture_folder_name = "pictures/"+refinement_level 
to_paraview = pp.Exporter(gb, file_name="variables", 
                          folder_name=picture_folder_name
                          )

fields = ["Ca2+", "CO3", "SO4", "H+", 
          "HCO3","HSO4", "OH-",
          "CaCO3", "CaSO4", 
          "ratio_perm", "passive_tracer",
          "aperture_difference", "pressure"]


time_store = np.array([0.1, 1.3, 2.5, 3.9, 4.5, 
                       5.1, 6.3])*1e3
j=0

conc_dict = data_2d[pp.PARAMETERS]["concentration"]
current_time = data_2d[pp.PARAMETERS]["transport"]["current_time"]
final_time = data_2d[pp.PARAMETERS]["transport"]["final_time"]

#%% Time loop
while current_time < final_time:
    print(f"current time {current_time}")
    
    # Solve
    solve_eqs(gb, dof_manager, equation_manager)
    
    # Connect and collect the equations
    equation_manager = equations.gather(gb,
                                        dof_manager=dof_manager,
                                        equation_manager=equation_manager,
                                        iterate=False)
    
    current_time = data_2d[pp.PARAMETERS]["transport"]["current_time"]  
    
    if j < len(time_store) and np.abs(current_time - time_store[j]) < 6:
        j+=1
        to_paraview.write_vtu(fields, time_step = current_time)
    # end if
    
# end i-loop

# Store result at the final time
to_paraview.write_vtu(fields, time_step = current_time)
print(f"current time {current_time}")

 #%% Check mass action law.
# Recall that the secondary species are reactants
# and the components are the products. Thus
# CaCO3 <-> Ca + CO3; Ca * CO3 / CaCO3 = 1
# where eq is the equilibrium constant when I take the exponential
# in the first part of the code

eq_test1 = np.minimum( 1 - conc_dict["Ca2+"] * conc_dict["CO3"] * equil_consts[3] , conc_dict["CaCO3"]) < 1e-14
eq_test2 = np.minimum( 1 - conc_dict["Ca2+"] * conc_dict["SO4"] * equil_consts[4] , conc_dict["CaSO4"]) < 1e-14
eq_test3 = np.abs(conc_dict["HCO3"] / (conc_dict["H+"] * conc_dict["CO3"]) - equil_consts[0]) < 1e-2
eq_test4 = np.abs(conc_dict["HSO4"] / (conc_dict["H+"] * conc_dict["SO4"]) - equil_consts[1]) < 1e-2
eq_test5 = np.abs(conc_dict["OH-"] * conc_dict["H+"] - equil_consts[2]) < 1e-2

#%% Store the grid bucket
gb_list = [gb] 
folder_name = "to_study/" # Assume this folder exist
refinement_level = "refinement_level_" + str(ref_level)
gb_refinement_name = folder_name + "gb_" + refinement_level 

def write_pickle(obj, path):
    """
    Store the current grid bucket for the convergence study
    """
    path = Path(path)
    raw = pickle.dumps(obj)
    with open(path, "wb") as f:
        raw = f.write(raw)        
    return
write_pickle(gb_list, gb_refinement_name)

#%% Plotting
newton_steps = data_2d[pp.PARAMETERS]["previous_newton_iteration"]["Number_of_Newton_iterations"]
time_steps = data_2d[pp.PARAMETERS]["previous_time_step"]["time_step"] 

newton_steps = np.asarray(newton_steps)
time_steps = np.asarray(time_steps)
time_points = np.linspace(0, final_time, newton_steps.size, endpoint=True)

vals_to_store = np.column_stack(
    (
     time_points,
     newton_steps,
     time_steps,
     )
    )

file_name = folder_name + "temporal_vals_" + refinement_level
np.savetxt(file_name + ".csv", vals_to_store,
            delimiter=",", header="time_points, newton_steps, time_steps")
