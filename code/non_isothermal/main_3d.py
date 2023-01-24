"""
Main script
@author: uw
"""

# %% Import the neseccary packages
import numpy as np
import scipy.sparse as sps
import porepy as pp

import equations
from update_param import rho
from solve_non_linear import solve_eqs
from update_param import taylor_app_equil, update_interface, update_intersection_aperture, equil_state_to_param

import pickle
from pathlib import Path

import sys
sys.path.insert(0, "..")
import create_mesh
import constant_params
 
# %% Initialize variables related to the chemistry and the domain

# Start with the chemical related variables
# i.e. stoichiometric coeffients, number of species

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

# Bookkeeping
# Index of aquatic and fixed components, referring to the cell-wise T vector
aq_components = np.array([0, 1, 2, 3])
fixed_components = 0
num_aq_components = aq_components.size
num_fixed_components = 0
num_components = num_aq_components + num_fixed_components
num_secondary_components = S.shape[0]

# Define fractures and a gb
mesh_args={"mesh_size_frac" : 0.03, 
           "mesh_size_min"  : 0.05, 
           "mesh_size_bound": 0.05}
gb = create_mesh.create_gb(mesh_args=mesh_args,
                           dim=3,
                           fractured=True)

phys = gb.bounding_box()[1] # [m]

domain = {"xmin": 0, 
          "ymin": 0,
          "zmin": 0,
          "xmax": phys[0], 
          "ymax": phys[1], 
          "zmax": phys[2]}  

# Keywords
mass_kw = "mass"
chemistry_kw = "chemistry"
transport_kw = "transport"
flow_kw = "flow"

pressure = "pressure"
tot_var = "T"
log_var = "log_X"
temperature = "temperature"
tracer = "passive_tracer"

minerals = "minerals"
min_1 = "CaCO3"
min_2 = "CaSO4"

mortar_pressure = "mortar_pressure"
mortar_transport = "mortar_transport"
mortar_temperature_convection = "mortar_temperature_convection"
mortar_temperature_conduction = "mortar_temperature_conduction"
mortar_tracer = "mortar_tracer"

# %% Loop over the gb, and set initial and default data

# Initial matrix permeability
matrix_permeability = 1e-11

# Some temperature param
init_temp = constant_params.ref_temp() # in K
bc_temp = init_temp - 30

init_equil_const = constant_params.ref_equil() * taylor_app_equil(temp=init_temp)
bc_equil_const = constant_params.ref_equil() * taylor_app_equil(temp=bc_temp)

# Initial uniform pressure
init_pressure = constant_params.ref_p() # [Pa]

for g, d in gb:
    
    gb.add_node_props(["is_tangential"])
    d["is_tangential"] = True

    # Initialize the primary variable dictionaries
    d[pp.PRIMARY_VARIABLES] = {pressure:    {"cells": 1},
                               tot_var:     {"cells": num_components},
                               log_var:     {"cells": num_components},
                               minerals:    {"cells": 2},
                               temperature: {"cells": 1},
                               tracer:      {"cells": 1},
                               "porosity": {"cells": 1},
                               "aperture": {"cells": 1},
                               "permeability": {"cells":1},
                               "equilibrium_constants": {"cells": 5}
                               }

    # Initialize a state
    pp.set_state(d)

    unity = np.ones(g.num_cells)
    
    # --------------------------------- #
    
    # Set inital concentration values
    
    if gb.dim_max() == g.dim:
        
        so4 = 10 * unity
        ca = 0.9 / (init_equil_const[4] * so4) * unity
        co3 = 1 / (init_equil_const[3] * ca) * unity
        oh = 1.5e3 * unity
        h = init_equil_const[2] / oh * unity
        hco3 = init_equil_const[0] * h * co3 * unity
        hso4 = init_equil_const[1] * h * so4 * unity
    
        # Initally calcite is present, but not anhydrite
        precipitated_init = np.zeros((2,g.num_cells))
        precipitated_init[0] = 20
        
        init_X = np.array([ca, co3, so4, h])
        init_X_log = np.log(init_X)
        alpha_init = np.array([hco3, hso4, oh])
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
        
        # Have initally calcite present, but not anhydrite 
        precipitated_init = np.zeros((2, g.num_cells))
        precipitated_init[0] = 20.0
   
        init_X = np.array([ca, co3, so4, h])
        init_X_log = np.log(init_X)
        
        alpha_init = np.array([hco3, hso4, oh])
        init_T = init_X + S.T * alpha_init + E.T * precipitated_init
        init_T = init_T.reshape((init_T.size), order="F")
    # end if-else
    
    # --------------------------------- #
    
    # Initial porosity, permeablity and aperture.
 
    # # Number of moles
    mol_CaCO3 = precipitated_init[0] * g.cell_volumes 
    mol_CaSO4 = precipitated_init[1] * g.cell_volumes

    # Next convert mol to g
    mass_CaCO3 = mol_CaCO3 * constant_params.molar_mass_CaCO3() 
    mass_CaSO4 = mol_CaSO4 * constant_params.molar_mass_CaSO4()

    # Densities
    density_CaCO3 = constant_params.density_CaCO3()  # kg/m^3
    density_CaSO4 = constant_params.density_CaSO4()  # kg/m^3

    # the mineral volumes
    mineral_vol_CaCO3 = mass_CaCO3 / density_CaCO3 
    mineral_vol_CaSO4 = mass_CaSO4 / density_CaSO4 
    

    S_0 = np.power(g.cell_volumes, 2/3) # reactive surface area 
        
    mineral_width_CaCO3 = mineral_vol_CaCO3 / S_0
    mineral_width_CaSO4 = mineral_vol_CaSO4 / S_0
    
    open_aperture = constant_params.open_aperture()
    
    if g.dim == gb.dim_max():
       aperture = unity
    elif g.dim == 1 : #gb.dim_min():
        aperture = update_intersection_aperture(gb,g)
    else:
        aperture = open_aperture - (
            mineral_width_CaCO3 +  mineral_width_CaSO4 
            ) 
    # end if

    
    specific_volume = np.clip(
        np.power(aperture, gb.dim_max()-g.dim),
        a_max=1, a_min=0)

    porosity = (
        1 - (mineral_vol_CaCO3 + mineral_vol_CaSO4) / g.cell_volumes  
        ) 
    
    non_reactive_volume_frac = 0.8 if g.dim==gb.dim_max() else 0.0

    if g.dim == gb.dim_max():
        porosity -= non_reactive_volume_frac # Porosity of the non-reactive mineral
        K = matrix_permeability * unity
    else:
        K = np.power(aperture, 2) / 12
    # end if-else
    
    kk = K * specific_volume / constant_params.dynamic_viscosity()
    # --------------------------------- #

    # Boundary conditions
    bound_faces = g.tags["domain_boundary_faces"].nonzero()[0]

    if bound_faces.size != 0:  # Non-zero in 2D/3D, zero otherwise

        bound_faces_centers = g.face_centers[:, bound_faces]
        
        # Neumann faces
        neu_faces = np.array(["neu"] * bound_faces.size)
        # Define the inflow and outflow in the domain
        inflow = bound_faces_centers[0,:] < domain["xmin"] + 1e-4
        outflow = bound_faces_centers[0,:] > domain["xmax"] - 1e-4
        
        # The BC labels for the flow problem
        labels_for_flow = neu_faces.copy()
        labels_for_flow[outflow] = "dir"
        labels_for_flow[inflow] = "dir"

        # Set the BC values for flow
        bc_values_for_flow = np.zeros(g.num_faces)
        bc_values_for_flow[bound_faces[inflow]] = 7*init_pressure
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
        expanded_left = pp.fvutils.expand_indices_nd(bound_faces[inflow], num_aq_components)
        bc_values_for_transport = np.zeros(g.num_faces * num_aq_components)
          
        bc_so4 = so4[0]
        bc_ca = 1 / (bc_equil_const[4] * bc_so4) # Precipitation of CaSO4
        bc_co3 = 0.5 / (bc_equil_const[3] * bc_ca) # Dissolution of CaCO3
        bc_oh = oh[0]
        bc_h = bc_equil_const[2] / bc_oh
        bc_hso4 = bc_equil_const[1] * bc_h * bc_so4
        bc_hco3 = bc_equil_const[0] * bc_h * bc_co3
        
        bc_X = np.array([bc_ca, bc_co3, bc_so4, bc_h])
        bc_alpha =np.array([bc_hco3, bc_hso4, bc_oh]) 
        bc_T = bc_X + S.T * bc_alpha 
        
        bc_values_for_transport[expanded_left] = np.tile(bc_T, bound_faces[inflow].size)
        
        # Right side
        expanded_right = pp.fvutils.expand_indices_nd(bound_faces[outflow], num_aq_components)
        bc_right_side = init_X[0:4,1] + S.T * alpha_init[0:3,0]
    
        bc_values_for_transport[expanded_right] = np.tile(bc_right_side, bound_faces[outflow].size)
        
        # Boundary conditions for temperature
        labels_for_temp = neu_faces.copy()
        labels_for_temp[np.logical_or(inflow,outflow)] = "dir"
        
        bound_for_temp = pp.BoundaryCondition(g, 
                                              faces=bound_faces, 
                                              cond=labels_for_temp)

        # The bc values
        bc_values_for_temp = np.zeros(g.num_faces)
        bc_values_for_temp[bound_faces[inflow]] = bc_temp  # in K
        bc_values_for_temp[bound_faces[outflow]] = init_temp 
        
        # Boundary conditions for the passive tracer
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
        bc_values_for_flow = np.zeros(g.num_faces)
        bound_for_flow = pp.BoundaryCondition(g)

        bc_values_for_transport = np.zeros(g.num_faces * num_aq_components)
        bound_for_transport = pp.BoundaryCondition(g)

        bc_values_for_temp = np.zeros(g.num_faces)
        bound_for_temp = pp.BoundaryCondition(g)
        
        bc_values_for_tracer = np.zeros(g.num_faces)
        bound_for_tracer = pp.BoundaryCondition(g)
    # end if-else

    # Initial guess for Darcy flux
    init_darcy_flux = np.ones(g.num_faces)
    #init_darcy_flux[g.get_internal_faces()] = 1.0
    
    # Calulate the effective heat capacity
    fluid_density = rho(init_pressure, init_temp)
    
    # The specific heat capacities
    # We assume temperature is only affected by water and non-reactive mineral
    # Hence, we put the specifc heat capacites of the minerals to zero
    
    # Solid (reactive and non-reactive minerals) density, 
    solid_density = 0 
    if g.dim==gb.dim_max(): # Only non-reactive mineral in the matrix
        solid_density += 2500 
    # end if
    
    heat_capacity = (
        porosity * fluid_density * constant_params.specific_heat_capacity_fluid() + # fluid part
        (1-porosity) * solid_density * constant_params.specific_heat_capacity_solid()  # solid poart
        )

    conduction = (
        porosity * constant_params.fluid_conduction() + 
        (1-porosity) * constant_params.solid_conduction()
        )
    
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

    temp_data = {
        "mass_weight": heat_capacity * specific_volume.copy(),
        "solid_density": solid_density,
        "bc_values": bc_values_for_temp,
        "bc": bound_for_temp,
        "darcy_flux": init_darcy_flux, 
        "second_order_tensor": pp.SecondOrderTensor(
            conduction * specific_volume.copy()
            )
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
        "specific_volume": specific_volume.copy(),
        "surface_area": S_0,
        "mass_weight": porosity.copy() * specific_volume.copy(),   
        "non_reactive_volume_frac": non_reactive_volume_frac

        
    }

    # --------------------------------- #

    # Set the parameters in the dictionary
    # Treat this different to make sure parameter dictionary is constructed
    d = pp.initialize_data(g, d, mass_kw, mass_data)

    d[pp.PARAMETERS][flow_kw] = flow_data
    d[pp.DISCRETIZATION_MATRICES][flow_kw] = {}
    d[pp.PARAMETERS][transport_kw] = transport_data
    d[pp.DISCRETIZATION_MATRICES][transport_kw] = {}
    d[pp.PARAMETERS][temperature] = temp_data
    d[pp.DISCRETIZATION_MATRICES][temperature] = {}
    d[pp.PARAMETERS][tracer] = passive_tracer_data
    d[pp.DISCRETIZATION_MATRICES][tracer] = {}

    # The reference values
    d[pp.PARAMETERS]["reference"] = reference_data

    # The concentrations
    d[pp.PARAMETERS]["concentration"] = {
        # Primary components
        "Ca2+": init_X[0] ,
        "CO3": init_X[1] ,
        "SO4": init_X[2] ,
        "H+": init_X[3] ,
        
        # Secondary species
        "HCO3": alpha_init[0] ,
        "HSO4": alpha_init[1] ,
        "OH-": alpha_init[2] ,
        
        # Minerals
        "CaCO3": precipitated_init[0] ,
        "CaSO4": precipitated_init[1] ,
    }

    # Set some data only in the highest dimension, in order to avoid techical issues later on
    if g.dim == gb.dim_max():

        # Make block matrices of S_W and equilibrium constants, one block per cell.
        cell_S = sps.block_diag([S for i in range(gb.num_cells())]).tocsr()
        cell_E = sps.block_diag([E for i in range(gb.num_cells())]).tocsr()     
        
        d[pp.PARAMETERS][chemistry_kw] = {        
            "stoic_coeff_S": S,
            "stoic_coeff_E": E,
            "cell_stoic_coeff_S": cell_S,
            "cell_stoic_coeff_E": cell_E
        }

        d[pp.PARAMETERS][transport_kw].update({
            "time_step": 0.05, 
            "current_time": 0,
            "final_time": 2000,
            "aqueous_components": aq_components
        })
        d[pp.PARAMETERS]["grid_params"] = {
            "all_2_aquatic": 
                constant_params.all_2_aquatic_mat(aq_components, num_components, num_aq_components, gb.num_cells()),
            "all_2_aquatic_at_interface":
                constant_params.all_2_aquatic_mat(aq_components, num_components, num_aq_components, gb.num_mortar_cells())}
        d[pp.PARAMETERS]["previous_time_step"] = {"time_step": []}
        d[pp.PARAMETERS]["previous_newton_iteration"] = {"Number_of_Newton_iterations":[]}
    # end if

    # --------------------------------- #

    # Set initial values

    # First for the concentrations, replicated versions of the equilibrium.
    log_conc = init_X_log.reshape((init_T.shape), order="F")
    
    # Minerals per cell
    mineral_per_cell = precipitated_init.reshape((2 * g.num_cells), order="F")
    
    d[pp.STATE].update({
        "dimension" : g.dim * unity, 
        pressure: init_pressure * unity,
        tot_var: init_T.copy(),
        log_var: log_conc,
        temperature: init_temp * unity, 
        tracer: unity,
        
        minerals: mineral_per_cell.copy(),
        min_1: precipitated_init[0].copy() ,
        min_2: precipitated_init[1].copy() ,
        
        "porosity": porosity,
        "aperture": aperture,
        "permeability": K,
        "equilibrium_constants": np.tile(init_equil_const, g.num_cells),
        
        "aperture_difference": 0 * unity, 
        "ratio_perm": unity, 

        pp.ITERATE: {
            pressure: init_pressure * unity,
            tot_var: init_T.copy(),
            log_var: log_conc.copy(),
            temperature: init_temp * unity, # in K (ie degree celsius + 273.15)
            tracer: unity,
            
            minerals: mineral_per_cell.copy(),
            min_1: precipitated_init[0].copy() ,
            min_2: precipitated_init[1].copy() ,
            
            "porosity": porosity,
            "aperture": aperture,
            "permeability": K,
            "equilibrium_constants": np.tile(init_equil_const, g.num_cells),
            
        }
    })
# end g,d-loop

equil_state_to_param(gb)
#pp.plot_grid(gb, "dimension", figsize=(15,12))

#%% Loop over the edges:
i=0
for e, d in gb.edges():
  
    # Initialize the primary variables and the state in the dictionary
    d[pp.PRIMARY_VARIABLES] = {mortar_pressure:    {"cells": 1},
                               mortar_transport:   {"cells": num_aq_components},
                               mortar_temperature_convection: {"cells": 1},
                               mortar_temperature_conduction: {"cells": 1},
                               mortar_tracer:      {"cells": 1}
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
    
    edge_t = (
        init_temp * 
        rho(init_pressure, init_temp) * 
        constant_params.specific_heat_capacity_fluid() 
        ) * unity
    #edge_t[int(mg_num_cells/2)::2]*=-1
    
    d[pp.STATE].update({
        mortar_pressure: 0.0 * unity,
        mortar_transport: 0*init_T_aq,
        mortar_temperature_convection: 0*edge_t * 1e-6,
        mortar_temperature_conduction: 0 * unity,
        mortar_tracer: np.zeros(unity.size),

        pp.ITERATE: {
            mortar_pressure: 0.0 * unity,
            mortar_transport: 0*init_T_aq.copy(),
            mortar_temperature_convection: 0*edge_t * 1e-6,
            mortar_temperature_conduction: 0 * unity,
            mortar_tracer: np.zeros(unity.size)
        }
    })

    #------------------------------------#
    # Set the parameter dictionary
    d = pp.initialize_data(e, d, transport_kw, {"num_components": num_aq_components,
                                                "darcy_flux": 0*unity}
                           )

    # For temperature, we need the flux
    d[pp.PARAMETERS][temperature] = {"darcy_flux": 0*unity}
    d[pp.DISCRETIZATION_MATRICES][temperature] = {}
    
    # Tracer
    d[pp.PARAMETERS][tracer]={"darcy_flux": 0*unity}
    d[pp.DISCRETIZATION_MATRICES][tracer] = {}
    
    # For flow, we need the the flux   
    d[pp.PARAMETERS][flow_kw] = {"darcy_flux": 0*unity}
    d[pp.DISCRETIZATION_MATRICES][flow_kw] = {}
# end e,d-loop

# Finally, set interfacial conductive flux
update_interface(gb)

#%% The data in various dimensions
gb_3d = gb.grids_of_dimension(gb.dim_max())[0]
data_3d = gb.node_props(gb_3d)

gb_frac = gb.grids_of_dimension(gb.dim_max()-1)[0]
data_frac = gb.node_props(gb_frac)

# Fill in grid related parameters
grid_list = [g for g,_ in gb]
edge_list = [e for e,_ in gb.edges()]
data_3d[pp.PARAMETERS]["grid_params"].update({
    "grid_list": grid_list,
    "edge_list": edge_list,
    "mortar_projection_single": pp.ad.MortarProjections(gb, grids=grid_list, edges=edge_list, nd=1),
    "mortar_projection_several": pp.ad.MortarProjections(gb, grids=grid_list, edges=edge_list, nd=num_components),
    "trace_single": pp.ad.Trace(grid_list, nd=1),
    "trace_several": pp.ad.Trace(grid_list, nd=num_components),
    "divergence_single": pp.ad.Divergence(grid_list, dim=1),
    "divergence_several": pp.ad.Divergence(grid_list, dim=num_aq_components),
    
    "all_2_aquatic": constant_params.all_2_aquatic_mat(
        aq_components, num_components, num_aq_components, gb.num_cells() 
        ),
    "all_2_aquatict_at_interface": constant_params.all_2_aquatic_mat(
        aq_components, num_components, num_aq_components, gb.num_mortar_cells()
        ),
    
    # mapping between matrix and rest of the medium
    "mat_to_omega": constant_params.enlarge_mat_2(gb_3d.num_cells, gb.num_cells()),
    
    "sum_mat": constant_params.sum_mat(gb.num_cells(), 2*gb.num_cells() ),
    "dim_2_mat": constant_params.dimensional_mat(3, gb),
    "dim_1_mat": constant_params.dimensional_mat(2, gb),
    "dim_0_mat": constant_params.dimensional_mat(1, gb),
    
    "extend_flux": constant_params.enlarge_mat_2(gb.num_faces(), num_aq_components*gb.num_faces()),
    "extend_edge_flux": constant_params.enlarge_mat_2(gb.num_mortar_cells(), num_aq_components*gb.num_mortar_cells()),
    
    "temp_equil": constant_params.enlarge_mat_2(gb.num_cells(), 5*gb.num_cells())
    })

#%% Conctruct an dof_manager, equation manager 
dof_manager = pp.DofManager(gb)
equation_manager = pp.ad.EquationManager(gb, dof_manager)

#%% and the initial equations
equation_manager = equations.gather(gb,
                                    dof_manager=dof_manager,
                                    equation_manager=equation_manager)

#%% Prepere for exporting

to_paraview = pp.Exporter(gb, file_name="variables", folder_name="pictures/3D")
fields = ["dimension", "pressure", 
          "Ca2+", "CO3", "SO4", "H+", 
          "HCO3", "HSO4", "OH-",
          "CaCO3", "CaSO4", 
          "passive_tracer","temperature",
          "aperture_difference", "ratio_perm"]

time_store = np.arange(1,20) * 1e2
j=0

current_time = data_3d[pp.PARAMETERS]["transport"]["current_time"]
final_time = data_3d[pp.PARAMETERS]["transport"]["final_time"]

#%% Time loop
while current_time < final_time:
    
    print(f"Current time {current_time}")

    # Solve
    solve_eqs(gb, dof_manager, equation_manager)
    
    # Connect and collect the equations
    equation_manager = equations.gather(gb,
                                        dof_manager=dof_manager,
                                        equation_manager=equation_manager,
                                        iterate=False)

    current_time = data_3d[pp.PARAMETERS]["transport"]["current_time"]

    if j < len(time_store) and np.abs(current_time-time_store[j]) < 4:
       
        j += 1
        to_paraview.write_vtu(fields, time_step = current_time)
    # end if
    
# end time-loop

# Store result at the final time
to_paraview.write_vtu(fields, time_step = current_time)    
print(f"Current time {current_time}")

#%% Store the grid bucket
gb_list = [gb] 
folder_name = "to_study/3D/" # Assume this folder exist
gb_name = "gb_3D" 

def write_pickle(obj, path):
    """
    Store the current grid bucket for the convergence study
    """
    path = Path(path)
    raw = pickle.dumps(obj)
    with open(path, "wb") as f:
        raw = f.write(raw) 
    return

write_pickle(gb_list,folder_name + gb_name)

#%% Store time step and number of Newton iterations
newton_steps = data_3d[pp.PARAMETERS]["previous_newton_iteration"]["Number_of_Newton_iterations"]
time_steps = data_3d[pp.PARAMETERS]["previous_time_step"]["time_step"] 

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

file_name = folder_name + "temporal_vals_" + gb_name
np.savetxt(file_name + ".csv", vals_to_store, 
            delimiter=",", header="time_points, newton_steps, time_steps")

