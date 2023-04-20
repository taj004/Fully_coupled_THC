"""
@author: uw
"""

import numpy as np
import porepy as pp
import scipy.sparse as sps

from update_param import rho

import sys

sys.path.insert(0, "..")
import constant_params


def repeat(v, reps, dof_manager=None, abs_val=True, to_ad=True):
    """
    Repeat a vector v, reps times
    Main target is the flux vectors, needed in the transport calculations
    """

    if isinstance(v, np.ndarray):
        num_v = v

    elif dof_manager is not None:
        # Get expression for evaluation and check attributes
        expression_v = v.evaluate(dof_manager)
        if isinstance(expression_v, np.ndarray):
            num_v = expression_v
        elif hasattr(expression_v, "val"):
            num_v = expression_v.val
    else:
        raise ValueError("Cannot repeat this vector")
    # end if-else

    v_reps = np.repeat(num_v, reps)

    # Wrap the array in Ad.
    # Note that we return absolute value of the expanded vectors
    # The reason is these vectors are ment to use as scale values
    # in the solute transport equation. The signs are handled in the upwind discretization

    if abs_val is True:
        v_reps = np.abs(v_reps)
    # end if

    if to_ad:
        v_reps = pp.ad.Array(v_reps)

    return v_reps


def to_vec(gb, param_kw, param, size_type="cells", to_ad=False):
    """
    Make an Ad-vector of a parameter param, from a param_kw
    """

    if size_type == "cells":
        vec = np.zeros(gb.num_cells())
    elif size_type == "faces":
        vec = np.zeros(gb.num_faces())
    else:
        raise ValueError("Unknown size type")
    # end if-else

    val = 0
    for g, d in gb:

        if size_type == "cells":
            num_size = g.num_cells
        else:
            num_size = g.num_faces
        # end if-else

        # Extract from the dictionary
        x = d[pp.PARAMETERS][param_kw][param]

        if isinstance(x, np.ndarray) is False:
            x *= np.ones(num_size)
        # end if

        # Indices
        inds = slice(val, val + num_size)
        # Store values and update for the next grid
        vec[inds] = x.copy()
        val += num_size
    # end g,d-loop

    if to_ad:
        return pp.ad.Array(vec)
    else:
        return vec

    #%%


def gather(gb, dof_manager, equation_manager, iterate=False):

    """
    Collect and discretize equations on a GB

    Parameters
    ----------
    gb : grid bucket
    dof_manager : dof_manager for ...
    equation_manager: an equation manager that keeps the
                      information about the equations
    iterate : bool, whether the equations are updated in the Newton iterations (True),
                    or formed at the start of a time step (False).

    """
    #%%

    # Keywords:
    mass_kw = "mass"
    flow_kw = "flow"
    transport_kw = "transport"
    chemistry_kw = "chemistry"

    # Primary variables:
    pressure = "pressure"
    tot_var = "T"  # T: The total concentration of the primary components
    log_var = "log_X"  # Logarithm of the aqueous part of the total concentration
    temperature = "temperature"
    minerals = "minerals"
    tracer = "passive_tracer"

    mortar_pressure = "mortar_pressure"
    mortar_tot_var = "mortar_transport"
    mortar_temperature_convection = "mortar_temperature_convection"
    mortar_temperature_conduction = "mortar_temperature_conduction"
    mortar_tracer = "mortar_tracer"

    # Loop over the gb
    for g, d in gb:

        # Get data that are necessary for later use
        if g.dim == gb.dim_max():
            data_transport = d[pp.PARAMETERS][transport_kw]
            data_chemistry = d[pp.PARAMETERS][chemistry_kw]
            data_prev_time = d[pp.PARAMETERS]["previous_time_step"]
            data_prev_newton = d[pp.PARAMETERS]["previous_newton_iteration"]
            data_grid = d[pp.PARAMETERS]["grid_params"]
            data_ref = d[pp.PARAMETERS]["reference"]
        # end_if
    # end g,d-loop

    # The list of grids and edges
    grid_list = data_grid["grid_list"]
    edge_list = data_grid["edge_list"]

    # Ad representations of the primary variables.
    p = equation_manager.merge_variables(
        [(g, pressure) for g in grid_list]
        )

    T = equation_manager.merge_variables(
        [(g, tot_var) for g in grid_list]
        )

    log_X = equation_manager.merge_variables(
        [(g, log_var) for g in grid_list]
        )

    precipitate = equation_manager.merge_variables(
        [(g, minerals) for g in grid_list]
        )

    temp = equation_manager.merge_variables(
        [(g, temperature) for g in grid_list]
        )

    passive_tracer = equation_manager.merge_variables(
        [(g, tracer) for g in grid_list]
        )

    porosity = equation_manager.merge_variables(
        [(g, "porosity") for g in grid_list]
        )

    aperture = equation_manager.merge_variables(
        [(g, "aperture") for g in grid_list]
        )

    perm = equation_manager.merge_variables(
        [(g, "permeability") for g in grid_list]
        )

    equilibrium_constants = equation_manager.merge_variables(
        [(g, "equilibrium_constants") for g in grid_list]
    )

    # The fluxes over the interfaces
    if len(edge_list) > 0:
        # Flow
        v = equation_manager.merge_variables(
            [e, mortar_pressure] for e in edge_list
            )

        # Transport
        eta = equation_manager.merge_variables(
            [e, mortar_tot_var] for e in edge_list
            )

        # Temperature
        w = equation_manager.merge_variables(
            [e, mortar_temperature_convection] for e in edge_list
        )
        q = equation_manager.merge_variables(
            [e, mortar_temperature_conduction] for e in edge_list
        )

        # Passive tracer
        eta_tracer = equation_manager.merge_variables(
            [e, mortar_tracer] for e in edge_list
        )
    # end if

    #%% Define equilibrium equations for Ad

    # Cell-wise chemical values
    cell_S = data_chemistry["cell_stoic_coeff_S"]
    cell_E = data_chemistry["cell_stoic_coeff_E"]
    cell_equil_comp = data_chemistry["cell_equilibrium_constants_comp"]
    cell_equil_prec = data_chemistry["cell_equilibrium_constants_prec"]

    def equilibrium_all_cells(total, log_primary, precipitated_species):
        """
        Residual form of the equilibrium problem,
        posed for all cells together.
        """
        # The secondary species
        secondary_C = cell_equil_comp * pp.ad.exp(cell_S * log_primary)
        # Residual equation
        eq = (
            pp.ad.exp(log_primary)
            + cell_S.transpose() * secondary_C
            + cell_E.transpose() * precipitated_species
            - total
        )
        return eq

    # Wrap the equilibrium residual into an Ad function.
    equil_ad = pp.ad.Function(equilibrium_all_cells, "equil")

    # Finally, make it into an Expression which can be evaluated.
    equilibrium_eq = equil_ad(T, log_X, precipitate)

    #%% The flow equation

    # The divergence
    div = data_grid["divergence_single"]

    # A projection between subdomains and mortar grids
    if len(edge_list) > 0:
        mortar_projection = data_grid["mortar_projection_single"]
    # end if

    # # Time step
    dt = data_transport["time_step"]

    # BC for flow
    bound_flux = pp.ad.BoundaryCondition(flow_kw, grid_list)

    if iterate:  # Newton iterations
        mass_density = pp.ad.MassMatrixAd(keyword=flow_kw, grids=grid_list)

        mass_density_prev = data_prev_time["mass_density_prev"]  # previous time step
        p_prev = data_prev_time["p_prev"]  # previous time step
        temp_prev = data_prev_time["temp_prev"]
    else:
        # The pressure and temperature at the previous time step
        p_prev = p.previous_timestep()
        temp_prev = temp.previous_timestep()
        # Acculmuation for the pressure equation
        mass_density = pp.ad.MassMatrixAd(keyword=flow_kw, grids=grid_list)
        mass_density_prev = pp.ad.MassMatrixAd(keyword=flow_kw, grids=grid_list)

        # Store for the Newton calculations
        data_prev_time["p_prev"] = p_prev
        data_prev_time["mass_density_prev"] = mass_density_prev
    # end if-else

    # # Wrap the constitutive law into ad, and construct the ad-wrapper of the PDE
    rho_ad = pp.ad.Function(rho, "density")
    rho_ad_main = rho_ad(p, temp_prev)

    # Build the flux term.
    # Remember that we need to multiply the fluxes by density.
    # The density is a cell-centerd variable,so we need to move it onto the faces
    # NOTE: the values in the darcy flux for this variable have to be +1
    # to ensure that the scaling becomes correct
    upwind_weight = pp.ad.UpwindAd(keyword=flow_kw, grids=grid_list)

    # The boundary condition for the temperature
    bc_temp = pp.ad.BoundaryCondition(keyword=temperature, grids=grid_list)

    # The boundary conditions
    rho_ad_bc = rho_ad(bound_flux, bc_temp)

    # Depending on grid type (e.g unstructured or Cartesian, 2- or 3D, etc.)
    # and type of boundary conditions (Dirichlet inflow and so on)
    # you might need to change the minus to pluss to get the pressure profile
    # as you want
    rho_on_face = (
        upwind_weight.upwind * rho_ad_main
        - upwind_weight.bound_transport_dir * rho_ad_bc
        - upwind_weight.bound_transport_neu * rho_ad_bc
    )

    # Ad wrapper of Tpfa discretization
    mpfa = pp.ad.TpfaAd(flow_kw, grids=grid_list)

    # The interior and boundary fluxes
    interior_flux = mpfa.flux * p
    boundary_flux = mpfa.bound_flux * bound_flux

    # The Darcy flux (in the higher dimension)
    full_flux = interior_flux + boundary_flux

    # Multiply the flux and the density
    full_density_flux = rho_on_face * full_flux

    # Add the fluxes from the interface to higher dimension,
    # the source accounting for fluxes to the lower dimansions
    # and multiply by the divergence
    if len(edge_list) > 0:

        # Include flux from the interface
        full_flux += mpfa.bound_flux * mortar_projection.mortar_to_primary_int * v

        # Tools to include the density in the source term
        upwind_coupling_weight = pp.ad.UpwindCouplingAd(
            keyword=flow_kw, edges=edge_list
        )
        trace = data_grid["trace_single"]

        # up_weight_flux = upwind_coupling_weight.flux
        up_weight_primary = upwind_coupling_weight.upwind_primary
        up_weight_secondary = upwind_coupling_weight.upwind_secondary

        # Map the density in the higher- and lower-dimensions
        # onto the interfaces
        high_to_low = (
            up_weight_primary
            * mortar_projection.primary_to_mortar_avg
            * trace.trace
            * rho_ad_main
        )

        low_to_high = (
            up_weight_secondary
            * mortar_projection.secondary_to_mortar_avg
            * rho_ad_main
        )

        # The density at the interface
        mortar_density = (high_to_low + low_to_high) * v

        # # Include the mortar density in the flux towards \Omega_h
        full_density_flux += (
            mpfa.bound_flux * mortar_projection.mortar_to_primary_int * mortar_density
        )

        # The source term for the flux to \Omega_l
        sources_from_mortar = mortar_projection.mortar_to_secondary_int * mortar_density

        conservation = div * full_density_flux - sources_from_mortar

    else:
        conservation = div * full_density_flux
    # end if

    # Construct the conservation equation
    density_wrap = (
        mass_density.mass * rho_ad_main
        - mass_density_prev.mass * rho_ad(p_prev, temp_prev)
        + dt * conservation
    )
    if len(edge_list) > 0:

        pressure_trace_from_high = mortar_projection.primary_to_mortar_avg * (
            mpfa.bound_pressure_cell * p
            + mpfa.bound_pressure_face
            * (mortar_projection.mortar_to_primary_int * v + bound_flux)
        )

        pressure_from_low = mortar_projection.secondary_to_mortar_avg * p

        robin = pp.ad.RobinCouplingAd(flow_kw, edge_list)

        # The interface flux,
        interface_flux = v + robin.mortar_discr * (
            pressure_trace_from_high - pressure_from_low
        )
    # end if

    # Make equations, feed them to the AD-machinery and discretize
    if len(edge_list) > 0:
        # Conservation equation
        density_wrap.discretize(gb)

        # Flux over the interface
        interface_flux.discretize(gb)
    else:
        density_wrap.discretize(gb)
    # end if-else

    # Store the flux
    data_prev_newton["AD_full_flux"] = full_flux

    if len(edge_list) > 0:
        data_prev_newton.update({"AD_edge_flux": v})
    # end if

    #%% The temperatue equtaion.

    # The construction is similar to the transport equation studied in detail below.
    # but is put above it to avoid having to redefine div, mortar projection etc

    # Upwind discretization for the convection
    upwind_temp = pp.ad.UpwindAd(keyword=temperature, grids=grid_list)
    heat_capacity = pp.ad.MassMatrixAd(keyword=temperature, grids=grid_list)
    if iterate:
        heat_capacity_prev = data_prev_time["heat_capacity_prev"]
        temp_prev = data_prev_time["temp_prev"]
    else:
        heat_capacity_prev = pp.ad.MassMatrixAd(keyword=temperature, grids=grid_list)

        # Store for later use
        data_prev_time["temp_prev"] = temp_prev
        data_prev_time["heat_capacity_prev"] = heat_capacity_prev
    # end if-else

    # Temperature "scaled" by density and specific heat capacites
    temp_density = (
        temp
        * constant_params.specific_heat_capacity_fluid()
        * rho_ad(p.previous_iteration(), temp.previous_iteration())
    )

    bound_temp = bc_temp * constant_params.specific_heat_capacity_fluid() * rho_ad_bc

    # Convection term
    convection = (
        (upwind_temp.upwind * temp_density) * full_flux
        - upwind_temp.bound_transport_dir * full_flux * bound_temp
        - upwind_temp.bound_transport_neu * bound_temp
    )

    # Conduction
    conduction_mpfa = pp.ad.TpfaAd(keyword=temperature, grids=grid_list)
    conduction = conduction_mpfa.flux * temp + conduction_mpfa.bound_flux * bc_temp

    # Cell-wise temperature equation
    temp_wrap = (
        heat_capacity.mass * temp - heat_capacity_prev.mass * temp_prev
    ) / dt + div * (convection + conduction)

    # Projections
    if len(edge_list) > 0:

        # Edge conduction
        temp_wrap += (
            div
            * conduction_mpfa.bound_flux
            * mortar_projection.mortar_to_primary_int
            * q
        )
        temp_wrap -= mortar_projection.mortar_to_secondary_int * q

        # Edge convection
        temp_wrap -= (
            div
            * upwind_temp.bound_transport_neu
            * mortar_projection.mortar_to_primary_int
            * w
        )

        temp_wrap -= mortar_projection.mortar_to_secondary_int * w
    # end if

    # convection over interface
    if len(edge_list) > 0:
        # Upstream convection
        convection_coupling = pp.ad.UpwindCouplingAd(
            keyword=temperature, edges=edge_list
        )
        high_to_low_convection = (
            convection_coupling.upwind_primary
            * mortar_projection.primary_to_mortar_avg
            * trace.trace
            * temp_density
        )

        low_to_high_convection = (
            convection_coupling.upwind_secondary
            * mortar_projection.secondary_to_mortar_avg
            * temp_density
        )

        interface_convection = (
            convection_coupling.mortar_discr * w
            - (high_to_low_convection + low_to_high_convection) * v
        )

        # Conductive flux
        temp_from_high = mortar_projection.primary_to_mortar_avg * (
            conduction_mpfa.bound_pressure_cell * temp
            + conduction_mpfa.bound_pressure_face
            * (mortar_projection.mortar_to_primary_int * q + bc_temp)
        )
        temp_from_low = mortar_projection.secondary_to_mortar_avg * temp

        robin_temp = pp.ad.RobinCouplingAd(keyword=temperature, edges=edge_list)

        conduction_over_interface = q + robin_temp.mortar_discr * (
            temp_from_high - temp_from_low
        )
    # end if

    # Discretize the equations
    if len(edge_list) > 0:
        temp_wrap.discretize(gb)
        interface_convection.discretize(gb)
        conduction_over_interface.discretize(gb)
    else:
        temp_wrap.discretize(gb)
    # end if-else

    #%% The Passive tracer

    # Upwind discretization for advection.
    upwind_tracer = pp.ad.UpwindAd(keyword=tracer, grids=grid_list)
    if len(edge_list) > 0:
        upwind_tracer_coupling = pp.ad.UpwindCouplingAd(keyword=tracer, edges=edge_list)
    # end if

    # Ad wrapper of boundary conditions
    bc_tracer = pp.ad.BoundaryCondition(keyword=tracer, grids=grid_list)

    if iterate:  # Newton-iteration
        mass_tracer = pp.ad.MassMatrixAd(tracer, grid_list)
        mass_tracer_prev = data_prev_time["mass_tracer_prev"]
        tracer_prev = data_prev_time["tracer_prev"]
    else:  # We are interested in "constructing" the equations

        # Mass matrix for accumulation
        mass_tracer = pp.ad.MassMatrixAd(tracer, grid_list)
        mass_tracer_prev = pp.ad.MassMatrixAd(tracer, grid_list)

        # The solution at time step n, i.e.
        # the one we need to use to fine the solution at time step n+1
        tracer_prev = passive_tracer.previous_timestep()

        # Store for Newton iterations
        data_prev_time["tracer_prev"] = tracer_prev
        data_prev_time["mass_tracer_prev"] = mass_tracer_prev
    # end if-else

    tracer_wrapper = (
        (mass_tracer.mass * passive_tracer - mass_tracer_prev.mass * tracer_prev) / dt
        # advection
        + div
        * (
            (upwind_tracer.upwind * passive_tracer) * full_flux
            - upwind_tracer.bound_transport_dir * full_flux * bc_tracer
            - upwind_tracer.bound_transport_neu * bc_tracer
        )
    )
    # Add the projections
    if len(edge_list) > 0:

        tracer_wrapper -= div * (
            upwind_tracer.bound_transport_neu
            * mortar_projection.mortar_to_primary_int
            * eta_tracer
        )

        tracer_wrapper -= mortar_projection.mortar_to_secondary_int * eta_tracer
    # end if

    # Tracer over the interface
    if len(edge_list) > 0:

        # Some tools we need
        upwind_tracer_coupling_primary = upwind_tracer_coupling.upwind_primary
        upwind_tracer_coupling_secondary = upwind_tracer_coupling.upwind_secondary

        # First project the concentration from high to low
        trace_of_tracer = trace.trace * passive_tracer

        high_to_low_tracer = (
            upwind_tracer_coupling_primary
            * mortar_projection.primary_to_mortar_avg
            * trace_of_tracer
        )

        # Next project concentration from lower onto higher dimension
        low_to_high_tracer = (
            upwind_tracer_coupling_secondary
            * mortar_projection.secondary_to_mortar_avg
            * passive_tracer
        )

        # Finally we have the transport over the interface equation
        tracer_over_interface_wrapper = (
            upwind_tracer_coupling.mortar_discr * eta_tracer
            - (high_to_low_tracer + low_to_high_tracer) * v
        )

    # end if

    if len(edge_list) > 0:
        tracer_wrapper.discretize(gb)
        tracer_over_interface_wrapper.discretize(gb)
    else:
        tracer_wrapper.discretize(gb)
    # end if-else

    #%% Next, the (solute) transport equations.

    # Upwind discretization for advection.
    upwind = pp.ad.UpwindAd(keyword=transport_kw, grids=grid_list)
    if len(edge_list) > 0:
        upwind_coupling = pp.ad.UpwindCouplingAd(keyword=transport_kw, edges=edge_list)
    # end if

    # Ad wrapper of boundary conditions
    bc_c = pp.ad.BoundaryCondition(keyword=transport_kw, grids=grid_list)

    # Divergence operator. Acts on fluxes of aquatic components only.
    div = data_grid["divergence_several"]

    # Mortar projection between subdomains and mortar grids
    if len(edge_list) > 0:
        mortar_projection = data_grid["mortar_projection_several"]
    # end if

    # The primary variable for concentrations is on log-form,
    # while the standard term is advected.
    # Conservation applies to the linear form of the total consentrations.
    # Make a function which carries out both conversion and summation over primary
    # and secondary species.
    def log_to_linear(c: pp.ad.Ad_array) -> pp.ad.Ad_array:
        return pp.ad.exp(c) + cell_S.T * cell_equil_comp * pp.ad.exp(cell_S * c)

    # Wrap function in an Ad function, ready to be parsed
    log2lin = pp.ad.Function(log_to_linear, "")

    # For the accumulation term, we need T and teh porosity at
    # the previous time step. These should be fixed in the Newton-iterations
    if iterate:  # Newton-iteration
        mass = pp.ad.MassMatrixAd(mass_kw, grid_list)
        mass_prev = data_prev_time["mass_prev"]
        T_prev = data_prev_time["T_prev"]
    else:  # We are interested in "constructing" the equations

        # Mass matrix for accumulation
        mass = pp.ad.MassMatrixAd(mass_kw, grid_list)
        mass_prev = pp.ad.MassMatrixAd(mass_kw, grid_list)

        # The solute solution at time step n, ie. the one we need to use to fine the solution at time step n+1
        T_prev = T.previous_timestep()

        # Store for Newton iterations
        data_prev_time["T_prev"] = T_prev
        data_prev_time["mass_prev"] = mass_prev
    # end if-else

    # This consists of terms for accumulation and advection, in addition to boundary conditions.
    # We need four terms for the solute transport equation:
    # 1) Accumulation
    # 2) Advection
    # 3) Boundary condition for inlet
    # 4) boundary condition for outlet.

    # NOTE: The upwind discretization matrices are based on the
    # signs of the fluxes, not the fluxes themselfs.
    # The fluxes are computed above, in the flow part as
    # full_flux = mpfa * p + bound mpfa * p_bound + ...
    # Need to expand the flux vector

    expanded_flux = pp.ad.Matrix(data_grid["extend_flux"]) * full_flux

    # breakpoint()
    all_2_aquatic = data_grid["all_2_aquatic"]
    log_c = log2lin(log_X)

    transport = (
        (mass.mass * T - mass_prev.mass * T_prev) / dt
        # advection
        + all_2_aquatic.transpose()
        * div
        * (
            (upwind.upwind * all_2_aquatic * log_c) * expanded_flux
            - upwind.bound_transport_dir * expanded_flux * bc_c
            - upwind.bound_transport_neu * bc_c
        )
    )
    # breakpoint()
    # Add the projections
    if len(edge_list) > 0:

        # The trace operator
        trace = data_grid["trace_several"]

        all_2_aquatic_at_interface = data_grid["all_2_aquatic_at_interface"]

        transport -= div * (
            upwind.bound_transport_neu
            * mortar_projection.mortar_to_primary_int
            * all_2_aquatic_at_interface.transpose()
            * eta  # Behaves like a "coupling factor" to ensure that  # the multiplication is ok
        )

        transport -= (
            mortar_projection.mortar_to_secondary_int
            * all_2_aquatic_at_interface.transpose()
            * eta
        )
    # end if

    #%% Transport over the interface
    if len(edge_list) > 0:

        # Some tools we need
        upwind_coupling_primary = upwind_coupling.upwind_primary
        upwind_coupling_secondary = upwind_coupling.upwind_secondary

        S = pp.ad.Matrix(data_grid["extend_edge_flux"])
        expanded_v = S * v
        # breakpoint()
        # First project the concentration from high to low
        # At the higher-dimensions, we have both fixed
        # and aqueous concentration. Only the aqueous
        # concentrations are transported
        trace_of_conc = trace.trace * log_c

        high_to_low_trans = (
            upwind_coupling_primary
            * all_2_aquatic_at_interface
            * mortar_projection.primary_to_mortar_avg
            * trace_of_conc
        )

        # Next project concentration from lower onto higher dimension
        # At the lower-dimension, we also have both
        # fixed and aqueous concentration
        low_to_high_trans = (
            upwind_coupling_secondary
            * all_2_aquatic_at_interface
            * mortar_projection.secondary_to_mortar_avg
            * log_c
        )

        # Finally we have the transport over the interface equation
        transport_over_interface = (
            upwind_coupling.mortar_discr * eta
            - (high_to_low_trans + low_to_high_trans) * expanded_v
        )
    # end if

    if len(edge_list) > 0:
        transport.discretize(gb)
        transport_over_interface.discretize(gb)
    else:
        transport.discretize(gb)
    # end if-else

    #%% The last equation is mineral precipitation-dissolution

    def F_matrix(x1, x2):
        """
        Construct diagonal matrices, representing active and inactive sets
        """
        f = x1 - x2 > 0
        f = f.astype(float)
        Fa = sps.diags(f)  # Active
        Fi = sps.diags(1 - f)  # Inactive
        return Fa, Fi

    def phi_min(mineral_conc, log_primary):
        """
        Evaluation of the minimum function

        The inputs becomes
        conc -> Ad_array (val, jac)
        conc.previous_iteration() -> np.ndarray.
        """
        sec_conc = 1 - cell_equil_prec * pp.ad.exp(cell_E * log_primary)
        Fa, Fi = F_matrix(mineral_conc.val, sec_conc.val)
        eq = Fa * sec_conc + Fi * mineral_conc
        return eq

    ad_min_1 = pp.ad.Function(phi_min, "")
    mineral_eq = ad_min_1(precipitate, log_X)

    #%% Porosity and aperture equations

    def sum_mat(n, m):
        """
        Create a matrix which upon multilpied with a vector,
        sums part of its entries
        """
        zz = []
        for i in range(n):
            x = np.zeros(m)
            x[2 * i : 2 * (i + 1)] = 1
            zz.append(x)
        # end i-loop

        S = sps.csr_matrix(zz)
        return S

    def porosity_res(porosity, minerals):
        "Residual equation for porosity"

        # Non-reactive fraction
        chi_nr = to_vec(gb, "reference", "non_reactive_volume_frac")

        # Molar masses
        mm1 = constant_params.molar_mass_CaCO3()
        mm2 = constant_params.molar_mass_CaSO4()
        # Densities
        d1 = constant_params.density_CaCO3()
        d2 = constant_params.density_CaSO4()

        mins = np.vstack(
            [mm1 / d1 * gb.cell_volumes(), mm2 / d2 * gb.cell_volumes()]
        ).ravel("F")

        S = data_grid["sum_mat"]
        # breakpoint()
        ptn = S * (minerals * mins) + chi_nr - np.ones(gb.num_cells())

        residual = porosity + ptn

        return residual

    porosity_eq = pp.ad.Function(porosity_res, "")
    porosity_eq = porosity_eq(porosity, precipitate)

    def dimensional_mat(dim):
        """
        An indicator matrix, based on dimension
        """

        if dim > 3 or dim < 0:
            raise ValueError("Not implemented for input dimension")

        # Adjust values according to dimension
        val = 0

        if dim < gb.dim_max():
            for g, _ in gb:
                if dim < g.dim:
                    val += g.num_cells
                # end if
            # end g-loop
        # end if

        grid = gb.grids_of_dimension(dim)

        x = np.zeros(gb.num_cells())

        for i in range(len(grid)):
            g = grid[i]
            indi = slice(val, val + g.num_cells)
            x[indi] = 1.0
            val += g.num_cells
        # en i-loop

        S = sps.diags(x)
        return S

    def one_to_zero_projection(gb):
        """Return a matrix transfering objects from 1D to 0D

        NOTE Currently only able to map one quantity from 1D to 0D,
        however extending to several quantities should be straightforward.
        It just requires some thinking of storage at each iteration and returning the projection matrices
        """
        num_parent = []
        p = []
        for g, d in gb:
            if g.dim < gb.dim_max() - 1:
                for edges in gb.edges_of_node(g):
                    e = edges[0]
                    gh = e[0]

                    if gh == g:
                        gh = e[1]
                    # end if

                    if gh.dim == gb.dim_max() - 1:
                        mg = gb.edge_props(e)["mortar_grid"]
                        # Projection operators
                        projection = (
                            mg.mortar_to_secondary_avg()
                            * mg.primary_to_mortar_avg()
                            * np.abs(gh.cell_faces)
                        )
                        p.append(projection)
                        num_parent.append(
                            np.sum(mg.mortar_to_secondary_int().A, axis=1)
                        )

        # end g,d-loop

        if len(p) > 0:
            SS = (sps.bmat([[m for m in p]])).tolil()
            temp_sum = np.sum(np.array(num_parent), axis=0)

            for i in range(len(temp_sum)):
                SS[i] /= temp_sum[i]
            # end i-loop
            SS = SS.tocsr()

        else:
            SS = sps.csr_matrix((0, 0))
        # end if-else
        return SS

    def proj(gb):
        """
        Mapping between a gb.num_cells size vector and fracture grids
        that have a lower-dimensional neighbour
        """

        cell_inds = 0
        M = np.zeros(gb.num_cells())

        for g, d in gb:
            # print(g.num_cells)
            inds = np.arange(cell_inds, cell_inds + g.num_cells)
            if g.dim == gb.dim_max() - 1:  # At a fracture
                for edge in gb.edges_of_node(g):
                    e = edge[0]
                    gl = e[1]
                    # If the fracture has a intersectional neighbour,
                    # put one in the mapping matrix
                    if gl.dim == gb.dim_max() - 2:
                        M[inds] = 1.0
                    # end if
                # end loop
            # end if

            cell_inds += g.num_cells

        nonzero = M.nonzero()[0]

        # Fill in
        if len(nonzero) > 0:
            S = sps.lil_matrix((nonzero.size, gb.num_cells()))
            for i in range(nonzero.size):
                S[i, nonzero[i]] = 1.0
            # end loop
            S = S.tocsr()
        else:
            S = sps.csr_matrix((0, gb.num_cells()))
        # end if-else

        return S

    def aperture_residual(a, minerals):
        """
        Calculate the aperture, using the precipitated from the previous iterate

        The inputs become:
            a -> pp.ad.Ad_array
            minerals -> np.ndarray
        """

        # The apertrue equation takes form
        # aperture_res = a - (
        #     S_2d * np.ones(gb.num_cells()) +
        #     S_1D * ... +
        #     S_0D * ...     )
        # where the ... are the frature and intersection apertures. Need to calulate them

        der_matrix = sps.diags(diagonals=np.zeros(gb.num_cells()), shape=a.jac.shape)

        mins = np.vstack(
            [
                constant_params.molar_mass_CaCO3()
                / constant_params.density_CaCO3()
                * gb.cell_volumes(),
                constant_params.molar_mass_CaSO4()
                / constant_params.density_CaSO4()
                * gb.cell_volumes(),
            ]
        ).ravel("F")

        surface_area = to_vec(gb, "reference", "surface_area")

        surface_area = repeat(surface_area, 2, to_ad=False)

        NN = one_to_zero_projection(gb)
        M = proj(gb)
        inter_aper = M * a

        # Mapping between data_grid[dim_0_mat], which is of size gb.num_cells x gb.num_cells
        # and NN * inter_aper, which has size intersection g.num_cells
        def intersection_and_gb(n, m):
            """
            Matrix that expands a vector with size intersectional grid num cells
            to a vector of size gb num_cells, with structure
            S * v = [0,
                     v]
            """

            S = sps.lil_matrix((m, n))
            rows = np.arange(0, n)
            cols = np.arange(m - n, m)
            S[cols, rows] = 1.0

            return S.tocsr()

        AA = intersection_and_gb(NN.shape[0], gb.num_cells())

        aperture_res = a - (
            data_grid["dim_2_mat"]
            * pp.ad.Ad_array(val=np.ones(gb.num_cells()), jac=der_matrix)
            + data_grid["dim_1_mat"]
            * (
                -data_grid["sum_mat"] * (minerals * mins / surface_area)
                + constant_params.open_aperture()
            )
            + data_grid["dim_0_mat"] * AA * NN * inter_aper
        )

        return aperture_res

    a = pp.ad.Function(aperture_residual, "")
    aperture_eq = a(aperture, precipitate)

    #%% Permeability

    def indicator(inds):
        "Construct an indicator matrix"
        I1 = sps.diags(inds).tocsr()
        I2 = sps.diags(1 - inds).tocsr()
        return I1, I2

    def mat_perm(phi):
        # reference perm and porosity are numpy arrays of size g.num_cells in 2d;
        # need to make replica for lower dimensional g.num_cells
        k0 = data_ref["permeability"]
        phi0 = data_ref["porosity"]

        k0 = data_grid["mat_to_omega"] * k0
        phi0 = data_grid["mat_to_omega"] * phi0
        # breakpoint()
        # In the Ad-framework, the Ad-variables must come first, to avoid
        # getting long list of Ad-arrays
        KC = (
            (1 - phi) ** (-2.0)
            * (phi) ** (3.0)
            * k0
            * np.power(phi0, -3)
            * np.power(1 - phi0, 2)
        )
        return KC

    def frac_perm(a):
        return a ** (2.0) / 12

    def gb_perm(K, phi, a):
        x = np.zeros(gb.num_cells())
        for g, _ in gb:
            if g.dim == gb.dim_max():
                x[0 : g.num_cells] = 1
        # end loop

        I1, I2 = indicator(x)

        K_mat = mat_perm(phi)
        K_frac = frac_perm(a)

        eq = K - (I1 * K_mat + I2 * K_frac)
        return eq

    perm_func = pp.ad.Function(gb_perm, "")
    kk = perm_func(perm, porosity, aperture)

    #%% Equilibirum constants

    def taylor(equil, temp):
        K0 = np.tile(constant_params.ref_equil(), gb.num_cells())
        c = np.tile(constant_params.standard_enthalpy(), gb.num_cells())
        tk = data_grid["temp_equil"] * temp
        dt = tk - constant_params.ref_temp()
        t = (
            1 + dt * c / constant_params.ref_temp() ** 2
        )  # First order Taylor approximation of van't Hoffs Equation

        e_const = equil - t * K0

        return e_const

    equilibrium_const = pp.ad.Function(taylor, "")
    equilibrium_const_eq = equilibrium_const(equilibrium_constants, temp)

    #%% The last step is to feed the equations to the equation manager and return the non-linear equations
    equation_manager.equations.clear()

    if len(edge_list) > 0:
        equation_manager.equations = {
            "fluid_conservation": density_wrap,
            "interface_flux": interface_flux,
            "solute_transport": transport,
            "tronsport_over_interface": transport_over_interface,
            "equilibrium": equilibrium_eq,
            "dissolution_and_precipitation": mineral_eq,
            "temperature": temp_wrap,
            "convetion": interface_convection,
            "conduction": conduction_over_interface,
            "passive_tracer": tracer_wrapper,
            "tracer_interface_flux": tracer_over_interface_wrapper,
            "porosity": porosity_eq,
            "aperture": aperture_eq,
            "permeability": kk,
            "equil_consts": equilibrium_const_eq,
        }
    else:
        equation_manager.equations = {
            "fluid_conservation": density_wrap,
            "solute_transport": transport,
            "equilibrium": equilibrium_eq,
            "dissolution_and_precipitation": mineral_eq,
            "temperature": temp_wrap,
            "passive_tracer": tracer_wrapper,
            "porosity": porosity_eq,
            "aperture": aperture_eq,
            "permeability": kk,
            "equil_consts": equilibrium_const_eq,
        }
    # end if-else

    return equation_manager
