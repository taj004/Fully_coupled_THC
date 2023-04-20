"""
Solve a system of non-linear equations on a mixed-dimensional grid,
using Newton's method and the AD-framework in PorePy

@author: uw
"""

import numpy as np
import scipy.sparse as sps
import porepy as pp

#import pypardiso
import equations
import update_param

def update_darcy(dof_manager):
    """ Update the darcy flux in the parameter dictionaries 
    
    """
    
    gb = dof_manager.gb
    
    # Get the Ad-fluxes
    gb_2d = gb.grids_of_dimension(gb.dim_max())[0]
    data = gb.node_props(gb_2d)
    full_flux = data[pp.PARAMETERS]["previous_newton_iteration"]["AD_full_flux"]
    
    # Convert to numerical values
    num_flux = full_flux.evaluate(dof_manager)
    if hasattr(num_flux, "val"):
        num_flux = num_flux.val
    # end if
    
    # Remove contribution from fracture faces
    val = 0
    for g,d in gb:
        fracture_faces = g.tags["fracture_faces"]
        is_fracture_faces = np.where(fracture_faces==True)[0]
        corr_frac_faces = is_fracture_faces + val
        if len(is_fracture_faces)>0:
            num_flux[corr_frac_faces]=0
        # end if
        val += g.num_faces
    # end loop
    
    # Get the signs
    sign_flux = np.sign(num_flux)
    
    # Finally, loop over the gb and return the signs of the darcy fluxes
    val = 0
    for g,d in gb:
            
        inds = slice(val, val + g.num_faces) 
        #print(num_flux[inds])
        d[pp.PARAMETERS]["transport"]["darcy_flux"] = sign_flux[inds].copy()
        d[pp.PARAMETERS]["flow"]["darcy_flux"] = np.abs(sign_flux[inds].copy())
        d[pp.PARAMETERS]["passive_tracer"]["darcy_flux"] = sign_flux[inds].copy() 
            
        val += g.num_faces
    # end g,d-loop
    
    # Do the same over the interfaces 
    if gb.dim_max() > gb.dim_min():
        
        # The flux
        edge_flux = data[pp.PARAMETERS]["previous_newton_iteration"]["AD_lam_flux"]
        num_edge_flux = edge_flux.evaluate(dof_manager).val
        sign_edge_flux = np.sign(num_edge_flux) 
        
        val = 0
        for e,d in gb.edges():
            inds = slice(val, val + d["mortar_grid"].num_cells)
            d[pp.PARAMETERS]["transport"]["darcy_flux"] = sign_edge_flux[inds].copy()
            d[pp.PARAMETERS]["flow"]["darcy_flux"] = sign_edge_flux[inds].copy()
            d[pp.PARAMETERS]["passive_tracer"]["darcy_flux"] = sign_edge_flux[inds].copy()
            val += d["mortar_grid"].num_cells
        # end e,d-loop
    
    return

def clip_variable(x, dof_manager, target_name, min_val, max_val):
    """ Helper method to cut the values of a target variable x.
    Intended use is the concentration variable.
    """
    dof_ind = np.cumsum(np.hstack((0, dof_manager.full_dof)))
    
    for key, val in dof_manager.block_dof.items():
        if key[1] == target_name:
            inds = slice(dof_ind[val], dof_ind[val + 1])
            x[inds] = np.clip(x[inds], a_min=min_val, a_max=max_val)
        # end if
    # end key,val-loop
  
    return x


def backtrack(equation, dof_manager, 
              grad_f, p_k, x_k, f_0, 
              maxiter=10, min_tol=1e-7):
    """ Compute a stp size, using Armijo interpolation backtracing
    """
    # Initialize variables
    c_1 = 1e-4
    alpha = 1.0 # initial step size
    dot = grad_f.dot(p_k)
    
    # The function to be minimized    
    def phi(alpha):
        dof_manager.distribute_variable(x_k + alpha * p_k, to_iterate=True)
        F_new = equation.assemble_rhs()
        phi_step = 0.5*F_new.dot(F_new)
        return phi_step
    
    # Variables
    f_k = f_0 
    alpha_prev = 1.0 # Initial step size
    phi_old = phi(alpha) 
    phi_new = phi_old.copy()
    
    for i in range(maxiter):
        
        # If the first Wolfe condition is satisfied, we stop
        f_new = phi_new.copy()
        if f_new < f_k + alpha*c_1*dot:
            break
        # end if
    
        # Upper and lower bounds
        u = 0.5 * alpha
        l = 0.1 * alpha
        
        # Compute the new step size
        if i == 0: # remember that we have not updated the iteration index yet, 
                   # hence we use the index one lower than what we expect            
            
            # The new step size                                     
            denominator = 2 * (phi_old - f_k - dot)
            alpha_temp = -dot/denominator
            
        else: 
        
            # The matrix-vector multiplication. 
            mat = np.array([
                [ 1/alpha**2         , -1/alpha**2],
                [-alpha_prev/alpha**2,  alpha/alpha_prev**2 ] 
                ])
            
            vec = np.array([phi_new - f_k - dot*alpha,
                            phi_old - f_k - dot*alpha_prev])
            
            denominator = alpha - alpha_prev 
        
            a,b = (1/denominator) * np.matmul(mat, vec)
        
            if np.abs(a) < 1e-3: # cubic interpolation becomes quadratic interpolation
                alpha_temp = -dot/(2*b)
            else:
                alpha_temp = (-b + np.sqrt(np.abs(b**2 - 3*a*dot))) / (3*a)
            # end if-else
       # end if-else    
       
        # Check if the new step size is to big
        # From a safty point of view, this helps if alpha_temp is inf.
        # Is it ok to use if alpha_temp is nan?
        alpha_temp = min(alpha_temp, u)
   
        # Update the values, while ensuring that step size is not too small
        alpha_prev = alpha
        phi_old = phi_new  
        alpha = max(alpha_temp, l)
        phi_new = phi(alpha) 
        
        # Check if norm(alpha*p_k) is small. Stop if yes.
        # In such a case we might expect convergence
        if  np.linalg.norm(alpha*p_k) < min_tol:
           # print(f"min_tol convergce, with step size {alpha}")
            break
        # end if
    # end i-loop
        
    return 
    
def newton_gb(gb: pp.GridBucket, 
              equation: pp.ad.EquationManager,
              dof_manager: pp.DofManager, 
              clip_low_and_up: np.array = np.array([1e-100, 1e100])):
    """
    Newton's method applied to an equation, pulling values and state from gb.
    
    Parameters
    ----------
    equation, The equation we want to solve
    dof_manager, the associated dof_manager
    clip_low_and_up, numpy array. the upper and lower bound for clipping. 
          The form is np.array([low, up]). The values are interpreted as log values. 
          E.g. np.array([-30,25]), where -30 and 25 is interpreted as
              -30=log(x1), 25=log(x2)     
    
    Returns
    ----------
    conv, bool, whether Newton converged within a tolerance and maximum number of iterations
    i, int, the number of iterations used
    """
    
    J, resid = equation.assemble()
   
    # The upper and lower bound
    min_val = clip_low_and_up[0]
    max_val = clip_low_and_up[1]
      
    conv = False
    i = 0
    maxit = 50
    
    norm_orig = np.linalg.norm(resid)

    print(norm_orig)
    while conv is False and i < maxit:
   
        # -------------------------------- #
        
        # The Newton step
        
        # Compute the search direction
        grad_f = J.T.dot(-resid)

        dx = sps.linalg.spsolve(J, resid, use_umfpack=False) 
        # dx = pypardiso.spsolve(J, resid)
        # For refinement level 5 PyPardiso was emplyed for the linear system. 
        # For installation, see e.g. https://github.com/haasad/PyPardisoProject
        
        # Solution from prevous iteration step
        x_prev = dof_manager.assemble_variable(from_iterate=True)
        f_0 = 0.5 * resid.dot(resid)
        
        # Step size
        backtrack(equation, dof_manager, grad_f, dx, x_prev, f_0)
        
        # New solution
        x_new = dof_manager.assemble_variable(from_iterate=True)
        x = clip_variable(x_new.copy(), 
                          dof_manager, "log_X", 
                          min_val, max_val) 

        
        dof_manager.distribute_variable(x.copy(), to_iterate=True)
      
        # ------------------------- #
        # Update parameters
        
        # Update the Darcy flux in the parameter dictionries
        update_darcy(dof_manager)   
        
        # Update concentrations in the dictionary
        update_param.update_concentrations(gb, dof_manager, to_iterate=True)
        
        # Update the grid parameters
        update_param.update(gb)
        
        # ------------------------------- #
        # The updated equations
        
        # Increase number of steps
        i += 1
        
        # Update the equations for the next Newton iteration        
        equation = equations.gather(gb, 
                                    dof_manager=dof_manager,
                                    equation_manager=equation,
                                    iterate=True
                                    )
        
        # New Jacobian and residual
        J, resid = equation.assemble()
        
        # Measure the error
        norm_now = np.linalg.norm(resid)
        err_dist = np.linalg.norm(dx) / np.linalg.norm(x_new)
        
        # Stop if converged. 
        if norm_now < norm_orig * 1e-7 or norm_now < 1e-6 or err_dist < 1e-9:
            print("Solution reached")
            conv = True
        # end if
            
    # end while
        
    # Print some information
    print(f"Number of Newton iterations {i}")
    print(f"Residual reduction {norm_now / norm_orig}")
    
    
    # Return the number of Newton iterations; 
    # we use it to adjust the time step
    return conv, i 
    
# end newton
