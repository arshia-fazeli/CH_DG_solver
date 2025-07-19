import logging
import math
from ngsolve import *
import numpy as np
from weakform import weak_form_oseen

new_line = "\n"

def ref_element_vertices_val(gfu, vertices, gfu_index, element_type):
    val = []

    if element_type == "TRIG":
        # coefficients of the linear equation
        a = gfu.vec[gfu_index]
        b = gfu.vec[gfu_index + 1]
        c = gfu.vec[gfu_index + 2]
        for j in range(len(vertices)):
            val.append((a - b - c) + (3 * b + c) * vertices[j][0] + (2 * c) * vertices[j][1])

    if element_type == "TET":
        a = gfu.vec[gfu_index]
        b = gfu.vec[gfu_index + 1]
        c = gfu.vec[gfu_index + 2]
        d = gfu.vec[gfu_index + 3]
        for j in range(len(vertices)):
            eq = ((a - b - 2 * c - 4 * d) + (4 * b + 2 * c + 4 * d) * vertices[j][0] + (6 * c + 4 * d) * vertices[j][1] + 8 * d * vertices[j][2])
            val.append(eq)

    return val


def vertices_gfu_val(gfu, mesh, element_type="TRIG"):
    """
    Evaluates gfu at the vertices of each mesh cell
    Args:
        gfu:  GridFunction
        mesh: ngsolve mesh

    Returns:
        gfu_vertices_val : each row includes the values of gfu at different
        vertices of a triangular mesh

    """
    dof_per_element = int(len(gfu.vec) / mesh.ne)  # dof per element
    # initial array for the values of gfu at different vertices
    gfu_vertices_val = np.zeros((mesh.ne, dof_per_element))

    # coordinates of the vertices of the reference triangle
    if element_type == "TRIG":
        vertices = [(0, 0), (1, 0), (0, 1)]

    # coordinates of the vertices of the reference tetrahedron
    if element_type == "TET":
        vertices = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)]

    for i in range(mesh.ne):
        # index pointer for gfu
        gfu_index = i * dof_per_element
        gfu_vertices_val[i, :] = ref_element_vertices_val(gfu, vertices, gfu_index, element_type)

    return gfu_vertices_val

def bound_limiter_Joshaghani(gfu, mesh, bounds):
    '''
    Evaluates the GridFunction at the quadrature points and scales the grid function within each element if the max/min
    value at the mesh cell violates the upper/lower bound.

    Args:
        gfu: GridFunction
        mesh: ngsolve.comp.Mesh
        bounds: tuple (lower bound, upper bound)

    Returns:
        None
    '''

    (r1, r2) = bounds # lower and upper bounds
    averaged_val_gfu = gfu.vec.FV().NumPy() # cell averages
    number_of_elements = mesh.ne  # number of mesh elements
    dof_per_element = int(len(averaged_val_gfu) / number_of_elements)  # dofs per element
    quad_val_gfu = vertices_gfu_val(gfu,mesh) # gf at quad points

    quad_min_val = quad_val_gfu.min(axis=1)
    quad_max_val = quad_val_gfu.max(axis=1)

    theta = np.ones(number_of_elements, dtype=float) # scaling coefficients

    # iterate over the elements
    for i in range(number_of_elements):
        nn = dof_per_element * i # index of the cell averaged value
        if gfu.vec[nn] < r1:
            gfu.vec[nn] = r1
        if gfu.vec[nn] > r2:
            gfu.vec[nn] = r2
        if (quad_min_val[i] < r1) :
            theta[i] = (gfu.vec[nn]-r1)/(gfu.vec[nn]-quad_min_val[i])
        if (quad_max_val[i] > r2):
            theta2 = (gfu.vec[nn]-r2)/(gfu.vec[nn]-quad_max_val[i])
            theta[i] = min(theta[i], theta2)
        if theta[i] < 1 :
            for k in range(1, dof_per_element):
                gfu.vec[nn+k] = theta[i]*gfu.vec[nn+k]


def bound_limiter_Joshaghani_new(gfu, mesh, bounds):
    '''
    Evaluates the GridFunction at the quadrature points and scales the grid function within each element if the max/min
    value at the mesh cell violates the upper/lower bound.

    Args:
        gfu: GridFunction
        mesh: ngsolve.comp.Mesh
        bounds: tuple (lower bound, upper bound)

    Returns:
        None
    '''

    (r1, r2) = bounds  # lower and upper bounds

    averaged_val_gfu = gfu.vec.FV().NumPy()  # cell averages
    number_of_elements = mesh.ne  # number of mesh elements
    dof_per_element = int(len(averaged_val_gfu) / number_of_elements)  # dofs per element
    quad_val_gfu = vertices_gfu_val(gfu, mesh, element_type="TRIG")
    quad_min_val = quad_val_gfu.min(axis=1)
    quad_max_val = quad_val_gfu.max(axis=1)

    theta = np.ones((number_of_elements, dof_per_element), dtype=float)  # scaling coefficients

    # iterate over the elements
    for i in range(number_of_elements):
        nn = dof_per_element * i  # index of the cell averaged value
        if i == 18:
           pass
        # if the average violates the bounds, change the average
        if gfu.vec[nn] < r1:
            gfu.vec[nn] = r1
        elif gfu.vec[nn] > r2:
            gfu.vec[nn] = r2

        for j in range(dof_per_element):
            if (quad_val_gfu[i, j] < r1):
                theta[i, j] = (gfu.vec[nn] - r1) / (gfu.vec[nn] - quad_val_gfu[i, j])
                if j == 0:
                    theta2 = (gfu.vec[nn]) / (gfu.vec[nn + 1] + gfu.vec[nn + 2])
                    gfu.vec[nn + 1] *= theta2
                    gfu.vec[nn + 2] *= theta2
                if j == 2:
                    gfu.vec[nn + 1] = (3 * theta[i, j] * gfu.vec[nn + 1] + gfu.vec[nn + 1] * (theta[i, j] - 1)) / 3

                if j == 1:
                    gfu.vec[nn + 2] *= theta[i, j]
                    gfu.vec[nn + 1] = (3 * gfu.vec[1] + gfu.vec[2] * (1 - theta[i, j])) / 3

            if quad_val_gfu[i, j] > r2:
                theta2 = (gfu.vec[nn] - r2) / (gfu.vec[nn] - quad_val_gfu[i, j])
                theta[i, j] = min(theta2, theta[i, j])
                if j == 0:
                    theta2 = (gfu.vec[nn] - 1) / (gfu.vec[nn + 1] + gfu.vec[nn + 2])
                    gfu.vec[nn + 1] *= theta2
                    gfu.vec[nn + 2] *= theta2
                if j == 2:
                    gfu.vec[nn + 1] = (3 * theta[i, j] * gfu.vec[nn + 1] + gfu.vec[nn + 1] * (theta[i, j] - 1)) / 3

                if j == 1:
                    gfu.vec[nn + 2] *= theta[i, j]
                    gfu.vec[nn + 1] = (3 * gfu.vec[1] + gfu.vec[2] * (1 - theta[i, j])) / 3
        pass
def newton_solve(s, mesh, rhs, b, sold, a , As, mstar, w):
    rhs.data = b.mat * sold.vec
    rhs.data -= b.mat * s.vec
    a.Apply(s.vec, As)
    rhs.data -= As
    a.AssembleLinearization(s.vec)

    mstar.AsVector().data = b.mat.AsVector() + a.mat.AsVector()
    invmat = mstar.Inverse()
    w.data = invmat * rhs
    s.vec.data += w
    #bound_limiter_Joshaghani_new(s.components[0], mesh, (0, 1))
    wnorm = w.Norm()
    print("|w| = {:7.3e} ".format(wnorm), end="")
    return wnorm

def error(A, B, mesh):
    return sqrt(Integrate((A - B) ** 2, mesh))


def update_iterate(UN,UIter):
    UIter.vec.data = UN.vec.data

def timestep(UN,UOld):
    UOld.vec.data = UN.vec.data

def non_linear_iteration_L2_error (mesh, UN, UIter):
    '''
    Returns the L2 error between the current and previous iteration solutions.
    Args:
        mesh (ngsolve.comp.Mesh): ngsolve mesh
        UN (ngsolve.comp.GridFunction) : current iteration solution space
        UIter (ngsolve.comp.GridFunction) : previous iteration solution space

    Returns:
        list_of_errors (list) : Includes list of errors for each variable
        [error_uc , error_ud , error_p, error_alpha]
    '''
    list_of_errors = []
    for i in range(2):
        error_component = error(UN.components[i], UIter.components[i], mesh)
        list_of_errors.append(error_component)
    return list_of_errors

def logit(message):
    logging.info(message)
    print(message)
def log_errors_LinearSolve (iter_counter, error_c, error_mu):
    '''
    Logging the information about the error for each non-linear iteration.
    Args:
        iter_counter (int) : iteration counter
        error_uc (float) : error for velocity of the continious phase
        error_ud (float) : error for velocity of the dispersed phase
        error_p (float) : error for the pressure
        error_alpha (float) : error for the volume fraction
    Returns:
        None
    '''
    new_line = "\n"
    logit(f"{new_line}After {iter_counter} non-linear iterations, non-linear error report:"
          f"{new_line}Continuous velocity error: {error_c:.2e}"
          f"{new_line}Dispersed velocity error: {error_mu:.2e}")




def LinearSolve (BL, L, mesh, UN, UIter, iter_counter, maxits=100):
    '''
    Performs a linear iteration and returns the solution space for the current and previous iteration.
    It returns the error for uc, ud, p and alpha_c compared to the previous iterate.

    Args:
        BL (ngsolve.comp.BilinearForm) : bilinear form
        L  (ngsolve.comp.LinearForm) : linear form
        mesh (ngsolve.comp.Mesh): ngsolve mesh
        UN (ngsolve.comp.GridFunction) : current iteration solution space
        UIter (ngsolve.comp.GridFunction) : previous iteration solution space
        c (ngsolve.comp.Preconditioner) :  Preconditioner
        solution_method (str) : Solution method
        maxits (int): maximum iteration number
    ->
    Returns:
        (UN , curr_error_uc, curr_error_ud, curr_error_p, curr_error_alpha)
    '''

    r = L.vec.CreateVector()
    r.data = L.vec
    #mstar = BL.mat.CreateMatrix()
    #mstar.AsVector().data = BL.mat.AsVector() + BL_dt.mat.AsVector()
    r.data -= BL.mat * UN.vec
    inv = BL.mat.Inverse(UN.space.FreeDofs(BL.condense), inverse="umfpack")
    UN.vec.data += inv * r

    # Calculate the absolute error between the current solution and the previous iterate
    [error_c, error_mu] = non_linear_iteration_L2_error (mesh, UN, UIter)
    iteration_L2_error_list = [error_c, error_mu]
    #bound_limiter_Joshaghani(UN.components[0], mesh, (0, 1))
    # logging the errors
    log_errors_LinearSolve(iter_counter, error_c, error_mu)



    return UN, iteration_L2_error_list


def NonLinearSolve(fes, mesh, UN, UIter, UOld, maxits, trial_list, test_list):
    """
    solves a single time step.
    Args:
        X (ngsolve.comp.FESpace) : Mixed finite element space for trial and weighting functions
        mesh (ngsolve.comp.Mesh): ngsolve mesh
        UN (ngsolve.comp.GridFunction) : current iteration solution space
        UIter (ngsolve.comp.GridFunction) : previous iteration solution space
        UOld (ngsolve.comp.GridFunction) : Previous time-step solution space
        UClosure (ngsolve.comp.GridFunction) : current iteration solution space for closures
        t (ngsolve.fem.Parameter) : current time
        dt (ngsolve.fem.Parameter): time-step
        maxits (int): maximum iteration number
        solve_val (int): solve mode, 1 for full_timestep, 2 for first half and 3 for second half timestep
    Returns:
        UN, converged, iter_counter, non_linear_solve_error

    """

    list_of_errors = []

    # indicator parameter for the convergence status
    converged = False
    diverge = False
    instability_indicator = 0

    # Set the iteration counter
    iter_counter = 0

    # perform an iteration
    bilinear, linear = weak_form_oseen(fes, UOld, UIter, trial_list, test_list)
    SetHeapSize(100000000)
    UN, iteration_error_list = \
        LinearSolve (bilinear, linear, mesh,  UN, UIter, iter_counter, maxits)
    list_of_errors.append(iteration_error_list)


    # update the previous iterate variables
    UIter.vec.data = UN.vec.data

    #   Inner (non-linear) loop
    while iter_counter < maxits and converged == False and diverge == False:

        # update the iteration counter
        iter_counter += 1

        # perform an iteration
        bilinear, linear = weak_form_oseen(fes, UOld, UIter, trial_list, test_list)
        SetHeapSize(100000000)
        UN, iteration_error_list = \
            LinearSolve (bilinear, linear, mesh,  UN, UIter, iter_counter, maxits)
        list_of_errors.append(iteration_error_list)

        if list_of_errors[iter_counter][1]/list_of_errors[iter_counter-1][1] > 1 or list_of_errors[iter_counter][0]/list_of_errors[iter_counter-1][0]>1:
            instability_indicator += 1

        # update the previous iterate variables
        UIter.vec.data = UN.vec.data

        if math.isnan(iteration_error_list[0]) or math.isnan(iteration_error_list[1]):
            diverge = True
        if instability_indicator > 4:
            print("the error is increasing")
            diverge = True

        # Check all errors (uc, ud, alpha) are less than the inner_error tolerance
        if max(iteration_error_list[0], iteration_error_list[1]) < 1e-8:
            # Update convergence to True
            converged = True
            non_linear_solve_error = iteration_error_list
            return UN, converged, iter_counter, non_linear_solve_error


    # Update convergence to True
    converged = False
    non_linear_solve_error = iteration_error_list
    return UN, converged, iter_counter, non_linear_solve_error


def SolveBVP_fixed_timestep(mesh, t, dt, X, maxits ,h, CL, time_stepping_method):
    """
    A function to solve the transient boundary value problem
    """
    # initialize the time-step counter
    step = 0
    # create solution spaces
    UN = GridFunction(X)
    UIter = GridFunction(X)
    UOld = GridFunction(X)
    # retrieve the initial conditions
    uc_ic, ud_ic, alpha_c_ic, p_init = IC()
    # apply the initial conditions for new simulations
    if restart == 0:
        set_initial_conditions(UN.components[0])
        UN.components[1].Set(CoefficientFunction(0.0))
        update_iterate(UN)
    # if a restart is requested, set the previous solution to the last .sol file
    elif restart == 1:
        list_of_files = glob.glob('solution/sol/*')
        restart_file = max(list_of_files, key=os.path.getctime)
        UOld.Load(restart_file)
        UN.vec.data = UOld.vec.data
        UOldOld.vec.data = UOld.vec.data
        update_iterates(X, UN, UIter, UClosure)

    SetNumThreads(16)
    # Initialize "task manager" -- NGSolves native threading routine
    with TaskManager():

        # while the current time is less than the final time, keep solving
        while t.Get() < t_final:

            #  Print to output the time step, overall time, and current time step
            logit(f'{new_line}Starting solve at time {t.Get():.2e}, using time step size {dt.Get():.2e}')

            # start recording the cpu time for each time step
            start_time = time.process_time()
            # for the first time-step, we have to use first-order implicit solve
            if step == 0:
                    (UN, converged, iter_counter, non_linear_solve_error) = \
                        NonLinearSolve(X, mesh, UN, UIter, UOld, UClosure, t, dt, maxits, time_stepping_method="implicit",
                                       solve_val=1)
                    [curr_error_uc, curr_error_ud, curr_error_p, curr_error_alpha] = non_linear_solve_error
            else:
                (UN, converged, iter_counter, non_linear_solve_error) = \
                    NonLinearSolve(X, mesh, UN, UIter, UOld, UClosure, t, dt, maxits, time_stepping_method,
                                    solve_val=1)
                [curr_error_uc, curr_error_ud, curr_error_p, curr_error_alpha] = non_linear_solve_error



            # a variable for the total number of iterations
            total_iteration_number = iter_counter

            # If const dt is used, and solution failed to converge, break out
            if converged == False:
                # record the process time
                time_step_cputime = time.process_time() - start_time
                # Break out of the while loop, and print the failure
                logit(f'{new_line}Solve rejected for non-linear non-convergence. '
                      f'Exiting.')
                write_csv(
                    [str(step), str(dt.Get()), str(t.Get()), 'rejected', str(total_iteration_number), str(curr_error_uc),
                     str(curr_error_ud), str(curr_error_p), str(curr_error_alpha), 'NA',
                     str(cfl_time(UN.components[0], UN.components[1], mesh, h)),
                     str(cfl_lillia(UN.components[0], UN.components[1], mesh, h, polynomial_order)),'NA', str(time_step_cputime)])
                break

            else:  #converged == True
                # record the process time
                time_step_cputime = time.process_time() - start_time
                # advance the time-step by dt
                t.Set(t.Get() + dt.Get())
                # calculate the error in conservation of mass of the mixture
                uc_incompressibility = UN.components[3] * div(UN.components[0]) + UN.components[0] * grad(
                    UN.components[3])
                ud_incompressibility = div(UN.components[1]) - UN.components[3] * div(UN.components[1]) - \
                                       UN.components[1] * grad(UN.components[3])
                mixture_CoM_error = Integrate(uc_incompressibility + ud_incompressibility, mesh)
                # Accept the solution:
                write_csv(
                    [str(step), str(dt.Get()), str(t.Get()), 'accepted', str(total_iteration_number),
                     str(curr_error_uc),
                     str(curr_error_ud), str(curr_error_p), str(curr_error_alpha), str(mixture_CoM_error),
                     str(cfl_time(UN.components[0], UN.components[1], mesh, h)),
                     str(cfl_lillia(UN.components[0], UN.components[1], mesh, h, polynomial_order)), 'NA', str(time_step_cputime)])

                logit(f'{new_line}Solve accepted.')
                logit(f'{new_line}solved dt = {dt.Get():.2e}'
                      f'{new_line}cfl dt = {cfl_time(UN.components[0], UN.components[1], mesh, h):.2e}'
                      f'{new_line}cfl lillia dt = {cfl_lillia(UN.components[0], UN.components[1], mesh, h, polynomial_order):.2e}')
                # Update all variables:
                update_timestep (UN ,UOld, UOldOld)

                #   Save to file
                rounder = "{:.8f}".format(round(t.Get(), 8))
                filename = "solution/sol/result_" + rounder + ".sol"
                filename2 = "solution/closure_sol/result_" + rounder + ".sol"

                if step % save_every == 0:
                    UN.Save(filename)
                    UClosure.Save(filename2)
                step += 1
