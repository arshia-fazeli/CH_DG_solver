from netgen.geom2d import unit_square
from ngsolve import *
import random
import numpy
import netgen.gui
from helpers import Jump , grad ,gradavg
from solver import newton_solve, NonLinearSolve
from user_settings import initial_roughness, M, order, tau, tend, lamdba, vtkoutput, solver
from ic_bc import sqr




# add gaussians with random positions and widths until we reach total mass >= 0.5

def set_initial_conditions(result_gridfunc):
    c0 = GridFunction(result_gridfunc.space)
    total_mass = 0.0
    vec_storage = c0.vec.CreateVector()
    vec_storage[:] = 0.0

    print("setting initial conditions")
    while total_mass < 0.3:
        print("\rtotal mass = {:10.6e}".format(total_mass), end="")
        center_x = random.random()
        center_y = random.random()
        thinness_x = initial_roughness * (1+random.random())
        thinness_y = initial_roughness * (1+random.random())
        c0.Set(exp(-(sqr(thinness_x) * sqr(x-center_x) + sqr(thinness_y) * sqr(y-center_y))))
        vec_storage.data += c0.vec
        c0.vec.data = vec_storage

        # cut off above 1.0
        result_gridfunc.Set(IfPos(c0-1.0,1.0,c0))
        total_mass = Integrate(s.components[0],mesh,VOL)

    print()


mesh = Mesh(unit_square.GenerateMesh(maxh=0.05))

V = L2(mesh, order=order, dgjumps=True)
fes = FESpace([V, V])
c, mu = fes.TrialFunction()
q, v = fes.TestFunction()
n= specialcf.normal(2)
h = specialcf.mesh_size

alpha = 4*order**2/h

s = GridFunction(fes)
sold = GridFunction(fes)
siter = GridFunction(fes)
set_initial_conditions(s.components[0])
sold.vec.data = s.vec.data
siter.vec.data = s.vec.data



#alternative: use random number generator pointwise:
#s.components[0].Set(RandomCF(0.0,1.0))
s.components[1].Set(CoefficientFunction(0.0))


Draw(s.components[1], mesh, "mu")
Draw(s.components[0], mesh, "c")

if vtkoutput:
    vtk = VTKOutput(ma=mesh,coefs=[s.components[1],s.components[0]],names=["mu","c"],filename="cahnhilliard_",subdivision=3)
    vtk.Do()

# implicit Euler
t = 0.0
while t < tend:
    print("\n\nt = {:10.6e}".format(t))

    sold.vec.data = s.vec.data

    s, converged, iter_counter, non_linear_solve_error = \
    NonLinearSolve(fes, mesh, s, siter, sold, 100, [c, mu], [q, v])

    t += tau
    Redraw(blocking=False)
    if vtkoutput:
        vtk.Do()