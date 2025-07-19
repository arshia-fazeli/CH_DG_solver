from netgen.geom2d import unit_square
from ngsolve import *
import random
import numpy as np
import netgen.gui
from helpers import Jump , grad ,gradavg
from solver import newton_solve
from user_settings import initial_roughness, M, order, tau, tend, lamdba, vtkoutput, solver



def sqr(x):
    return x * x

def weak_form_nonlinear():
    a = BilinearForm(fes)
    a += tau * M * grad(mu) * grad(q) * dx
    a += -tau * M * n * gradavg(mu) * Jump(q) * dx(skeleton=True)
    a += -tau * M * n * gradavg(q) * Jump(mu) * dx(skeleton=True)
    a += tau * M * alpha * Jump(q) * Jump(mu) * dx(skeleton=True)
    a += mu * v * dx
    a += -200 * (c - 3 * c ** 2 + 2 * c ** 3) * v * dx
    a += -lamdba * grad(c) * grad(v) * dx
    a += lamdba * n * gradavg(c) * Jump(v) * dx(skeleton=True)
    a += lamdba * n * gradavg(v) * Jump(c) * dx(skeleton=True)
    a += -lamdba * alpha * Jump(c) * Jump(v) * dx(skeleton=True)

    
    #convection term
    vel = CoefficientFunction((2,2))
    a += tau * -c * vel * grad(q) * dx
    a += tau * c * IfPos(vel * n,vel * n , 0) * Jump(q) * dx(skeleton=True)
    a += tau * c * IfPos(vel * n, 0, vel * n) * Jump(q) * dx(skeleton=True)

    b = BilinearForm(fes)
    b += SymbolicBFI(c * q)

    b.Assemble()
    return a, b

# add gaussians with random positions and widths until we reach total mass >= 0.5

def set_initial_conditions(result_gridfunc):
    c0 = GridFunction(result_gridfunc.space)
    total_mass = 0.0
    vec_storage = c0.vec.CreateVector()
    vec_storage[:] = 0.0

    print("setting initial conditions")
    while total_mass < 0.5:
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


mesh = Mesh(unit_square.GenerateMesh(maxh=0.04))

V = L2(mesh, order=order, dgjumps=True)
fes = FESpace([V, V])
c, mu = fes.TrialFunction()
q, v = fes.TestFunction()
n= specialcf.normal(2)
h = specialcf.mesh_size

alpha = 4*order**2/h

s = GridFunction(fes)
sold = GridFunction(fes)
set_initial_conditions(s.components[0])
sold.vec.data = s.vec.data
s.Load('IC')
a , b = weak_form_nonlinear()
mstar = b.mat.CreateMatrix()


#alternative: use random number generator pointwise:
#s.components[0].Set(RandomCF(0.0,1.0))
s.components[1].Set(CoefficientFunction(0.0))

rhs = s.vec.CreateVector()
As = s.vec.CreateVector()
w = s.vec.CreateVector()

Draw(s.components[1], mesh, "mu")
Draw(s.components[0], mesh, "c")

if vtkoutput:
    vtk = VTKOutput(ma=mesh,coefs=[s.components[1],s.components[0]],names=["mu","c"],filename="solution/cahnhilliard_",subdivision=3)
    vtk.Do()

# implicit Euler
t = 0.0
while t < tend:
    print("\n\nt = {:10.6e}".format(t))

    sold.vec.data = s.vec.data
    wnorm = 1e99

    if solver == "Newton":
        # newton solver
        while wnorm > 20:
            wnorm = newton_solve(s, mesh, rhs, b, sold, a , As, mstar, w)
    t += tau
    Redraw(blocking=False)
    if vtkoutput:
        vtk.Do()
