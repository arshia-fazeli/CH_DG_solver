from ngsolve import *
from user_settings import *
from helpers import *

def weak_form_oseen(fes, sold, siter, trial_list, test_list):
    [c , mu] = trial_list
    [q , v] = test_list
    n = specialcf.normal(2)
    h = specialcf.mesh_size
    alpha = 4 * order ** 2 / h
    a = BilinearForm(fes)
    a += tau * M * grad(mu) * grad(q) * dx
    a += -tau * M * n * gradavg(mu) * Jump(q) * dx(skeleton=True)
    a += -tau * M * n * gradavg(q) * Jump(mu) * dx(skeleton=True)
    a += tau * M * alpha * Jump(q) * Jump(mu) * dx(skeleton=True)
    a += mu * v * dx
    # a += -200 * (c - 3 * c ** 2 + 2 * c ** 3) * v * dx
    a += -200 * (c - 3 * c * siter.components[0] + 2 * c * siter.components[0] ** 2) * v * dx
    a += -lamdba * grad(c) * grad(v) * dx
    a += lamdba * n * gradavg(c) * Jump(v) * dx(skeleton=True)
    a += lamdba * n * gradavg(v) * Jump(c) * dx(skeleton=True)
    a += -lamdba * alpha * Jump(c) * Jump(v) * dx(skeleton=True)

    a += SymbolicBFI(c * q)

    a.Assemble()

    l = LinearForm(fes)
    l += sold.components[0] * q * dx

    l.Assemble()
    return a, l