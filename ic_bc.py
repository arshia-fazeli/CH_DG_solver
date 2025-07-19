from ngsolve import *
import random
from user_settings import initial_roughness


def sqr(x):
    return x * x

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
        #total_mass = Integrate(s.components[0],mesh,VOL)
    print()
