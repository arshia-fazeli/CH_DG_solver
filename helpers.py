from ngsolve import grad

def Jump(u):
    return u - u.Other()

def avg(u):
    return (u + u.Other())/2

def gradavg(u):
    return (grad(u) + grad(u).Other())/2