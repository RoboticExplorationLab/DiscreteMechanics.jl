using ForwardDiff
using LinearAlgebra
using Plots
using Statistics

n = 1

# Params
mass = 1
g = -9.81
ℓ = 0.5

# Lagrangian
L(q,q̇) = 1/2*mass*ℓ^2*q̇^2 - mass*g*ℓ*(1-cos(q))

# Discrete Lagrangian (midpoint)
Ld(q,q_,h) = h*L((q+q_)/2, (q_-q)/h)
Ld2(q,q_,h) = h*mass*ℓ*( 1/2*ℓ*((q_-q)/h)^2 - g*(1-cos((q+q_)/2)) )
# Ld2(q,q_,h) = g*(1-cos((q+q_)/2))

q = rand()
q_ = rand()
h = 0.01
Ld(q,q_,h) - Ld2(q,q_,h)

# Discrete Euler Lagrange
D1(q,q_,h) = h*mass*ℓ*( -ℓ/h^2*(q_-q) - g/2*sin((q+q_)/2) )
D2(_q,q,h) = mass*ℓ^2/h*(q-_q) - mass*ℓ*h*g/2*sin((_q+q)/2)

Ld_aug(Q) = Ld2(Q[1],Q[2],h)

Q = [q,q_]
u = [0.0]
ForwardDiff.gradient(Ld_aug,Q)
D1(q,q_,h)
D2(q,q_,h)


function dyn_discrete(q1,q2,u; eps=1e-10)
    q3 = 2*q2-q1  # Guess using Euler

    # Discrete Euler-Lagrange
    P(q1,q2,q3,u) = D2(q1,q2,h) + D1(q2,q3,h) + u[1]
    D2_Ld = ForwardDiff.gradient(Ld_aug,[q1;q2])[n+1:end]

    for i = 1:10
        hessLd = ForwardDiff.hessian(Ld_aug,[q2;q3])
        gradLd = ForwardDiff.gradient(Ld_aug,[q2;q3])
        D1_Ld = gradLd[1:n]
        val = D2_Ld + D1_Ld + u
        q3 = q3 - hessLd[1:n,n+1:end]\val
        if norm(val,Inf) < eps
            break
        end
    end
    return q3
end
q1 = q
q2 = q + q_*h
q3 = dyn_discrete([q1],[q2],u)


# Continuous
L_aug(Q) = L(Q[1],Q[2])
gradL = ForwardDiff.gradient(L_aug,Q)
hessL = ForwardDiff.hessian(L_aug,Q)
function dyn(Q,u)
    gradL = ForwardDiff.gradient(L_aug,Q)
    hessL = ForwardDiff.hessian(L_aug,Q)
    Ldd = hessL[n+1:end,n+1:end]
    qdd = Ldd\(u + gradL[1:n] - hessL[1:n,n+1:end]*Q[n+1:end])
    return [Q[2]; qdd]
end
dyn(Q,u)

# Analytical
function dyn0(Q,u)
    return [Q[2]; u[1] - g/ℓ*sin(Q[1])]
end
dyn0(Q,u)
dyn0(Q,u) ≈ dyn(Q,u)

function rk3(f::Function, dt::Float64)
        # Runge-Kutta 3 (zero order hold)
    fd(x,u,dt=dt) = begin
        k1 = f(x,u)*dt
        k2 = f(x+k1/2,u)*dt
        k3 = f(x - k1 + 2*k2,u)*dt
        return x + (k1 + 4k2 + k3)/6
    end
end

dyn_discrete_rk3 = rk3(dyn,h)
dyn_discrete_rk3(Q,u)


function simulate1(Q0,N)
    Qs = zeros(length(Q0),N)
    Qs[:,1] = Q0
    for k = 2:N
        Qs[:,k] = dyn_discrete_rk3(Qs[:,k-1],[0])
    end
    return Qs
end

function simulate2(Q0,N)
    Q0 = [Q0[1:n];Q0[1:n]+Q[n+1:end]*h]
    Qs = zeros(n,N)
    Qs[:,1] = Q0[1:n]
    Qs[:,2] = Q0[n+1:end]
    for k = 3:N
        Qs[:,k] = dyn_discrete(Qs[:,k-2],Qs[:,k-1],[0])
    end
    return Qs
end

Q0 = [deg2rad(170),0]
res1 = simulate1(Q0,501)
plot(rad2deg.(res1[1,3:end]),label=:continuous)

res2 = simulate2(Q0,501)
plot!(rad2deg.(res2[1,:]),label=:discrete)
