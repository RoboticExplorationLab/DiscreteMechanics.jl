## Doggo leg model
using ForwardDiff, DiffResults
using Plots
using LinearAlgebra
using BenchmarkTools
using PartedArrays
using MatrixCalculus
using Test

include("doggo_model.jl")

# model
g = 9.81

L1 = 0.5
L2 = 0.5
L3 = 1.5
L4 = L3

l1 = L1/2
l2 = L2/2
l3 = L3/2
l4 = L4/2

m1 = 1.0
m2 = 1
m3 = 1
m4 = 1

m = [m1, m2, m3, m4]

J1 = 1
J2 = 1
J3 = 1
J4 = 1

M1 = Diagonal([m1*ones(2);J1])
M2 = Diagonal([m2*ones(2);J2])
M3 = Diagonal([m3*ones(2);J3])
M4 = Diagonal([m4*ones(2);J4])

M = [M1, M2, M3, M4]

# M = cat(M1,M2,M3,M4,dims=(1,2))

# Create Doggo Model
doggo = Doggo(L1,L3, [m1,m2,m3,m4], [J1,J2,J3,J4])

# inputs
a = pi/3 + rand()/5
b = -pi/4 + rand()/5
q = [a;b]

ȧ = 1.0
ḃ = 1.0
q̇ = [ȧ;ḃ]

# Forward Kinematics
function theta(a,b)
    d = L1*sin(0.5*(a-b))
    h1 = L1*cos(0.5*(a-b))
    h2 = sqrt(L3^2 -d^2)
    h = h1 + h2
    atan(2*h*d,L1^2 + L3^2 -h^2)
end

function trigtheta(a,b)
    d = L1*sin(0.5*(a-b))
    h1 = L1*cos(0.5*(a-b))
    h2 = sqrt(L3^2 -d^2)
    h = h1 + h2
    hyp = sqrt((2*h*d)^2 + (L1^2 + L3^2 - h^2)^2)
    st = 2*h*d/hyp
    ct = (L1^2 + L3^2 - h^2)/hyp
    return st, ct
end

function trigtheta2(a,b,L1,L3)
    d = L1*sin(0.5*(a-b))
    h1 = L1*cos(0.5*(a-b))
    h2 = sqrt(L3^2 -d^2)
    h = h1 + h2
    st = h*d/(L1*L3)
    ct = (L1^2 + L3^2 - h^2)/(2*L1*L3)
    return st, ct
end

theta(q) = theta(q...)

r1(a,b) = [l1*sin(a), -l1*cos(a), a]

r2(a,b) = [l2*sin(b), -l2*cos(b), b]

function r3_(a,b)
    a2 = theta(a,b) + a - pi
    t = theta(a,b)
    [L1*sin(a) + l3*sin(a2), -L1*cos(a) + -l3*cos(a2), a + t - pi]
end

function r3(a,b)
    st,ct = trigtheta(a,b)
    t = theta(a,b)
    [(L1 - l3*ct)*sin(a) - l3*st*cos(a),
     (-L1 + l3*ct)*cos(a) - l3*st*sin(a),
     a + t - pi]
 end


function r4(a,b)
    b2 = pi - theta(a,b) + b
    t = theta(a,b)
    [L2*sin(b) + l4*sin(b2), -L2*cos(b) + -l4*cos(b2), b - t + pi]
end

function rt(a,b)
    st,ct = trigtheta(a,b)
    a2 = theta(a,b) + a - pi
    b2 = pi - theta(a,b) + b
    [L1*sin(a) + L3*sin(a2), L1*cos(a) + L3*cos(a2), ]
end

r1(q) = r1(q...)
r2(q) = r2(q...)
r3(q) = r3(q...)
r4(q) = r4(q...)
rt(q) = rt(q...)

# Test the FK derivatives
fk_fun = [r1,r2,r3,r4]
val  = [r(q) for r in fk_fun]
grad = [ForwardDiff.jacobian(r,q) for r in fk_fun]
hess = [ForwardDiff.jacobian(q->vec(ForwardDiff.jacobian(r,q)),q) for r in fk_fun]
fk(doggo, q) ≈ val
fk_hess(doggo, q)[1] ≈ grad
fk_hess(doggo, q)[2] ≈ hess
jacobian(doggo, q) ≈ grad
jac = jacobian(doggo, q)

# @btime [ForwardDiff.jacobian(q->vec(ForwardDiff.jacobian(r,q)),$q) for r in $fk]
# @btime fk_hess($doggo, $q)

# Dynamics
x = [q; q̇]
part = create_partition2((2,2),(:x,:v))

d,h = leg_dims(doggo, q)
@inferred leg_dims(doggo, q)
@inferred trigtheta(doggo, d, h)
@inferred theta(doggo, d, h)
@inferred fk(doggo, q)
@inferred jacobian(doggo, q)
@inferred fk_hess(doggo, q)
@inferred mass_matrix(doggo, jac)
@inferred mass_matrix(doggo, q)
@inferred lagrangian(doggo, q, q̇)
@inferred get_V(doggo, q)

# Check gradient of lagrangian
lagrangian(x) = lagrangian(doggo, x[1:2], x[3:4])
L_grad = ForwardDiff.gradient(lagrangian, x)
grad_L(doggo, q, q̇) ≈ L_grad

# Check potential energy gradient
get_V(q) = get_V(doggo, q)
ForwardDiff.gradient(get_V, q) ≈ get_∇V(doggo, jac)

# Test mass matrix
part = NamedTuple{(:xx,:xv,:vx,:vv)}(create_partition2((2,2)))
L_hess = BlockArray(ForwardDiff.hessian(lagrangian, x), part)
M_ = mass_matrix(doggo, q)
M_ ≈ L_hess.vv

# Test full euler lagrange equation (assuming some qdd)
qdd = rand(2)
qd = q̇
el = L_hess.vx*q̇ + L_hess.vv*qdd - L_grad[1:2]
euler_lagrange_auto(doggo, q, qd, qdd) == el
euler_lagrange(doggo, q, qd, qdd) ≈ el
@inferred euler_lagrange(doggo, q, qd, qdd)
@inferred euler_lagrange_auto(doggo, q, qd, qdd)
@btime euler_lagrange_auto($doggo, $q, $qd, $qdd)
@btime euler_lagrange($doggo, $q, $qd, $qdd)


@code_warntype create_partition2((2,2),(2,2),(:x,:v),(:x,:v))
names_all = create_partition2((2,2),(2,2),(:x,:v),(:x,:v))
@code_warntype PartedArrays.create_nt(names_all,part)

res = DiffResults.HessianResult(x)
ForwardDiff.hessian!(res, lagrangian, x)
DiffResults.gradient(res) ≈ L_grad

dynamics(doggo, q, qd, [0,0])
n,m = 4,2
f(x,u) = [x[3:4]; dynamics(doggo, x[1:2], x[3:4], u)]
fd = rk3(f, 0.1)
x = [deg2rad(45), deg2rad(-45), 0, 0]
u = zeros(2)
fd(x,u)

function simulate(x0,u,N, doplot=false)
    X = zeros(n,N)
    X[:,1] = x0
    for k = 1:N-1
        X[:,k+1] = fd(X[:,k],u)
        if doplot
            p = plot(doggo, X[:,k])
            display(p)
        end
    end
    return X
end
u = [1,-1]*5
X = simulate(x,u,51, true)
plot(doggo, X[:,51])


plot(doggo, [deg2rad(50), deg2rad(-90)], 2)



# Second Derivatives
ForwardDiff.hessian(theta,q)
hess = ForwardDiff.hessian(sintheta,q)
hess = ForwardDiff.hessian(costheta,q)[1,1]
ForwardDiff.gradient(sintheta,q)
phi = (a-b)/2

dsdaa = 1/(4L3)*((-(d^2/h2^2 + 3)*h1*d/h2)*cos(phi) - (h-d^2/h2)*sin(phi) - 2h1*sin(phi) - d*cos(phi))
dcdaa = 1/4L3*(1/h2*(-d^2 + h*h1 + (h/h2 - 1)*d^2*h1/h2 + h*h2)*cos(phi) - d*(h/h2 + (1+h1/h2))*sin(phi))
dsdbb = dsdaa
dcdbb = dcdaa
dsdab = -dsdaa
dcdab = -dcdaa

dx3aa = -(L1-l3*ct)* sin(a) - l3*cos(a)*dcda +
             l3*st * cos(a) + l3*sin(a)*dsda -
             l3*dcda*cos(a) - l3*sin(a)*dcdaa +
             l3*dsda*sin(a) - l3*cos(a)*dsdaa

dx3ab = -l3*cos(a)*dcdb  + l3*sin(a)*dsdb -
         l3*sin(a)*dcdab - l3*cos(a)*dsdab

dx3bb = -l3*sin(a)*dcdbb - l3*cos(a)*dsdbb

rx3(q) = r3(q)[1]
ForwardDiff.hessian(rx3,q) ≈ [dx3aa dx3ab; dx3ab dx3bb]

dy3a = (L1-l3*ct)*sin(a) - l3*st*cos(a) + l3*cos(a)*dcda - l3*sin(a)*dsda
dy3b = l3*cos(a)*dcdb - l3*sin(a)*dsdb

dy3aa = (L1-l3*ct)*cos(a)  - l3*sin(a)*dcda +
            l3*st*sin(a)   - l3*cos(a)*dsda -
            l3*sin(a)*dcda + l3*cos(a)*dcdaa -
            l3*cos(a)*dsda - l3*sin(a)*dsdaa
dy3ab = -l3*sin(a)*dcdb - l3*cos(a)*dsdb +
         l3*cos(a)*dcdab - l3*sin(a)*dsdab
dy3bb = l3*cos(a)*dcdbb -l3*sin(a)*dsdbb

ry3(q) = r3(q)[2]
ForwardDiff.gradient(ry3,q) ≈ [dy3a,dy3b]
ForwardDiff.hessian(ry3,q) ≈ [dy3aa dy3ab; dy3ab dy3bb]


dx4a = -l4*sin(b)*dcda + l4*cos(b)*dsda
dx4b = (L2 - l4*ct)*cos(b) - l4*st*sin(b) - l4*sin(b)*dcdb + l4*cos(b)*dsdb

dx4aa = -l4*sin(b)*dcdaa + l4*cos(b)*dsdaa
dx4ab = -l4*dcda*cos(b) - l4*sin(b)*dcdab +
        -l4*dsda*sin(b) + l4*cos(b)*dsdab
dx4bb = -(L2-l3*ct)*sin(b) - l4*cos(b)*dcdb +
        -l4*st*cos(b)      - l4*sin(b)*dsdb +
        -l3*dcdb*cos(b)    - l3*sin(b)*dcdbb +
        -l4*dsdb*sin(b)    + l3*cos(b)*dsdbb


rx4(q) = r4(q)[1]
ForwardDiff.gradient(rx4,q) ≈ [dx4a,dx4b]
ForwardDiff.hessian(rx4,q) ≈ [dx4aa dx4ab; dx4ab dx4bb]

dy4a = l4*cos(b)*dcda + l4*sin(b)*dsda
dy4b = (L2 - l4*ct)*sin(b) + l4*st*cos(b) + l4*cos(b)*dcdb + l4*sin(b)*dsdb

dy4aa = l4*cos(b)*dcdaa + l4*sin(b)*dsdaa
dy4ab = -l4*dcda*sin(b) + l4*cos(b)*dcdab +
         l4*dsda*cos(b) + l4*sin(b)*dsdab
dy4bb = (L2-l4*ct)*cos(b) - l4*sin(b)*dcdb +
        -l4*st*sin(b)     + l4*cos(b)*dsdb +
        -l4*dcdb*sin(b)   + l4*cos(b)*dcdbb +
         l4*dsdb*cos(b)   + l4*sin(b)*dsdbb

ry4(q) = r4(q)[2]


println("passed")

## generate function for four bar mechanism joint positions as function of α, β
function gen_four_bar(L1::T,L3::T) where T
    L2 = L1
    L4 = L3
    l3 = L3/2
    l4 = L4/2

    function theta(a::T,b::T) where t
        d = L1*sin(0.5*(a-b))
        h1 = L1*cos(0.5*(a-b))
        h2 = sqrt(L3^2 -d^2)
        h = h1 + h2
        atan(2*h*d,L1^2 + L3^2 -h^2)
    end

    function pos(a::
        T,b::T) where T
        t = theta(a,b)

        a2 = t + a - pi
        b2 = pi - t + b

        # right elbow
        x1 = L1*sin(a)
        y1 = -L1*cos(a)

        #left elbow
        x2 = L2*sin(b)
        y2 = -L2*cos(b)

        # bottom joint
        x3 = L1*sin(a) + l3*sin(a2)
        y3 = -L1*cos(a) -l3*cos(a2)

        x4 = L2*sin(b) + l4*sin(b2)
        y4 = -L2*cos(b) - l4*cos(b2)

        # tip
        xt = L1*sin(a) + L3*sin(a2)
        yt = -L1*cos(a) -L3*cos(a2)

        [(0.0,0.0), (x1,y1), (x2,y2), (xt,yt)]
    end

    function pos(a::T,b::T,c::Bool) where T

        # right elbow
        x1 = L1*sin(a)
        y1 = -L1*cos(a)

        #left elbow
        x2 = L2*sin(b)
        y2 = -L2*cos(b)

        st, ct = trigtheta2(a,b,L1,L3)

        x3 = (L1 - l3*ct)*sin(a) - l3*st*cos(a)
        y3 = (-L1 + l3*ct)*cos(a) - l3*st*sin(a)

        x4 = (L2 - l4*ct)*sin(b) + l4*st*cos(b)
        y4 = (-L2 + l4*ct)*cos(b) + l4*st*sin(b)

        xt = (L2 - L4*ct)*sin(b) + L4*st*cos(b)
        yt = (-L2 + L4*ct)*cos(b) + L4*st*sin(b)

        [(0.0,0.0), (x1,y1), (x2,y2), (xt,yt)]
    end
    return pos
end

function plot_four_bar(f::Function,α::T,β::T) where T
    pos = f(α,β)
    p = plot(title="Doggo Leg alpha=$(round(α,digits=2)),beta=$(round(β,digits=2))",xlim=(-2,2),ylim=(-2,2),aspectratio=:equal,label="")
    plot!([pos[1][1],pos[2][1]],[pos[1][2],pos[2][2]],color=:black,linewidth=2,label="")
    plot!([pos[1][1],pos[3][1]],[pos[1][2],pos[3][2]],color=:black,linewidth=2,label="")
    plot!([pos[2][1],pos[4][1]],[pos[2][2],pos[4][2]],color=:black,linewidth=2,label="")
    plot!([pos[3][1],pos[4][1]],[pos[3][2],pos[4][2]],color=:black,linewidth=2,label="")
end

four_bar = gen_four_bar(rand()+1,rand()+2)
four_bar(q[1],q[2])[4] .≈ four_bar(q[1],q[2],true)[4]
@btime four_bar($q[1],$q[2])
@btime four_bar($q[1],$q[2],true)
plot_four_bar(four_bar,pi/4,-pi/3)
