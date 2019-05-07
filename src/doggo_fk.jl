using PartedArrays
using StaticArrays
using ForwardDiff
using BenchmarkTools
using Rotations
using MeshCatMechanisms
using MeshCat
using CoordinateTransformations
using LinearAlgebra
include("doggo_model.jl")

vis = Visualizer()
open(vis)
setobject!(vis,Triad(1))

# Position of leg frame
d1 = [2,-1,0]
d2 = [2,1,0]
d3 = [-2,-1,0]
d4 = [-2,1,0]
d_legs = (d1,d2,d3,d4)

# Orientation of leg frames
q1 = Quaternion(RotX(pi/2))
q2 = Quaternion(RotZ(pi)*RotX(pi/2))
q3 = q1
q4 = q2
q_legs = (q1,q2,q3,q4)
q1
T_legs = [compose(Translation(d_legs[i]), LinearMap(Quat(q_legs[i]))) for i = 1:4]
settransform!(vis,T_legs[1])
T_legs

# Doggo
include("doggo_default.jl")

# Inputs
r0 = @SVector [0.1, 0.2, 0.3]
q0 = Quaternion(RotY(-pi/2))

q = [deg2rad(45), deg2rad(-45),
     deg2rad(45), deg2rad(-45),
     deg2rad(45), deg2rad(-45),
     deg2rad(45), deg2rad(-45)]

# Forward Kinematics
function leg2Dto3D(r)
    z = zeros(7)
    z[1:2] = r[1:2]
    q = Quat(RotZ(r[3]))
    z[4] = q.w
    z[5] = q.x
    z[6] = q.y
    z[7] = q.z
    return z
end

function transform_leg(q)
    r = leg2Dto3D.(fk(leg, q[1:2]))
    pos = [q_legs[1]*r[i][1:3] + d_legs[1] for i = 1:4]
    rot = [q_legs[1]*Quat(r[i][4:7]...) for i = 1:4]
    return pos, rot
end

function leg2Dto3D_2(r)
    T = Translation(r[1],r[2],0) ∘ LinearMap(RotZ(r[3]))
end
function transform_leg2(q)
    r = leg2Dto3D_2.(fk(leg, q[1:2]))
    T = [T_legs[1] ∘ r[k] for k = 1:4]
    rot = [trans.linear for trans in T]
    pos = [trans.translation for trans in T]
    T
end
T1 = transform_leg(q)
T2 = transform_leg2(q)
T1 .≈ T2
# @btime transform_leg($q)
# @btime transform_leg2($q)

# First Leg, First Link
function fk11(q)
    leg_num = 1
    link_num = 3

    # Leg kinematics
    q_2d = fk(leg, q[8:9])[link_num]

    # Convert to 3D coordinate frame
    t11 = leg2Dto3D_2(q_2d)

    # Move to body frame
    tb1 = T_legs[leg_num] ∘ t11

    # Create world transformation
    r0 = q[1:3]
    q0 = q[4:7]
    twb = Translation(r0) ∘ LinearMap(Quat(q0...,true))

    # Transform to World
    tw1 = twb ∘ tb1
    [tw1.translation; SVector(Quaternion(tw1.linear))]
end

function fk11_(q)
    leg_num = 1
    link_num = 3

    # Leg kinematics
    q_2d = fk(leg, q[8:9])[link_num]

    # Convert to 3D coordinate frame
    t11 = leg2Dto3D_2(q_2d)

    # Move to body frame
    tb1 = T_legs[leg_num] ∘ t11

    # Create world transformation
    r0 = q[1:3]
    q0 = Quaternion(q[4:7])

    # Transform to World
    r = q0*tb1.translation + r0
    q = q0*Quaternion(tb1.linear)
    [r; SVector(q)]
end

function link_jacobian(q)
    leg_num = 1
    link_num = 3

    # Get leg kinematics
    r = fk(leg, q[8:9])[link_num]
    th = r[3]

    r0 = q[1:3]
    q0 = Quaternion(q[4:7])
    qw1 = q0*q_legs[leg_num]

    # Get translation from link to body frame
    t11 = leg2Dto3D_2(r)
    tb1 = T_legs[leg_num] ∘ t11
    rb1 = tb1.translation

    # Initialize Jacobian
    jac = zeros(7,9)

    # Jacobian of position wrt body
    jac[1:3,1:3] = Diagonal(I,3)
    jac[1:3,4:7] = deriv_conj(q0, rb1)

    # Get leg Jacobian and convert to 3D
    J1 = jacobian(leg, q[8:9])[link_num]
    dqdth = repeat([-sin(th/2)/2, 0, 0, cos(th/2)/2],1,2)
    J3d = [J1[1:2,:]; zeros(1,2); dqdth.*J1[3:3,:]]
    jac[1:3,8:9] = conj(qw1)*J3d[1:3,:]

    # Jacobian of orientation
    jac[4:7,4:7] = Rmult(Quaternion(tb1.linear))
    jac[4:7,8:9] = Lmult(qw1)*J3d[4:7,:]
    return jac
end

T_legs

r0 = @SVector [0, 0, 0.0]
q0 = Quaternion(RotY(-pi/2))
q0 = Quaternion(ones(Quat))
q = [r0; SVector(q0); deg2rad(40); deg2rad(-45)]
t11 = fk11(q)
T = Translation(t11[1:3]) ∘ LinearMap(Quat(t11[4:7]...))
settransform!(vis, T)
fk11_(q) ≈ t11

jac = link_jacobian(q)
jac_fd = ForwardDiff.jacobian(fk11,q)
ForwardDiff.jacobian(fk11_,q) ≈ jac

@code_warntype link_jacobian(q)

@btime ForwardDiff.jacobian($fk11_,$q)
@btime link_jacobian($q)





settransform!(vis,t11)




# World Transformation
Tw = [Twb ∘ T for T in T2]



settransform!(vis,T_legs[1])




# Test Quaternion stuff
q1 = Quat(q1)
q2 = Quat(q2)
q1_ = Quaternion(q1)
q2_ = Quaternion(q2)
q1_*q2_
Lmult(q1_)*q2_ == Rmult(q2_)*q1_

a = rand(3)
q1_*Quaternion(0.,a)*inv(q1_) == q1_*a
(Lmult(q1_)*Rmult(inv(q1_))*Quaternion(0.,a))[2:4] ≈ q1_*a
(Rmult(inv(q1_))*Lmult(q1_)*Quaternion(0.,a))[2:4] ≈ q1_*a
conj(q1_)*a ≈ q1_*a

r = a
s,v = scalar(q1_), vec(q1_)
j2 = 2(s*I + skew(v))*r
j3 = v*r' + I*v'r - 2s*skew(r) - skew(v)*skew(r) - skew(skew(v)*r)
jac = [zeros(1,4); j2 j3]

q = rand(Quat)
q = [q.w, q.x, q.y, q.z]

myfun(q) = Quaternion(q)*a
ForwardDiff.jacobian(myfun,q)
deriv_conj(Quaternion(q),a)

Quat(q...)*a ≈ Quaternion(q)*a
myfun2(q) = Quat(q[1],q[2],q[3],q[4])*a
ForwardDiff.jacobian(myfun2,q)

a = @SVector rand(3)
a = rand(3)
@btime $q1_*$a
@btime $q1*$a
q1 = Quat(q1)
Quat(q1)*a
Quat(q1)*Quat(q2)
