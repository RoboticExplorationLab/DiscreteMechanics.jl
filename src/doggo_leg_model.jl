## Doggo leg model
using ForwardDiff
using Plots
using LinearAlgebra

# model
g = 9.81

L1 = 0.5
L2 = 0.5
L3 = 1.0
L4 = 1.0

l1 = L1/2
l2 = L2/2
l3 = L3/2
l4 = L4/2

m1 = 1
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

# inputs
a = pi/4
b = -pi/4
q = [a;b]

ȧ = 1.0
ḃ = 1.0
q̇ = [ȧ;ḃ]

#FK
d = L1*sin(0.5*(a-b))
h1 = L1*cos(0.5*(a-b))
h2 = sqrt(L3^2 -d^2)
h = h1 + h2
t = atan(2*h*d,L1^2 + L3^2 -h^2)

function theta(a,b)
    d = L1*sin(0.5*(a-b))
    h1 = L1*cos(0.5*(a-b))
    h2 = sqrt(L3^2 -d^2)
    h = h1 + h2
    atan(2*h*d,L1^2 + L3^2 -h^2)
end

theta(q) = theta(q...)

a2 = t + a - pi
b2 = pi - t + b

x1 = l1*sin(a)
y1 = -l1*cos(a)

x2 = l2*sin(b)
y2 = -l2*cos(b)

x3 = L1*sin(a) + l3*sin(a2)
y3 = -L1*cos(a) + -l3*cos(a2)

x4 = L2*sin(b) + l4*sin(b2)
y4 = -L2*cos(b) + -l4*cos(b2)

xt = L1*sin(a) + L3*sin(a2)
yt = L1*cos(a) + L3*cos(a2)

height = [y1, y2, y3, y4]

r1(a,b) = [l1*sin(a); -l1*cos(a)]

r2(a,b) = [l2*sin(b); -l2*cos(b)]

function r3(a,b)
    a2 = theta(a,b) + a - pi
    [L1*sin(a) + l3*sin(a2); -L1*cos(a) + -l3*cos(a2)]
end

function r4(a,b)
    b2 = pi - theta(a,b) + b
    [L2*sin(b) + l4*sin(b2); -L2*cos(b) + -l4*cos(b2)]
end

function rt(a,b)
    a2 = theta(a,b) + a - pi
    b2 = pi - theta(a,b) + b
    [L1*sin(a) + L3*sin(a2); L1*cos(a) + L3*cos(a2)]
end

r1([a;b])
r1(q) = r1(q...)
r2(q) = r2(q...)
r3(q) = r3(q...)
r4(q) = r4(q...)
rt(q) = rt(q...)

L = L1^2 + L3^2
dda = L1/2*cos((a-b)/2)
dha = -L1/2*sin(0.5*(a-b)) - 1/sqrt(L3^2 - d^2)*d*dda
h
d2hda = 2*h*dda + 2*d*dha
dta = -2*h*d/(4*(h^2)*d^2 + (L - h^2)^2)*(-2*h*dha) + (L - h^2)/(4*(h^2)*d^2 + (L-h^2)^2)*d2hda

ddb = -L1/2*cos((a-b)/2)
dhb = L1/2*sin(0.5*(a-b)) - 1/sqrt(L3^2 - d^2)*d*ddb
d2hdb = 2*h*ddb + 2*d*dhb
dtb = -2*h*d/(4*(h^2)*d^2 + (L - h^2)^2)*(-2*h*dhb) + (L - h^2)/(4*(h^2)*d^2 + (L-h^2)^2)*d2hdb

da2a = dta + 1.0
da2b = dtb
db2a = -dta
db2b = -dtb + 1.0

dx1a = l1*cos(a)
dx1b = 0

dy1a = l1*sin(a)
dy1b = 0

dx2a = 0
dx2b = l2*cos(b)

dy2a = 0
dy2b = l2*sin(b)

dx3a = L1*cos(a) + l3*cos(a2)*da2a
dx3b = l3*cos(a2)*da2b

dy3a = L1*sin(a) + l3*sin(a2)*da2a
dy3b = l3*sin(a2)*da2b

dx4a = l4*cos(b2)*db2a
dx4b = L2*cos(b) + l4*cos(b2)*db2b

dy4a = l4*sin(b2)*db2a
dy4b = L2*sin(b) + l4*sin(b2)*db2b

dr1 = [dx1a dx1b; dy1a dy1b]
dr2 = [dx2a dx2b; dy2a dy2b]
dr3 = [dx3a dx3b; dy3a dy3b]
dr4 = [dx4a dx4b; dy4a dy4b]

ForwardDiff.jacobian(r1,[a;b]) ≈ dr1
ForwardDiff.jacobian(r2,[a;b]) ≈ dr2
ForwardDiff.jacobian(r3,[a;b]) ≈ dr3
ForwardDiff.jacobian(r4,[a;b]) ≈ dr4
ForwardDiff.gradient(theta,[a;b]) ≈ [dta;dtb]

dr1 = [dx1a dx1b; dy1a dy1b; 1.0 0.0]
dr2 = [dx2a dx2b; dy2a dy2b; 0.0 1.0]
dr3 = [dx3a dx3b; dy3a dy3b; 1.0+dta dtb]
dr4 = [dx4a dx4b; dy4a dy4b; dta 1.0-dtb]

jac = [dr1, dr2, dr3, dr4]

function gen_M(M,jac)
    M̄ = zeros(2,2)

    for k = 1:4
        M̄ += jac[k]'*M[k]*jac[k]
    end

    return M̄
end

M̄ = gen_M(M,jac)

function gen_V(m,height)
    V = 0.0
    for k = 1:4
        V += m[k]*g*height[k]
    end
    return V
end

V = gen_V(m,height)

function gen_L(q,q̇)
    M̄ = gen_M(M,jac)
    V = gen_V(m,height)

    L = 0.5*q̇'*M̄*q̇ - V
end

L = gen_L(q,q̇)

## generate function for four bar mechanism joint positions as function of α, β
function gen_four_bar(L1::T,L2::T,L3::T,L4::T) where T

    function theta(a::T,b::T) where t
        d = L1*sin(0.5*(a-b))
        h1 = L1*cos(0.5*(a-b))
        h2 = sqrt(L3^2 -d^2)
        h = h1 + h2
        atan(2*h*d,L1^2 + L3^2 -h^2)
    end

    function pos(a::T,b::T) where T
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
        x3 = L1*sin(a) + L3*sin(a2)
        y3 = -L1*cos(a) + -L3*cos(a2)

        # x4 = L2*sin(b) + L4*sin(b2)
        # y4 = -L2*cos(b) + -L4*cos(b2)

        [(0.0,0.0), (x1,y1), (x2,y2), (x3,y3)]
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

four_bar = gen_four_bar(L1,L2,L3,L4)
plot_four_bar(four_bar,pi/3,-pi/2)
