## Doggo leg model
using ForwardDiff
using Plots
using LinearAlgebra


# model
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

J1 = 1
J2 = 1
J3 = 1
J4 = 1

M1 = Diagonal([m1*ones(2);J1])
M2 = Diagonal([m2*ones(2);J2])
M3 = Diagonal([m3*ones(2);J3])
M4 = Diagonal([m4*ones(2);J4])

# M = cat(M1,M2,M3,M4,dims=(1,2))

# inputs
a = pi/4
b = -pi/4

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
dr4 = [dx4a dx4b; dy4a dy4b; dta 1.0+dtb]