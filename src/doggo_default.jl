

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

leg = DoggoLeg(L1,L3, [m1,m2,m3,m4], [J1,J2,J3,J4])
