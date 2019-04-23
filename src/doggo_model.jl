
struct Doggo{T}
    L::Vector{T}
    l::Vector{T}
    M::Vector{M} where M <: AbstractMatrix{T}
    mass::Vector{T}
    dof::Int
end

function Doggo(L_top, L_bot, m::Vector, J::Vector)
    L1 = L_top
    L2 = L1
    L3 = L_bot
    L4 = L3
    L = [L1,L2,L3,L4]
    l = [Lk/2 for Lk in L]
    M = [Diagonal([m[k]*ones(2); J[k]]) for k = 1:4]
    Doggo(L, l, M, m, 2)
end

num_links(model::Doggo) = length(model.L)

function trigtheta(doggo::Doggo,d,h)
    L1,L3 = doggo.L[[1,3]]
    st = h*d/(L1*L3)
    ct = (L1^2 + L3^2 - h^2)/(2*L1*L3)
    return st,ct
end

function theta(doggo::Doggo,d,h)
    atan(2*h*d, doggo.L[1]^2 + doggo.L[3]^2 - h^2)
end

function leg_dims(doggo::Doggo, q)
    L1,L3 = doggo.L[[1,3]]
    a,b = q

    # Calculate internal sizes
    d = L1*sin(0.5*(a-b))
    h1 = L1*cos(0.5*(a-b))
    h2 = sqrt(L3^2 -d^2)
    h = h1 + h2
    return d, h, h1, h2
end

function fk(doggo::Doggo, q)
    dof = doggo.dof
    n_links = num_links(doggo)

    # Get pieces out of model
    L1,L2,L3,L4 = doggo.L
    l1,l2,l3,l4 = doggo.l
    a,b, = q

    d,h = leg_dims(doggo, q)
    t = theta(doggo, d, h)
    st, ct = trigtheta(doggo, d, h)
    r = [zeros(eltype(q), dof) for k = 1:n_links]
    r[1] = [l1*sin(a), -l1*cos(a), a]
    r[2] = [l2*sin(b), -l2*cos(b), b]
    r[3] = [(L1 - l3*ct)*sin(a) - l3*st*cos(a),
           (-L1 + l3*ct)*cos(a) - l3*st*sin(a),
             a + t - pi]
    r[4] = [(L2 - l4*ct)*sin(b) + l4*st*cos(b),
           (-L2 + l4*ct)*cos(b) + l4*st*sin(b),
             b - t + pi]
    return r
end

function jacobian(doggo::Doggo, q)
    dof = doggo.dof
    n_links = num_links(doggo)

    # Get pieces out of model
    L1,L2,L3,L4 = doggo.L
    l1,l2,l3,l4 = doggo.l
    a,b, = q

    # Calculate internal sizes
    d = L1*sin(0.5*(a-b))
    h1 = L1*cos(0.5*(a-b))
    h2 = sqrt(L3^2 -d^2)
    h = h1 + h2

    # Calculate the sines and cosines of θ
    phi = (a-b)/2
    st,ct = trigtheta(doggo,d,h)
    dsda = ((h-d^2/h2)*cos(phi) - d*sin(phi))/(2L3)
    dcda = (h*sin(phi) + h*d/h2*cos(phi))/(2L3)
    dsdb = -dsda
    dcdb = -dcda

    # Derivatives of theta
    L = L1^2 + L3^2
    dta = (-d*(L+h^2)*sin(phi) + (-d*(L+h^2)*d/h2 + h*(L-h^2))*cos(phi))/(4*L1*L3^2)
    dtb = -dta

    # Gradient of forward Kinematics
    grad = [zeros(eltype(q),3,dof) for k = 1:n_links]
    grad[1][1,1] = l1*cos(a)
    grad[1][1,2] = 0

    grad[1][2,1] = l1*sin(a)
    grad[1][2,2] = 0

    grad[1][3,1] = 1
    grad[1][3,2] = 0

    grad[2][1,1] = 0
    grad[2][1,2] = l2*cos(b)

    grad[2][2,1] = 0
    grad[2][2,2] = l2*sin(b)

    grad[2][3,1] = 0
    grad[2][3,2] = 1

    # dx3a = L1*cos(a) + l3*cos(a2)*da2a
    # dx3b = l3*cos(a2)*da2b
    grad[3][1,1] = (L1-l3*ct)*cos(a) + l3*st*sin(a) - l3*sin(a)*dcda - l3*cos(a)*dsda
    grad[3][1,2] = -l3*sin(a)*dcdb - l3*cos(a)*dsdb

    # dy3a = L1*sin(a) + l3*sin(a2)*da2a
    # dy3b = l3*sin(a2)*da2b
    grad[3][2,1] = (L1-l3*ct)*sin(a) - l3*st*cos(a) + l3*cos(a)*dcda - l3*sin(a)*dsda
    grad[3][2,2]= l3*cos(a)*dcdb - l3*sin(a)*dsdb

    grad[3][3,1] = 1 + dta
    grad[3][3,2] = dtb

    # dx4a = l4*cos(b2)*db2a
    # dx4b = L2*cos(b) + l4*cos(b2)*db2b
    grad[4][1,1] = -l4*sin(b)*dcda + l4*cos(b)*dsda
    grad[4][1,2] = (L2 - l4*ct)*cos(b) - l4*st*sin(b) - l4*sin(b)*dcdb + l4*cos(b)*dsdb

    # dy4a = l4*sin(b2)*db2a
    # dy4b = L2*sin(b) + l4*sin(b2)*db2b
    grad[4][2,1] = l4*cos(b)*dcda + l4*sin(b)*dsda
    grad[4][2,2] = (L2 - l4*ct)*sin(b) + l4*st*cos(b) + l4*cos(b)*dcdb + l4*sin(b)*dsdb

    grad[4][3,1] = -dta
    grad[4][3,2] = 1-dtb

    return grad
end

function fk_hess(doggo::Doggo, q)
    dof = doggo.dof
    n_links = num_links(doggo)

    # Get pieces out of model
    L1,L2,L3,L4 = doggo.L
    l1,l2,l3,l4 = doggo.l
    a,b, = q

    # Calculate internal sizes
    d = L1*sin(0.5*(a-b))
    h1 = L1*cos(0.5*(a-b))
    h2 = sqrt(L3^2 -d^2)
    h = h1 + h2

    # Calculate the sines and cosines of θ
    phi = (a-b)/2
    st,ct = trigtheta(doggo,d,h)
    dsda = ((h-d^2/h2)*cos(phi) - d*sin(phi))/(2L3)
    dcda = (h*sin(phi) + h*d/h2*cos(phi))/(2L3)
    dsdb = -dsda
    dcdb = -dcda

    # Hessian of sines and cosines
    dsdaa = 1/(4L3)*((-(d^2/h2^2 + 3)*h1*d/h2)*cos(phi) - (h-d^2/h2)*sin(phi) - 2h1*sin(phi) - d*cos(phi))
    dcdaa = 1/4L3*(1/h2*(-d^2 + h*h1 + (h/h2 - 1)*d^2*h1/h2 + h*h2)*cos(phi) - d*(h/h2 + (1+h1/h2))*sin(phi))
    dsdbb = dsdaa
    dcdbb = dcdaa
    dsdab = -dsdaa
    dcdab = -dcdaa

    # Derivatives of theta
    L = L1^2 + L3^2
    dta = (-d*(L+h^2)*sin(phi) + (-d*(L+h^2)*d/h2 + h*(L-h^2))*cos(phi))/(4*L1*L3^2)
    dtb = -dta

    dtaa = (( 2d^2*h*(1+h1/h2)  + (d^2 - h1*h2)*(L+h^2)/h2 - h*(L-h^2) )*sin(phi)/2  -
        ( (2h1/h2 + 1 + d^2*h1/h2^3)*(L+h^2) + (L - 3h^2 - d^2*2h/h2)*(1+h1/h2) )*d*cos(phi)/2 ) /
        (4*L1*L3^2)
    dtab = -dtaa
    dtbb = dtaa

    # Gradient of forward Kinematics
    grad = [zeros(3,dof) for k = 1:n_links]
    grad[1][1,1] = l1*cos(a)
    grad[1][1,2] = 0

    grad[1][2,1] = l1*sin(a)
    grad[1][2,2] = 0

    grad[1][3,1] = 1
    grad[1][3,2] = 0

    grad[2][1,1] = 0
    grad[2][1,2] = l2*cos(b)

    grad[2][2,1] = 0
    grad[2][2,2] = l2*sin(b)

    grad[2][3,1] = 0
    grad[2][3,2] = 1

    # dx3a = L1*cos(a) + l3*cos(a2)*da2a
    # dx3b = l3*cos(a2)*da2b
    grad[3][1,1] = (L1-l3*ct)*cos(a) + l3*st*sin(a) - l3*sin(a)*dcda - l3*cos(a)*dsda
    grad[3][1,2] = -l3*sin(a)*dcdb - l3*cos(a)*dsdb

    # dy3a = L1*sin(a) + l3*sin(a2)*da2a
    # dy3b = l3*sin(a2)*da2b
    grad[3][2,1] = (L1-l3*ct)*sin(a) - l3*st*cos(a) + l3*cos(a)*dcda - l3*sin(a)*dsda
    grad[3][2,2]= l3*cos(a)*dcdb - l3*sin(a)*dsdb

    grad[3][3,1] = 1 + dta
    grad[3][3,2] = dtb

    # dx4a = l4*cos(b2)*db2a
    # dx4b = L2*cos(b) + l4*cos(b2)*db2b
    grad[4][1,1] = -l4*sin(b)*dcda + l4*cos(b)*dsda
    grad[4][1,2] = (L2 - l4*ct)*cos(b) - l4*st*sin(b) - l4*sin(b)*dcdb + l4*cos(b)*dsdb

    # dy4a = l4*sin(b2)*db2a
    # dy4b = L2*sin(b) + l4*sin(b2)*db2b
    grad[4][2,1] = l4*cos(b)*dcda + l4*sin(b)*dsda
    grad[4][2,2] = (L2 - l4*ct)*sin(b) + l4*st*cos(b) + l4*cos(b)*dcdb + l4*sin(b)*dsdb

    grad[4][3,1] = -dta
    grad[4][3,2] = 1-dtb

    # Hessian of forward Kinematics
    hess = [zeros(3*dof, dof) for k = 1:n_links]
    hess[1][1,1] = -l1*sin(a)         # x1/αα
    hess[1][1,2] = 0                  # x1/αβ
    hess[1][2,1] = l1*cos(a)          # y1/αα
    hess[1][2,2] = 0                  # y1/αβ
    hess[1][3,1] = 0                  # θ1/αα
    hess[1][3,2] = 0                  # θ1/αβ
    hess[1][4,1] = hess[1][1,2]       # x1/βα
    hess[1][4,2] = 0                  # x1/ββ
    hess[1][5,1] = hess[1][2,2]       # y1/βα
    hess[1][5,2] = 0                  # y1/ββ
    hess[1][6,1] = hess[1][3,2]       # θ1/βα
    hess[1][6,2] = 0                  # θ1/ββ

    hess[2][1,1] = 0                  # x2/αα
    hess[2][1,2] = 0                  # x2/αβ
    hess[2][2,1] = 0                  # y2/αα
    hess[2][2,2] = 0                  # y2/αβ
    hess[2][3,1] = 0                  # θ2/αα
    hess[2][3,2] = 0                  # θ2/αβ
    hess[2][4,1] = hess[1][1,2]       # x2/βα
    hess[2][4,2] = -l2*sin(b)         # x0/ββ
    hess[2][5,1] = hess[1][2,2]       # y2/βα
    hess[2][5,2] = l2*cos(b)          # y2/ββ
    hess[2][6,1] = hess[1][3,2]       # θ2/βα
    hess[2][6,2] = 0                  # θ2/ββ


    dx3a = (L1-l3*ct)*cos(a) + l3*st*sin(a) - l3*sin(a)*dcda - l3*cos(a)*dsda
    dx3b = -l3*sin(a)*dcdb - l3*cos(a)*dsdb
    dx3aa = -(L1-l3*ct)* sin(a) - l3*cos(a)*dcda +
             l3*st * cos(a) + l3*sin(a)*dsda -
             l3*dcda*cos(a) - l3*sin(a)*dcdaa +
             l3*dsda*sin(a) - l3*cos(a)*dsdaa

    dx3aa = -(L1-l3*ct)* sin(a) - l3*cos(a)*dcda +
             l3*st * cos(a) + l3*sin(a)*dsda -
             l3*dcda*cos(a) - l3*sin(a)*dcdaa +
             l3*dsda*sin(a) - l3*cos(a)*dsdaa
    dx3ab = -l3*cos(a)*dcdb  + l3*sin(a)*dsdb -
             l3*sin(a)*dcdab - l3*cos(a)*dsdab
    dx3bb = -l3*sin(a)*dcdbb - l3*cos(a)*dsdbb

    dy3a = (L1-l3*ct)*sin(a) - l3*st*cos(a) + l3*cos(a)*dcda - l3*sin(a)*dsda
    dy3b = l3*cos(a)*dcdb - l3*sin(a)*dsdb

    dy3aa = (L1-l3*ct)*cos(a)  - l3*sin(a)*dcda +
                l3*st*sin(a)   - l3*cos(a)*dsda -
                l3*sin(a)*dcda + l3*cos(a)*dcdaa -
                l3*cos(a)*dsda - l3*sin(a)*dsdaa
    dy3ab = -l3*sin(a)*dcdb - l3*cos(a)*dsdb +
             l3*cos(a)*dcdab - l3*sin(a)*dsdab
    dy3bb = l3*cos(a)*dcdbb -l3*sin(a)*dsdbb


    dx4a = -l4*sin(b)*dcda + l4*cos(b)*dsda
    dx4b = (L2 - l4*ct)*cos(b) - l4*st*sin(b) - l4*sin(b)*dcdb + l4*cos(b)*dsdb

    dx4aa = -l4*sin(b)*dcdaa + l4*cos(b)*dsdaa
    dx4ab = -l4*dcda*cos(b) - l4*sin(b)*dcdab +
            -l4*dsda*sin(b) + l4*cos(b)*dsdab
    dx4bb = -(L2-l3*ct)*sin(b) - l4*cos(b)*dcdb +
            -l4*st*cos(b)      - l4*sin(b)*dsdb +
            -l3*dcdb*cos(b)    - l3*sin(b)*dcdbb +
            -l4*dsdb*sin(b)    + l3*cos(b)*dsdbb

    dy4a = l4*cos(b)*dcda + l4*sin(b)*dsda
    dy4b = (L2 - l4*ct)*sin(b) + l4*st*cos(b) + l4*cos(b)*dcdb + l4*sin(b)*dsdb

    dy4aa = l4*cos(b)*dcdaa + l4*sin(b)*dsdaa
    dy4ab = -l4*dcda*sin(b) + l4*cos(b)*dcdab +
             l4*dsda*cos(b) + l4*sin(b)*dsdab
    dy4bb = (L2-l4*ct)*cos(b) - l4*sin(b)*dcdb +
            -l4*st*sin(b)     + l4*cos(b)*dsdb +
            -l4*dcdb*sin(b)   + l4*cos(b)*dcdbb +
             l4*dsdb*cos(b)   + l4*sin(b)*dsdbb


    hess[3][1,1] = dx3aa
    hess[3][1,2] = dx3ab
    hess[3][2,1] = dy3aa
    hess[3][2,2] = dy3ab
    hess[3][3,1] = dtaa
    hess[3][3,2] = dtab
    hess[3][4,1] = dx3ab
    hess[3][4,2] = dx3bb
    hess[3][5,1] = dy3ab
    hess[3][5,2] = dy3bb
    hess[3][6,1] = dtab
    hess[3][6,2] = dtbb

    hess[4][1,1] = dx4aa
    hess[4][1,2] = dx4ab
    hess[4][2,1] = dy4aa
    hess[4][2,2] = dy4ab
    hess[4][3,1] = -dtaa
    hess[4][3,2] = -dtab
    hess[4][4,1] = dx4ab
    hess[4][4,2] = dx4bb
    hess[4][5,1] = dy4ab
    hess[4][5,2] = dy4bb
    hess[4][6,1] = -dtab
    hess[4][6,2] = -dtbb

    return grad, hess
end

function mass_matrix(doggo::Doggo, jac::Vector{Matrix{T}}) where T
    dof = doggo.dof
    M = doggo.M
    M̄ = zeros(T,dof,dof)

    for k = 1:num_links(doggo)
        M̄ += jac[k]'*M[k]*jac[k]
    end

    return M̄
end
mass_matrix(doggo::Doggo{T}, q::AbstractVector) where T = mass_matrix(doggo, jacobian(doggo, q))


get_T(doggo::Doggo, q, q̇) = 0.5*q̇'mass_matrix(doggo, q)*q̇

function get_V(doggo::Doggo, q)
    pos = fk(doggo, q)
    n_links = num_links(doggo)
    m = doggo.mass

    heights = [pos[k][2] for k = 1:n_links]
    V = 0.0
    for k = 1:n_links
        V += m[k]*g*heights[k]
    end
    return V
end

function get_∇V(doggo::Doggo, jac)
    dof = doggo.dof
    ∇V = zeros(dof)
    m = doggo.mass
    for k = 1:num_links(doggo)
        ∇V += m[k]*g*jac[k][2,:]
    end
    return ∇V
end

function lagrangian(doggo::Doggo, q, q̇)
    T = get_T(doggo, q, q̇)
    V = get_V(doggo, q)
    return T - V
end

function grad_L(doggo::Doggo, q, q̇)
    dof = doggo.dof
    grad = zeros(2dof)

    jac, hess = fk_hess(doggo, q)
    C = comm(3,dof)
    dMdq = sum([(kron(speye(dof), jac[k]'M[k])*hess[k] +
                 kron(jac[k]'M[k], speye(dof))*C*hess[k]) for k = 1:4])
    grad[1:dof] = (0.5*kron(q̇', q̇')*dMdq)' - get_∇V(doggo, jac)
    grad[dof+1:2dof] = mass_matrix(doggo, jac)*q̇
    return grad
end

function euler_lagrange(doggo::Doggo, q, qd, qdd)
    dof = doggo.dof
    jac, hess = fk_hess(doggo, q)

    C = comm(3,dof)
    dMdq = sum([(kron(speye(dof), jac[k]'M[k])*hess[k] +
                 kron(jac[k]'M[k], speye(dof))*C*hess[k]) for k = 1:4])
    Mdot = dMdq*qd
    return reshape(Mdot,dof,dof)*q̇ + mass_matrix(doggo, jac)*qdd - (0.5*kron(qd',qd')*dMdq)' + get_∇V(doggo, jac)
end

function euler_lagrange_auto(doggo, q, qd, qdd)
    dof = doggo.dof
    x = [q; qd]
    part = create_partition2((dof,dof),(:x,:v))

    res = DiffResults.HessianResult(x)
    # lagrangian(x) = lagrangian(doggo, x[1:dof], x[dof+1:2dof])
    ForwardDiff.hessian!(res, lagrangian, x)
    L_hess = BlockArray(DiffResults.hessian(res), part)
    L_grad = DiffResults.gradient(res)
    L_hess.vx*q̇ + L_hess.vv*qdd - L_grad[1:2]
end
