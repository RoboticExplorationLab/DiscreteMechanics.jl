G(q) = 2*[-q[2:4] q[1]*I - skew(q[2:4])]

q = Quaternion(rand(Quat))
qd = Quaternion(rand(Quat))
q_ = SVector(q)
G(q_)
R = [zeros(3,1) Diagonal(I,3)]

λ = rand(3)
λhat_ = [0;λ]
λhat = Quaternion(λhat_)
ω = rand(3)
qd = q*Quaternion(0.,ω/2)
qd_ = SVector(qd)

# dGdq
2R*Rmult(qd)*N == -G(qd_)
2λhat_'Rmult(qd)*N == -λ'G(qd_)
2λhat_'Rmult(qd)*N == -λ'G(qd_)
2N*Rmult(qd)'λhat_ == -G(qd_)'λ
2*SVector(qd*inv(λhat)) ≈ -G(qd_)'λ

2λ'R*Lmult(inv(q)) == (2*Lmult(q)*λhat_)'
λ'ω ≈ 2λ'R*Lmult(inv(q))*qd_
λ'ω ≈ 2*λhat_'SVector(inv(q)*qd)
λ'ω
2*qd_'SVector(q*λhat)

λhat_'Rmult(qd)
(qd_'Rmult(λhat))*N

N*qd_

# First term: λ'Gprime(qd)
ForwardDiff.jacobian(x->G(x)*qd_,q_) == -G(qd_)
-λ'G(qd_) == -2λ'R*Lmult(inv(qd))
-λ'G(qd_) ≈ λ'R*Lmult(Quaternion(0.,ω))*Lmult(q)'


# Second term: -λdot'G(q)
λ'G(q_) == 2λ'R*Lmult(inv(q))

# Third term: λ'dGdq*qdot
ForwardDiff.jacobian(x->G(x)*qd_,q_) == -2*R*Lmult(inv(qd))
λ'ForwardDiff.jacobian(x->G(x)*qd_,q_) ≈ λ'R*Lmult(Quaternion(0.,ω))*Lmult(q)'
G(q_) == 2*R*Lmult(inv(q))
-2*SVector(qd*Quaternion(0.,λ)) ≈ 2*[0 λ'; -λ skew(λ)]*qd_
ForwardDiff.gradient(x->λ'G(x)*qd_,q_) == 2*[0 λ'; -λ skew(λ)]*qd_
ForwardDiff.gradient(x->λ'G(x)*qd_,q_)
 #≈ -2*Lmult(qd)*R'λ
-2*Lmult(qd)*R'λ ≈ -Lmult(q)*Lmult(Quaternion(0.,ω))*R'λ
-2*Lmult(qd)*R'λ ≈ (λ'R*Lmult(Quaternion(0.,ω))*Lmult(q)')'
-2*Lmult(qd)*R'λ ≈ (-2*λ'R*Lmult(qd)')'

Lmult(Quaternion(0.,ω/2))'Lmult(q)' ≈ Lmult(inv(qd))

Lmult(inv(q))*q_

-2*SVector(qd*Quaternion(0.,λ))

Lmult(q)' == Lmult(inv(q))

Lmult(q)*[0;ω/2]
Lmult(inv(q*Quaternion(0.,ω/2)))


2λ'R*Lmult(inv(q))
2λ'R*SVector(inv(q))

N = Diagonal([1,-1,-1,-1])
N*q_
