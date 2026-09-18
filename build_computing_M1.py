#!/usr/bin/env python3
"""
Builds `computing_M1.ipynb`.

Edit this file, run it, and re-execute the notebook:

    python3 build_computing_M1.py
    jupyter nbconvert --to notebook --execute --inplace computing_M1.ipynb
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
co = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# First-order geometric sources $M^{(1)}_a$

Symbolic derivation of the geometric (roughness) sources of Section 4 of the paper:
the terms that the coordinate change puts on the right-hand side of the transformed
Helmholtz equation at first order in the surface profile.

The coordinate change is

$$z = \zeta + \gamma(\vec x,\zeta)\,,\qquad \gamma(\vec x,\zeta)=h(\vec x)\,f(\zeta)\,,$$

with the gauge conditions of eq. (2.6),

$$f(0)=1\,,\qquad f'(0)=0\,,\qquad f(\zeta)\to0 \ \text{ as }\ \zeta\to\pm\infty\,.$$

Writing the metric as the identity plus corrections ordered by the number of
derivatives of $\gamma$, the first-order current is eq. (4.1),

$$M^{(1)}_a=\Big(\delta_{ab}G^{(1)}_{de}+G^{(1)}_{ab}\delta_{de}\Big)\,
\epsilon^{bcd}\epsilon^{efg}\,\partial_c\partial_f E^{(0)}_g
\;+\;\delta_{ab}\,\epsilon^{bcd}\big(\partial_cG^{(1)}_{de}\big)\,
\epsilon^{efg}\,\partial_f E^{(0)}_g\,,$$

where $G^{(1)}_{ab}=m^{(1)}_{ab}-M^{(1)}\delta_{ab}$, $M^{(1)}=\gamma_{,\zeta}$ and

$$m^{(1)}=\begin{pmatrix}0&0&\gamma_{,x}\\0&0&\gamma_{,y}\\
\gamma_{,x}&\gamma_{,y}&2\gamma_{,\zeta}\end{pmatrix}\,.$$

The notebook evaluates that contraction for TE and TM incidence and transforms the
result to momentum space, giving eqs. (4.2)–(4.4) and (4.6)–(4.8), and then their
backscattering limit, which is the integrand of Appendix A.

Throughout, `z` stands for the transformed coordinate $\zeta$.
""")

md("## 1 · Setup")

co(r"""
import sympy as sp
from sympy import (symbols, Function, Array, KroneckerDelta, LeviCivita,
                   diff, simplify, expand, exp, I, S)

sp.init_printing(use_unicode=True)

x, y, z = symbols('x y z')                   # z is the transformed coordinate, zeta
X = [x, y, z]
k, px, py = symbols('k p_x p_y', real=True)  # incident and scattered horizontal wavevectors
H = symbols('H')                             # Fourier amplitude of the surface, H(p - k)

h = Function('h')(x, y)                      # surface profile
f = Function('f')(z)                         # gauge profile
gamma = h*f
print("ready")
""")

md(r"""
## 2 · Metric perturbation and the first-order current

$G^{(1)}$ collects the first-order part of the metric; $M^{(1)}_a$ follows from it by
the double Levi-Civita contraction of eq. (4.1). The zeroth-order field is left as a
generic $\vec E^{(0)}$ with no $y$-dependence, which is the configuration used in the
paper: the incident wavevector is $\vec k=k\,\hat x$.
""")

co(r"""
def G1_tensor():
    '''First-order metric perturbation  G^(1)_ab = m^(1)_ab - gamma_,z delta_ab.'''
    gx, gy, gz = diff(gamma, x), diff(gamma, y), diff(gamma, z)
    m1 = Array([[S.Zero, S.Zero, gx],
                [S.Zero, S.Zero, gy],
                [gx,     gy,     2*gz]])
    ident = Array([[KroneckerDelta(i, j) for j in range(3)] for i in range(3)])
    return m1 - gz*ident


def M1(a, E0):
    '''Component a of the first-order geometric source, eq. (4.1).'''
    G = G1_tensor()
    tot = 0
    for b in range(3):
        for c in range(3):
            for d in range(3):
                if LeviCivita(b, c, d) == 0:
                    continue
                for e in range(3):
                    for ff in range(3):
                        for g in range(3):
                            if LeviCivita(e, ff, g) == 0:
                                continue
                            lc = LeviCivita(b, c, d)*LeviCivita(e, ff, g)
                            d1 = diff(E0[g], X[ff])
                            d2 = diff(diff(E0[g], X[c]), X[ff])
                            tot += KroneckerDelta(a, b)*lc*diff(G[d, e], X[c])*d1
                            tot += KroneckerDelta(a, b)*lc*G[d, e]*d2
                            tot += G[a, b]*lc*KroneckerDelta(d, e)*d2
    return expand(tot)
print("ready")
""")

md(r"""
## 3 · Zeroth-order fields

For TE incidence the field is transverse, $\vec E^{(0)}=\hat y\,\varphi^{(0)}$ with
$\varphi^{(0)}=e^{\imath\vec k\cdot\vec x}\Phi^{(0)}(\zeta)$, and $\Phi^{(0)}$ is the
flat-interface solution of Section 3.

For TM incidence the field has $(\hat x,\hat\zeta)$ components,
$E^{(0)}_x=e^{\imath\vec k\cdot\vec x}A(\zeta)$ and
$E^{(0)}_\zeta=e^{\imath\vec k\cdot\vec x}B(\zeta)$.
""")

co(r"""
Phi = Function('Phi')(z)        # TE:  Phi^(0)(zeta)
A   = Function('A')(z)          # TM:  E_x^(0)     = e^{ikx} A(zeta)
B   = Function('B')(z)          # TM:  E_zeta^(0)  = e^{ikx} B(zeta)

E0_TE = [S.Zero,        exp(I*k*x)*Phi, S.Zero]
E0_TM = [exp(I*k*x)*A,  S.Zero,         exp(I*k*x)*B]

M_TE = [M1(a, E0_TE) for a in range(3)]
M_TM = [M1(a, E0_TM) for a in range(3)]
print("computed")
""")

md(r"""
## 4 · Real space

Collected in the derivatives of $\gamma$. The TM components involve the zeroth-order
field only through the combination
$\partial_\zeta E^{(0)}_x-\partial_x E^{(0)}_\zeta$, which is the auxiliary field
$\mathcal{H}^{(0)}$ of eq. (4.5), proportional to the incident magnetic field
$H^{(0)}_y$. That this particular combination is the one that appears is an output of
the contraction, not an assumption, and it is what makes $\mathcal{H}^{(0)}$ the
natural variable for the TM channel.
""")

co(r"""
# plain symbols for the derivatives, so the output is readable
names = ['x', 'y', 'z']
gsym, grep = {}, {}
for i, ni in enumerate(names):
    sy = sp.Symbol(f'gamma_{ni}')
    gsym[ni] = sy
    grep[diff(gamma, X[i])] = sy
    for j, nj in enumerate(names[i:], start=i):
        sy2 = sp.Symbol(f'gamma_{ni}{nj}')
        grep[diff(diff(gamma, X[i]), X[j])] = sy2
        gsym[ni + nj] = sy2

def show_real(M, label):
    for a, nm in enumerate('xyz'):
        e = expand(M[a]/exp(I*k*x)).subs(grep, simultaneous=True)
        print(f"\n  M1_{nm}  ({label})")
        sp.pprint(sp.collect(expand(e), list(gsym.values())))

show_real(M_TE, 'TE')
show_real(M_TM, 'TM')
""")

md(r"""
## 5 · Momentum space

The sources are bilinear in the surface and in the zeroth-order field. Since
$\vec E^{(0)}\propto e^{\imath\vec k\cdot\vec x}$, the factor of $h$ in a source
evaluated at the scattered wavevector $\vec p$ carries $\vec p-\vec k$; with
$\vec k=k\,\hat x$ this gives

$$\partial_x^m\partial_y^n h \;\longrightarrow\;
\big[\imath(p_x-k)\big]^m\,(\imath p_y)^n\,H(\vec p-\vec k)\,,$$

while derivatives of $e^{\imath k x}$ give factors of $\imath k$. No factor of $2\pi$
survives: the convention of eq. (2.1) puts $(2\pi)^{-2}$ on the forward transform, so
that the convolution of two fields carries none, which is eq. (2.2).
""")

co(r"""
def to_fourier(expr):
    '''Strip e^{ikx} and replace derivatives of h by powers of i(p-k).'''
    e = expand(expr/exp(I*k*x))
    reps = {}
    for m in range(4):
        for n in range(4):
            if 0 < m + n < 4:
                reps[diff(h, *([x]*m + [y]*n))] = (I*(px - k))**m * (I*py)**n * H
    reps[h] = H
    return sp.simplify(expand(e.subs(reps, simultaneous=True)))

MF_TE = [to_fourier(M_TE[a]) for a in range(3)]
MF_TM = [to_fourier(M_TM[a]) for a in range(3)]
print("transformed")
""")

md(r"### TE incidence — eqs. (4.2), (4.3) and (4.4)")

co(r"""
for a, nm, eq in zip(range(3), 'xyz', ['(4.2)', '(4.3)', '(4.4)']):
    print(f"\n  M1_{nm}(p, zeta)   TE        {eq}")
    sp.pprint(sp.collect(expand(MF_TE[a]), H))
""")

md(r"""
### TM incidence — eqs. (4.6), (4.7) and (4.8)

Written in terms of $\Psi^{(0)}$ by eliminating $A$ in favour of
$\Psi^{(0)}=A'-\imath kB$, the momentum-space form of the auxiliary field (4.5).
""")

co(r"""
Psi = Function('Psi')(z)
sub_Psi = {diff(A, z, 3): diff(Psi, z, 2) + I*k*diff(B, z, 2),
           diff(A, z, 2): diff(Psi, z)    + I*k*diff(B, z),
           diff(A, z, 1): Psi             + I*k*B}

MF_TM_psi = [sp.simplify(expand(MF_TM[a].subs(sub_Psi, simultaneous=True)))
             for a in range(3)]

for a, nm, eq in zip(range(3), 'xyz', ['(4.6)', '(4.7)', '(4.8)']):
    print(f"\n  M1_{nm}(p, zeta)   TM        {eq}")
    sp.pprint(sp.collect(expand(MF_TM_psi[a]), H))
""")

md(r"""
## 6 · Backscattering

Setting $p_x=-k$ and $p_y=0$, the limit used throughout Section 5 and Appendix A.

In TE only the $y$ component survives, and the term carrying $(k^2-p_x^2)$ in (4.3)
drops out, leaving the two-term integrand of Appendix A.1. In TM only the $x$ and
$\zeta$ components survive, and (4.6) reduces to the combination
$\big[2k^2f+f''\big]\Psi^{(0)}+2f'\Psi^{(0)}_{,\zeta}$ that enters $I_\parallel$ in
Appendix A.2.
""")

co(r"""
bs = {px: -k, py: 0}

for label, table in (('TE', {nm: MF_TE[a] for a, nm in enumerate('xyz')}),
                     ('TM', {nm: MF_TM_psi[a] for a, nm in enumerate('xyz')})):
    print(f"\n{label}   (p_x = -k,  p_y = 0)")
    for nm in 'xyz':
        e = sp.simplify(expand(table[nm].subs(bs)))
        print(f"\n  M1_{nm}")
        sp.pprint(e)
""")

md(r"""
---

### Summary

The contraction of eq. (4.1) gives, in momentum space:

* **TE** — eqs. (4.2), (4.3) and (4.4), built on $\Phi^{(0)}$ and its normal
  derivatives.
* **TM** — eqs. (4.6), (4.7) and (4.8), which depend on the zeroth-order field only
  through the auxiliary field
  $\Psi^{(0)}=\partial_\zeta E^{(0)}_x-\partial_x E^{(0)}_\zeta$ of eq. (4.5).

In backscattering these reduce to the two-term integrands that Appendix A evaluates
exactly, and from which the geometric amplitudes $I_{HH}$ and $I_{VV}$ follow.
""")

nb["cells"] = C
nb["metadata"] = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                 "name": "python3"},
                  "language_info": {"name": "python", "version": "3.12"}}
nbf.write(nb, "computing_M1.ipynb")
print(f"wrote computing_M1.ipynb  ({len(C)} cells)")
