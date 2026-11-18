# Theory

All quantities in SI units. Subscripts *m* (matrix) and *r* (reinforcement);
$V_f$ is the reinforcement volume fraction.

## 1. Macro problem

A straight bar of length $L$ and cross-section $A$, clamped at $x = 0$, loaded by a
tip force $P$ and a uniform axial load $q$ (N/m):

$$
\frac{d}{dx}\left(E_\mathrm{eff} A \frac{du}{dx}\right) + q = 0, \qquad
u(0) = 0, \qquad E_\mathrm{eff} A \, u'(L) = P .
$$

Closed form (used to verify the FE solution):

$$
u(x) = \frac{(P + qL)\,x - \tfrac12 q x^2}{E_\mathrm{eff} A}, \qquad
\sigma(x) = \frac{P + q(L - x)}{A}.
$$

**Discretisation.** Two-node linear elements, element stiffness
$k_e = E_e A / h_e \begin{bmatrix}1 & -1\\ -1 & 1\end{bmatrix}$, consistent load
$f_e = q h_e / 2 \,[1, 1]^T$. Constrained DOFs are eliminated, so the reduced system is
symmetric positive definite and solved with a sparse direct solver. In 1D this
scheme is exact at the nodes, and the constant element stress equals the exact
stress at the element midpoint.

## 2. Micro problem (unit cell / RVE)

A periodic cell of length $\ell$ is subjected to an average strain $\bar\varepsilon$.
With periodic boundary conditions the displacement splits into an affine part and a
periodic fluctuation $w$:

$$
u(\xi) = \bar\varepsilon\,\xi + w(\xi), \qquad w(0) = w(\ell),
$$

which in 1D is imposed as $u(0) = 0$, $u(\ell) = \bar\varepsilon \ell$. The cell is
solved with the same FE routine. Then

$$
E_\mathrm{eff} = \frac{\langle\sigma\rangle}{\bar\varepsilon}, \qquad
A_i = \frac{\langle\varepsilon\rangle_i}{\bar\varepsilon}\quad\text{(strain concentration)},
$$

with $\langle\cdot\rangle$ the volume average over the cell and $\langle\cdot\rangle_i$
the average over phase $i$. By construction $\sum_i c_i A_i = 1$.

**Localization.** For each macro element with strain $\varepsilon^M$:
$\varepsilon_i = A_i \varepsilon^M$ and $\sigma_i = E_i A_i \varepsilon^M$.

### Closed-form results in 1D

| Arrangement | Mechanism | $E_\mathrm{eff}$ | $A_i$ |
|---|---|---|---|
| series (layers along the axis) | iso-stress | Reuss: $\left(V_f/E_r + (1-V_f)/E_m\right)^{-1}$ | $E_\mathrm{eff}/E_i$ |
| parallel (phases along the axis) | iso-strain | Voigt: $V_f E_r + (1-V_f) E_m$ | $1$ |

For any two-phase microstructure with the same volume fraction,
$E_\mathrm{Reuss} \le E_\mathrm{eff} \le E_\mathrm{Voigt}$.

The numerical RVE reproduces both rows to machine precision; this is the
first verification check.

## 3. Fully resolved model and scale separation

`solve_resolved` meshes $N$ cells explicitly along the bar (series arrangement).
Because each cell is symmetric and whole, the tip displacement of the resolved model
equals the homogenized one exactly. Inside the bar the resolved displacement
zig-zags around the homogenized curve with an amplitude proportional to the cell size
$L/N$, so the maximum relative difference decays like $1/N$. That is the
**scale-separation** assumption behind homogenization, shown in the
*Verification* tab.

## 4. Assumptions and validity

- Uniaxial stress, small strains, linear elasticity, perfect interfaces.
- Periodic microstructure with a cell much smaller than the bar ($N \gg 1$).
- Strains above about 1 % violate the small-strain assumption for typical
  polymer matrices; the app and CLI warn in that case.
- No failure criteria: the localized phase stresses should be compared with
  the phase strengths separately.
