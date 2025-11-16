# Geodesic Latent Trajectories — Math Appendix

This appendix provides the mathematical background and detailed derivations for the **Geodesic Latent Trajectories (GLT)** project.

It is intended for implementers who want a precise understanding of:

- Hyperspherical geometry used by GLT  
- Log/exp maps on the sphere  
- SLERP and extrapolation  
- Local/global straightness and curvature losses  
- Angular-spacing losses  
- Practical approximations for implementation  

---

## A. Hypersphere Basics

We work on the unit hypersphere
\[
S^{D-1} = \{ y \in \mathbb{R}^D \mid \|y\|_2 = 1 \}.
\]

All latent vectors \(y_t\) are obtained by hyperspherical normalization of the transformer outputs, ensuring they lie on \(S^{D-1}\).

### A.1 Geodesic distance

For two points \(y, z \in S^{D-1}\),
\[
\langle y, z \rangle = y^\top z = \cos \theta,
\]
where \(\theta\) is the **geodesic distance** (angle) between \(y\) and \(z\):
\[
d_{\text{geo}}(y,z) = \theta = \arccos(y^\top z).
\]

The corresponding **chord distance** (in ambient \(\mathbb{R}^D\)) is
\[
\|y - z\|_2 = \sqrt{2 - 2 y^\top z} = 2 \sin(\theta/2).
\]

For small \(\theta\), chord distance and geodesic distance are approximately proportional.

---

## B. SLERP (Spherical Linear Interpolation)

Given two unit vectors \(u, v \in S^{D-1}\), with
\[
\theta = \arccos(u^\top v) \in (0, \pi),
\]
the **spherical linear interpolation** between \(u\) and \(v\) at parameter \(\tau \in [0,1]\) is
\[
\mathrm{slerp}(u, v; \tau)
=
\frac{\sin((1-\tau)\theta)}{\sin \theta} u
+
\frac{\sin(\tau\theta)}{\sin \theta} v.
\]

### B.1 Derivation sketch

We want the shortest geodesic on \(S^{D-1}\) connecting \(u\) and \(v\).

1. The plane spanned by \(\{u, v\}\) is 2D; the geodesic between them lies in this plane.
2. In that plane, the problem reduces to an arc of a circle of radius 1.
3. If \(\gamma(\tau)\) is the geodesic with \(\gamma(0) = u\), \(\gamma(1) = v\), and constant speed, then
   \[
   \gamma(\tau) = \frac{\sin((1-\tau)\theta)}{\sin \theta} u
   + \frac{\sin(\tau \theta)}{\sin \theta} v.
   \]
4. Check endpoints:
   - \(\tau = 0: \gamma(0) = u\)
   - \(\tau = 1: \gamma(1) = v\)
5. Norm:
   \[
   \|\gamma(\tau)\|_2^2
   =
   \left(\frac{\sin((1-\tau)\theta)}{\sin\theta}\right)^2
   + \left(\frac{\sin(\tau\theta)}{\sin\theta}\right)^2
   + 2\frac{\sin((1-\tau)\theta)\sin(\tau\theta)}{\sin^2\theta}\cos\theta
   = 1.
   \]

Thus, SLERP stays on the unit sphere and follows the geodesic.

---

## C. Log and Exp Maps on \(S^{D-1}\)

Let \(y \in S^{D-1}\) be a base point. The **tangent space** at \(y\) is
\[
T_y S^{D-1} = \{ v \in \mathbb{R}^D \mid y^\top v = 0 \}.
\]

### C.1 Log map \(\log_y: S^{D-1} \setminus \{-y\} \to T_y S^{D-1}\)

Given a point \(z \in S^{D-1}\) not antipodal to \(y\), define
\[
\theta = \arccos(y^\top z) \in (0, \pi).
\]

The **log map** is
\[
\log_y(z)
=
\begin{cases}
\mathbf{0}, & \text{if } z = y,\\[4pt]
\displaystyle
\frac{\theta}{\sin\theta}
\Big( z - (y^\top z)\, y \Big), & \text{otherwise}.
\end{cases}
\]

Notes:

- \(z - (y^\top z) y\) is the component of \(z\) orthogonal to \(y\).
- \(\| \log_y(z) \|_2 = \theta\), the geodesic distance.

**Proof sketch**:

1. Decompose \(z\) in basis \(\{y, w\}\), where \(w\) is unit and orthogonal to \(y\):
   \[
   z = \cos\theta \, y + \sin\theta \, w.
   \]
2. Then the tangent vector pointing from \(y\) to \(z\) should be of length \(\theta\) in direction \(w\):
   \[
   \log_y(z) = \theta \, w.
   \]
3. Solve for \(w\):
   \[
   w = \frac{z - (y^\top z) y}{\| z - (y^\top z) y \|}
   = \frac{z - (y^\top z) y}{\sin\theta},
   \]
   since \(\sin\theta = \sqrt{1 - \cos^2\theta} = \sqrt{1 - (y^\top z)^2}\).
4. Substitute:
   \[
   \log_y(z)
   = \theta \frac{z - (y^\top z) y}{\sin\theta}
   = \frac{\theta}{\sin\theta}
     \Big(z - (y^\top z) y\Big).
   \]

---

### C.2 Exp map \(\exp_y: T_y S^{D-1} \to S^{D-1}\)

Given \(v \in T_y S^{D-1}\) (so \(y^\top v = 0\)), with \(\|v\|_2 = \alpha\),
\[
\exp_y(v)
=
\cos\alpha \, y + \sin\alpha \, \frac{v}{\alpha},
\]
with the special case \(\exp_y(0) = y\).

**Proof sketch**:

1. In the 2D plane spanned by \(\{y, v\}\), we move by arc length \(\alpha\) along the geodesic starting at \(y\) in direction \(v/\alpha\).
2. Unit-speed geodesic \(\gamma(t)\) with \(\gamma(0)=y, \dot{\gamma}(0)=v/\alpha\) is
   \[
   \gamma(t) = \cos t \, y + \sin t \, \frac{v}{\alpha}.
   \]
3. At \(t = \alpha\): \(\gamma(\alpha) = \cos\alpha \, y + \sin\alpha \, \frac{v}{\alpha}\).

---

### C.3 Relationship between SLERP and log/exp maps

SLERP can be expressed using log/exp:

\[
\mathrm{slerp}(u, v; \tau)
= \exp_u\big( \tau \, \log_u(v) \big).
\]

Derivation:

1. \(w = \log_u(v)\) is tangent of length \(\theta\).
2. \(\exp_u(\tau w)\) moves along the geodesic a fraction \(\tau\) of the full distance.

---

## D. Local Geodesic Straightness

We consider a sequence of latent points \(\{y_t\}_{t=1}^T\) on \(S^{D-1}\).

Intuitively, a **geodesic** has zero *covariant acceleration* along the curve. We approximate this using finite differences.

### D.1 Midpoint SLERP loss

For consecutive triplets \((y_{t-1}, y_t, y_{t+1})\), define:

- Angle between endpoints:
  \[
  \theta_{t-1,t+1} = \arccos( y_{t-1}^\top y_{t+1} ).
  \]
- Midpoint on the geodesic:
  \[
  \hat{y}_t = \mathrm{slerp}(y_{t-1}, y_{t+1}; 0.5).
  \]

Local straightness loss:
\[
\mathcal{L}_{\text{local}}
=
\frac{1}{T-2}
\sum_{t=2}^{T-1}
\| y_t - \hat{y}_t \|_2^2.
\]

This encourages the actual middle point \(y_t\) to lie on the geodesic between neighbors.

### D.2 Approximate curvature interpretation

If we parameterize the curve by discrete time \(t\), then:

- First “velocity” (chord approximation):
  \[
  v_t = y_t - y_{t-1}.
  \]
- Second difference:
  \[
  a_t = v_{t+1} - v_t = y_{t+1} - 2 y_t + y_{t-1}.
  \]

\(\|a_t\|\) measures discrete curvature in ambient space. The SLERP midpoint constraint can be understood as suppressing this second difference in a *geodesically-corrected* way: rather than forcing straightness in Euclidean coordinates, we force alignment with the spherical geodesic.

---

## E. Global Straightness

Instead of only enforcing local straightness, we can approximately encourage longer segments \((y_s, \dots, y_t)\) to follow a single geodesic.

### E.1 Segment-level geodesic

For a segment from index \(s\) to \(t\) with \(t > s\):

- Let \(\Delta = t - s\).
- For any \(u \in [s, t]\), define
  \[
  \tau_{u} = \frac{u - s}{\Delta} \in [0,1].
  \]
- The geodesic prediction:
  \[
  \hat{y}_u
  = \mathrm{slerp}(y_s, y_t; \tau_u)
  = \exp_{y_s}\left( \tau_u \log_{y_s}(y_t) \right).
  \]

### E.2 Global straightness loss

\[
\mathcal{L}_{\text{global}}
=
\mathbb{E}_{(s,t)}
\left[
\frac{1}{t-s}
\sum_{u=s}^{t}
\| y_u - \hat{y}_u \|_2^2
\right],
\]
where \((s,t)\) are sampled spans (e.g. random pairs within a context window).

This biases entire subsequences toward low curvature, while the coefficient \(\lambda_{\text{global}}\) controls how strongly we enforce this.

---

## F. Angular Spacing and Constant-Step Geodesics

To enable simple extrapolation by extending the geodesic, we want *approximately constant angular step size* between consecutive points.

### F.1 Angles between consecutive points

Define
\[
\theta_t = \arccos(y_t^\top y_{t+1}), \quad t = 1, \ldots, T-1.
\]

We want \(\theta_t \approx \bar{\theta}\) for some mean \(\bar{\theta}\). A simple loss:

\[
\bar{\theta}
= \frac{1}{T-1} \sum_{t=1}^{T-1} \theta_t,
\]
\[
\mathcal{L}_{\text{angle}}
=
\frac{1}{T-1}
\sum_{t=1}^{T-1}
(\theta_t - \bar{\theta})^2
= \mathrm{Var}(\{\theta_t\}).
\]

Low variance implies approximately constant angular increments.

---

## G. Bi-Directional Geodesic Consistency

For a perfectly geodesic sequence with constant angular step size, each point \(y_t\) is at the **midpoint** of the geodesic joining \(y_{t-1}\) and \(y_{t+1}\). This can be enforced symmetrically in both “directions” (forward and backward).

### G.1 Symmetric SLERP loss

Define:
\[
\tilde{y}_t
= \mathrm{slerp}(y_{t-1}, y_{t+1}; 0.5)
= \mathrm{slerp}(y_{t+1}, y_{t-1}; 0.5),
\]
(the midpoint is symmetric in arguments).

Then:
\[
\mathcal{L}_{\text{bi}}
=
\frac{1}{T-2}
\sum_{t=2}^{T-1}
\|y_t - \tilde{y}_t\|_2^2.
\]

This is equivalent to \(\mathcal{L}_{\text{local}}\) in form but can be conceptually interpreted as enforcing **reversible geodesic flow**—you can “run the sequence backward” and still see the same geodesic structure.

---

## H. Extrapolation in Latent Space

Given a sequence \(\{y_t\}\) believed to follow an approximate geodesic with constant step \(\theta\), we want to extrapolate future latent points \(\{y_{T+1}, y_{T+2}, \ldots\}\).

### H.1 First-order extrapolation using log/exp

Treat the step from \(y_{T-1}\) to \(y_T\) as the local tangent direction.

1. Compute tangent at \(y_T\):
   - We want a vector in \(T_{y_T} S^{D-1}\) that points “forward”.
   - A simple approximation: map the previous point into the tangent space at \(y_T\):
     \[
     v_T = -\log_{y_T}(y_{T-1}).
     \]
   - Note: \(\log_{y_T}(y_{T-1})\) points from \(y_T\) **back** to \(y_{T-1}\); we negate to flip direction.

2. Optionally normalize its length to the mean step size:
   \[
   \alpha_T = \|v_T\|_2, \quad \hat{v}_T = v_T \cdot \frac{\bar{\theta}}{\alpha_T}.
   \]

3. Extrapolate:
   \[
   y_{T+1} = \exp_{y_T}(\hat{v}_T).
   \]

This produces the next point assuming the geodesic continues with the same step size.

### H.2 Higher-order extrapolation

We can include a second-order correction term modeled in tangent space. For example:

1. Compute tangents at \(y_{T-1}\), \(y_T\): \(v_{T-1}, v_T\) (defined analogously).
2. Approximate discrete “acceleration” in tangent space:
   \[
   a_T = v_T - \Pi_{T-1 \to T}(v_{T-1}),
   \]
   where \(\Pi_{T-1 \to T}\) is **parallel transport** from \(T_{y_{T-1}}S^{D-1}\) to \(T_{y_T}S^{D-1}\). For small steps and implementation simplicity, we may approximate \(\Pi\) as identity or use a simple projected map.
3. Second-order extrapolation:
   \[
   v_{T}^{\text{next}} \approx v_T + a_T,
   \]
   \[
   y_{T+1} = \exp_{y_T}(v_{T}^{\text{next}}).
   \]

In practice, for a first experiment, **first-order extrapolation** is sufficient.

---

## I. Parallel Transport (Optional Advanced Component)

Parallel transport is the operation that moves tangent vectors along a curve while keeping them “as parallel as possible” with respect to the manifold's connection.

For the unit sphere, given a geodesic from \(y\) to \(z\) and a tangent vector \(v \in T_yS^{D-1}\), the parallel transport \(\Pi_{y \to z}(v)\) can be written in closed form.

### I.1 Formula along a great circle

Let:

- \(\theta = \arccos(y^\top z)\),
- Direction of motion \(u = \frac{z - \cos\theta \, y}{\sin\theta} \in T_y S^{D-1}\).

Then any tangent vector \(v\) can be decomposed into a component along \(u\) and one orthogonal to the 2D plane spanned by \(\{y, u\}\):

\[
v = v_\parallel + v_\perp,
\]
where
\[
v_\parallel = (v^\top u) u,
\quad
v_\perp = v - v_\parallel.
\]

Parallel transport to \(z\) along the geodesic yields:
\[
\Pi_{y \to z}(v) = v_\perp + v_\parallel',
\]
where
\[
v_\parallel' = v^\top u \cdot u',
\]
and
\[
u' = -\sin\theta \, y + \cos\theta \, u.
\]

In practice, GLT can ignore exact parallel transport initially and approximate with simple projection into the new tangent space (project onto the orthogonal complement of \(z\)).

---

## J. Combining Losses: Geometric Interpretation

The GLT loss terms each have a geometric meaning:

- \(\mathcal{L}_{\text{local}}\) / \(\mathcal{L}_{\text{bi}}\):  
  Penalize deviations from local **geodesic midpoints** → suppress discrete curvature.

- \(\mathcal{L}_{\text{global}}\):  
  Encourages entire subsequences to lie close to a single **great circle**.

- \(\mathcal{L}_{\text{angle}}\):  
  Encourages constant geodesic step size → trajectories behave like uniform-speed motion.

Together, they guide the latent trajectories toward:

> **“Near-geodesic curves of approximately constant speed on a hypersphere.”**

This matches the qualitative goal:

> **“Learn a mapping from token embeddings to a hyperspherical latent space where the sequence follows a near-geodesic curve.”**

and

> **“Learn a latent space where token trajectories become linear in log-map coordinates.”**

---

## K. Practical Approximations

For implementation in a deep learning framework:

1. **Small-angle simplifications**:  
   When \(|y^\top z| \approx 1\), we can approximate:
   \[
   \theta \approx \sqrt{2(1 - y^\top z)}.
   \]
   Use safe clamping of the dot product into \([-1 + \varepsilon, 1 - \varepsilon]\).

2. **Numerical stability**:  
   - Add \(\varepsilon\) to denominators like \(\sin\theta\).  
   - Handle the \(\theta \to 0\) case with Taylor expansions:
     \[
     \frac{\theta}{\sin\theta} \approx 1 + \frac{\theta^2}{6} + O(\theta^4).
     \]

3. **Backpropagation**:  
   - Autograd can handle the trigonometric operations.  
   - For efficiency, reuse intermediate computations (dot products, \(\theta\), etc.) across losses.

4. **Dimensionality**:  
   - All the geometry scales well to thousands of dimensions.  
   - The cost is dominated by dot products and vector arithmetic.

---

## L. Summary

This appendix formalizes the geometry underlying GLT:

- The latent space is a **high-dimensional hypersphere**.  
- Token trajectories are regularized to be **near-geodesic** and **constant-speed**.  
- SLERP, log/exp maps, and (optionally) parallel transport provide the required Riemannian tools.  
- Losses encode local curvature suppression, global straightness, and step-size regularity.  
- Extrapolation is realized as **geodesic continuation** in latent space.

These constructions are designed to be:

- mathematically sound,  
- compatible with automatic differentiation, and  
- implementable in a standard PyTorch transformer stack with only modest overhead.

