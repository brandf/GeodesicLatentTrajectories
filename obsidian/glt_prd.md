# Geodesic Latent Trajectories (GLT)
### Project Requirements Document (PRD)

## 1. Overview
Geodesic Latent Trajectories (GLT) is an experimental modification to language‑model pretraining that replaces next‑token prediction in raw hidden‑state space with prediction in a *geometrically structured latent space*. The core idea:

> **Learn a latent space where token trajectories become linear in log‑map coordinates.**  
> **Learn a mapping from token embeddings to a hyperspherical latent space where the sequence follows a near‑geodesic curve.**

Instead of treating the model as learning arbitrarily tangled trajectories in embedding space, GLT introduces a projection into a *smooth, approximately geodesic* latent manifold. This enables prediction through *extrapolation* (extending the local geodesic direction) and potentially offers:

- more stable long‑context behavior  
- a disentangled representation of semantic flow  
- simpler extrapolation for multi‑completion generation  
- improved interpretability of latent dynamics  

This PRD defines the objectives, mathematical formulation, architectural components, and all losses needed to implement GLT inside a nanoGPT/nanoChat‑style training loop.

---

## 2. Key High‑Dimensional Intuitions
This project relies on several properties of high‑dimensional normalized spaces:

### 2.1 Hyperspherical concentration
LayerNorm places vectors on an \(S^{d-1}\) hypersphere.  
In high‑D:

- volume concentrates near the surface  
- magnitudes provide no information  
- directions encode all semantics  

### 2.2 Random vectors are nearly orthogonal
For random \(x, y \in S^{d-1}\):

\[
\mathbb{E}[x\cdot y] = 0,\quad \mathrm{Var}(x\cdot y) = rac{1}{d}
\]

Distances and angles are *very stable*, making angular deviations highly meaningful.

### 2.3 Geodesics on the hypersphere
The natural interpolant between two normalized vectors is SLERP:

\[
\mathrm{slerp}(u,v;	au)=
rac{\sin((1-	au)	heta)}{\sin	heta}u
+rac{\sin(	au	heta)}{\sin	heta}v
\]

where \(	heta=rccos(u\!\cdot\! v)\).

### 2.4 Log/exp maps
On \(S^{d-1}\):

- Log map: maps sphere → tangent space  
- Exp map: maps tangent space → sphere  

They linearize local geodesics.

GLT will exploit these for straightening.

---

## 3. System Architecture

### 3.1 Pipeline
```
token_ids → input embeddings → Transformer → raw hidden states h_t
                                        ↓
                               GLT Head f(h_t)
                                        ↓
                          latent points y_t ∈ S^{D-1}
                                        ↓
                   prediction head → logits → token loss
```

### 3.2 GLT head
A trainable projection:

\[
y_t = rac{W_2\,\sigma(W_1 h_t + b_1) + b_2}{\|W_2\,\sigma(W_1 h_t+b_1)+b_2\|}
\]

to ensure \(y_t \in S^{D-1}\).

### 3.3 Untied embedding/un‑embedding
Decoder projection:

\[
	ext{logits}_t = V y_t + c
\]

where V is unconstrained.

---

## 4. GLT Objectives

We use multiple loss components:

---

### 4.1 Token Prediction Loss (standard cross‑entropy)
\[
\mathcal{L}_{	ext{CE}} = -\log p(x_{t+1}\mid y_t)
\]

---

### 4.2 Local Geodesic Straightness Loss
Use a sliding window of size \(k\) (default 3).

For each triplet:

\[
y_{t-1},\;y_t,\;y_{t+1}
\]

compute predicted midpoint via SLERP:

\[
\hat{y}_t = \mathrm{slerp}(y_{t-1}, y_{t+1}; 0.5)
\]

Loss:

\[
\mathcal{L}_{	ext{local}}=rac{1}{T}\sum_t
ig\| y_t - \hat{y}_t ig\|_2^2
\]

This enforces **local minimal curvature**.

Generalizing beyond midpoint:

Use polynomial‑order extrapolation in log space.

Define:

\[
m_t = \log_{y_t}(y_{t+1}) \in T_{y_t}S^{D-1}
\]

For window \(k\), fit linear model:

\[
m_{t+j} pprox a j + b
\]

Loss:

\[
\mathcal{L}_{	ext{poly}} = \sum_{t} \sum_{j=-k}^k
\| m_{t+j} - (aj + b)\|^2
\]

---

### 4.3 Global Straightness Loss (optional)
Encourage overall curve to follow a near‑geodesic path.

Pick random anchor positions \(s,t\):

\[
\hat{y}_{u}= \mathrm{slerp}(y_s,y_t; 	au=rac{u-s}{t-s})
\]

Loss:

\[
\mathcal{L}_{	ext{global}}
= \mathbb{E}_{s<t}
igg[
rac{1}{t-s} \sum_{u=s}^{t} \|y_u-\hat{y}_{u}\|^2
igg]
\]

Scaling parameter:

- \(\lambda_{	ext{global}} \in [0,1]\)  
- 0 = no global straightness  
- 1 = fully straightened sequences  

---

### 4.4 Constant Angular Spacing Loss
To extrapolate future tokens, we want:

\[
	heta_{t,t+1} pprox 	ext{constant}
\]

Define:

\[
	heta_t = rccos(y_t\cdot y_{t+1})
\]

Loss:

\[
\mathcal{L}_{	ext{angle}} = 
\mathrm{Var}(\{	heta_t\})
\]

---

### 4.5 Bi‑Directional Interpolation Loss
Use reverse context as well:

\[
y_t pprox \mathrm{slerp}(y_{t+1}, y_{t-1}; 0.5)
\]

Equivalently enforces symmetric geodesic consistency.

Define:

\[
\mathcal{L}_{	ext{bi}} = \sum_t 
\| y_t - \mathrm{slerp}(y_{t+1}, y_{t-1}; 0.5)\|^2
\]

Helps stabilize local curvature.

---

## 5. Total Loss

\[
\mathcal{L} = 
\lambda_{	ext{CE}} \mathcal{L}_{	ext{CE}}
+ \lambda_{	ext{local}}\mathcal{L}_{	ext{local}}
+ \lambda_{	ext{global}}\mathcal{L}_{	ext{global}}
+ \lambda_{	ext{angle}}\mathcal{L}_{	ext{angle}}
+ \lambda_{	ext{bi}}\mathcal{L}_{	ext{bi}}
\]

Recommended starting weights:

- \(\lambda_{	ext{CE}} = 1.0\)  
- \(\lambda_{	ext{local}} = 0.3\)
- \(\lambda_{	ext{global}} = 0.05\)
- \(\lambda_{	ext{angle}} = 0.1\)
- \(\lambda_{	ext{bi}} = 0.1\)

---

## 6. Extrapolation for Next‑Token Prediction

Given final latent point \(y_T\):

1. Compute tangent direction  
   \[
   v_T = \log_{y_{T-1}}(y_T)
   \]

2. Extrapolate forward:

   \[
   y_{T+1} = \exp_{y_T}(v_T)
   \]

3. Decode via vocab projection:

   \[
   	ext{logits}_{T+1} = V y_{T+1} + c
   \]

This yields predictions consistent with geodesic continuation.

---

## 7. Parameters & Hyperparameters

### Structural
- latent dimension \(D\)
- sliding window \(k\)
- global straightness coefficient \(\lambda_{	ext{global}}\)

### Loss coefficients
As above.

### Training
- Use standard GPT architecture
- GLT head inserted after transformer blocks
- Decoder uses y_t instead of final hidden state

---

## 8. Deliverables for Coding Agent
1. GLT module implemented in PyTorch  
2. Log/exp map implementations for \(S^{D-1}\)  
3. SLERP and geodesic utilities  
4. Full training script integration  
5. Ablations for each loss component  
6. Visualization tools:  
   - curvature over sequence  
   - angle distributions  
   - latent trajectory plots (PCA/UMAP)  

---

## 9. Success Criteria
- Sequences in latent space show reduced local curvature  
- Predictive performance matches or exceeds baseline nanoGPT  
- Extrapolated latents produce plausible completions  
- Multiple completions achieved by small angular perturbations  

---

## 10. Summary
GLT redefines the next‑token prediction problem by transforming sequence embeddings into a smooth latent manifold where trajectories are *locally geodesic and globally coherent*. By leveraging high‑dimensional spherical geometry, it offers a clean and expressive model of semantic flow.

> **Goal: Build a latent space where the model “thinks” by walking along a gently curving geodesic — instead of fighting tangled trajectories.**

This concludes the PRD.

