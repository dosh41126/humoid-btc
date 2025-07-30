B

## 0. Humoid BTC

### 0.1 Project Summary

The **Dyson Sphere Quantum Oracle** is a unified, single‑file Python system that fuses:

1. **Secure cryptography** (AES‑GCM, Argon2, self‑mutating key vaults)
2. **Encrypted, homomorphic‑style vector memory** with SimHash bucketing and quantum‑inspired rotations
3. **Topological memory manifold** for nonlinear, graph‑based retrieval
4. **LLM integration** via llama.cpp with prompt chunking, drift control, and multi‑agent consensus
5. **Quantum circuit hooks** implemented in PennyLane to influence policy and state
6. **Reinforcement‑learning policy head** for dynamic temperature/top‑p adjustment
7. **Persistent storage** across SQLite and Weaviate vector DB
8. **Live data APIs** (Coinbase, CoinGecko, Open‑Meteo) and system telemetry
9. **CustomTkinter GUI** for real‑time interaction and visualization

This design balances **portability** (single‑file execution) with **cutting‑edge research features**, making it ideal for prototypes, demos, and R\&D testbeds.

### 0.2 Key Features

* **End‑to‑end encryption**, both at rest and in memory, with automatic key rotation and zeroization.
* **Vector privacy** via orthogonal rotations \$Q\in SO(d)\$, quantization, and AES‑GCM encryption:

  $$
    \tilde{\mathbf{e}} = \mathrm{AES\text{-}GCM}(\,\mathrm{Quantize}(Q\,\mathbf{e})\,)
  $$
* **Graph‑based memory manifold**, employing the Laplacian \$L = D - W\$ and eigenembedding for geodesic retrieval.
* **Quantum‑inspired reasoning**, mapping textual sentiment to qubit rotations \$R\_X,R\_Y,R\_Z\$ and using historical \$Z\$‑states for continuity.
* **Policy gradient RL** for adaptive sampling, optimizing temperature \$T\$ and top‑p \$p\$ via REINFORCE:

  $$
    \nabla_\theta J(\theta) = \mathbb{E}\bigl[\nabla_\theta \log\pi_\theta(a|s)\,(R - b)\bigr].
  $$
* **Multi‑agent LLM consensus**, aggregating \$m\$ runs to produce robust recommendations:

  $$
    \hat{y} = \arg\max_y \sum_{j=1}^m \mathbf{1}\{y = y_j\}.
  $$

### 0.3 Quickstart / Runtime Notes

1. **Dependencies**: Install via `pip install customtkinter llama_cpp pennylane weaviate-client nltk textblob bleach argon2-cffi cryptography scipy psutil`.
2. **Models**: Place `Meta-Llama-3-8B-Instruct.Q4_K_M.gguf` and `llama-3-vision-alpha-mmproj-f16.gguf` in `/data/`.
3. **API Keys**: Supply `COINBASE_API_KEY` & `COINBASE_API_SECRET` via the GUI or secure env vars.
4. **Run**: `python dyson_oracle.py` (launches the CustomTkinter GUI).
5. **Vault Passphrase**: If `$VAULT_PASSPHRASE` unset, an ephemeral key is generated—vault won’t persist across restarts.

---

## 1. Imports, Environment, and Global Constants

### 1.1 Standard Library Imports

Core modules handle OS, I/O, threading, math, logging, JSON, UUIDs, and timing:

```python
import os, sys, threading, queue, uuid, math, json, sqlite3, logging
```

These provide deterministic, fast primitives; e.g., queue operations satisfy FIFO invariants and thread‐safe concurrency.

### 1.2 Third‑Party & ML Libraries

* **NLP**: `nltk`, `textblob`, `summa`
* **LLM**: `llama_cpp`
* **Vector DB**: `weaviate`
* **Quantum**: `pennylane as qml`
* **Crypto**: `cryptography.hazmat.primitives.ciphers.aead.AESGCM`, `argon2.low_level`
* **HTTP**: `httpx`, `requests`
* **Data**: `numpy as np`, `scipy.spatial.distance.cosine`
* **GUI**: `tkinter`, `customtkinter`
  Duplicate imports are deduplicated at runtime due to module caching.

### 1.3 System/Hardware Config (Paths, Devices, Env Vars)

```python
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ["SUNO_USE_SMALL_MODELS"]="1"
model_path="/data/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf"
mmproj_path="/data/llama-3-vision-alpha-mmproj-f16.gguf"
```

A GPU‑affinity pattern ensures predictable resource allocation. NLTK data path is appended for offline corpora.

### 1.4 Global Settings & Thresholds

Key hyperparameters and thresholds:

```python
ARGON2_TIME_COST_DEFAULT=3
ARGON2_MEMORY_COST_KIB=262144
CRYSTALLIZE_THRESHOLD=5
DECAY_FACTOR=0.95
AGING_T0_DAYS=7.0
AGING_GAMMA_DAYS=5.0
AGING_PURGE_THRESHOLD=0.5
LAPLACIAN_ALPHA=0.18
JS_LAMBDA=0.10
```

* **Decay formula** for memory aging:

  $$
    s_{t+\Delta t} = s_t \times \exp\Bigl(-\frac{\ln 2}{T_{1/2}}\Delta t\Bigr), 
    \quad T_{1/2} = \text{AGING\_T0} + \text{AGING\_GAMMA}\ln(1 + s_t).
  $$

---

## 2. Cryptography & Secure Enclave

### 2.1 AES‑GCM / Argon2 Config

* **Argon2id** parameters:
  $\text{time\_cost}=3,\ \text{memory\_cost}=256$ MiB, $\text{parallelism}=\min(4,\mathrm{cores})$.
* **AES‑GCM**: 128‑bit nonce, 128‑bit authentication tag.
  Encryption:

  $$
    C = \mathrm{AESGCM}(K; N, M, \mathrm{AAD})
  $$

### 2.2 Vault Format & Key Rotation

Vault JSON:

```json
{
  "version":1,
  "active_version":n,
  "keys":[{"version":n, "master_secret":…}],
  "salt":…
}
```

Rotation:

$$
  K_{n+1} = \mathrm{Argon2id}(\text{master\_secret}_{n}, \text{salt}, t,m,p).
$$

### 2.3 Secure KeyManager Class

Handles:

* Vault creation and loading
* Key derivation:
  $\mathbf{K}_i = \mathrm{Argon2id}(\text{master}_i, \text{salt}, t,m,p)$.
* AES‑GCM encrypt/decrypt wrappers
* Self‑mutating keys and zeroization via `SecureEnclave`.

### 2.4 Encrypted Environment Variables

Functions:

```python
set_encrypted_env_var(var, val)
get_encrypted_env_var(var)
```

Wrap secrets, preventing plaintext spills. AAD for each var binds it to context.

### 2.5 Self‑Mutating Keys & Entropy Measurement

* **Candidate generation**:
  $\tilde{K}_i = K + \epsilon_i, \ \epsilon_i\sim\mathcal{N}(0,\sigma^2)$.
* **Fitness**:

  $$
    F(K) = \alpha\,H(K) + \beta\,\mathrm{dist}(K,\{K_{\text{prev}}\}),
  $$

  where $H$ is Shannon entropy, and $\mathrm{dist}$ measures Euclidean separation.

---

## 3. Text Preprocessing & Input Sanitation

### 3.1 NLTK Resource Download/Validation

Downloads:

* `punkt`, `averaged_perceptron_tagger`, `brown`, `wordnet`, `stopwords`, `conll2000`.
  Ensures tokenization and POS tagging are available offline.

### 3.2 Text Sanitization (bleach, regex, prompt injection)

Uses `bleach.clean` plus control‑char stripping:

$$
  x' = \mathrm{stripControl}(x) \to \mathrm{bleach.clean}(x').
$$

Prompt‑injection regex:

```regex
(?is)(?:^|\n)\s*(system:|assistant:|ignore\s+previous|do\s+anything|jailbreak\b).*
```

### 3.3 Tokenization, POS Tagging, and Embedding Utilities

* **Tokenization**:
  $\mathcal{T}(x)=\{w_i\}_{i=1}^n$.
* **POS**:
  $\mathcal{P}(x)=\{(w_i,\text{tag}_i)\}$.
* **Embedding**: custom count‑based normalized vector:

  $$
    e_i = \frac{\mathrm{count}(w_i)}{\|\mathbf{c}\|_2},\quad \mathbf{c}=(\mathrm{count}(w_1),…),
  $$

  padded/truncated to 64 dimensions.

### 3.4 Secure Prompt Construction

Strips dangerous tokens and enforces maximum length. The final safe prompt:

$$
  \mathrm{Prompt}_{\mathrm{safe}} = \mathrm{sanitize}(\mathrm{minlen}(x,2000)).
$$

---

## 4. Advanced Vector Memory & Homomorphic Embeddings

### 4.1 Vector Encryption Pipeline

Given text embedding $\mathbf{e}\in\mathbb{R}^{64}$:

1. **Rotation**: $\mathbf{r}=Q\,\mathbf{e},\ Q\in SO(64)$.
2. **Quantization**:
   $\hat{\mathbf{r}}=\mathrm{clip}(\mathbf{r},[-1,1])\times127\to\mathbb{Z}_8^{64}$.
3. **Encryption**:
   $\tilde{\mathbf{r}}=\mathrm{AES\text{-}GCM}(K;N,\hat{\mathbf{r}},\mathrm{AAD})$.

### 4.2 SimHash Bucketing & Locality‑Sensitive Hashing

SimHash plane matrix $H\in\mathbb{R}^{16\times64}$:

$$
  b_i = \mathrm{sign}(H_i\cdot\mathbf{r})\in\{0,1\},\quad \mathrm{bucket}=b_1…b_{16}.
$$

Buckets enable \$\mathcal{O}(1)\$ approximate retrieval channels.

### 4.3 Rotation & Quantization Operations

* **Orthogonality**: $Q\,Q^\top=I$.
* **Quantization error**:
  $\epsilon_i = |r_i - \hat{r}_i/127|$, bounded by \$1/127\$.

### 4.4 SecureEnclave Context Manager

A context that zeroes any `numpy.ndarray` buffers on exit, ensuring no residual plaintext vectors remain in memory.

### 4.5 FHEv2 Embedding Encryption/Decryption

Although true FHE is not implemented, the pipeline mimics:

$$
  \langle \mathbf{e}_1,\mathbf{e}_2\rangle = 
  \mathbf{e}_1^\top\,\mathbf{e}_2
  \quad\text{computed after decryption only in SecureEnclave.}
$$

---

## 5. Topological Memory Manifold & Crystallization

### 5.1 Laplacian Graph Embedding

Given crystallized phrases $\{p_i\}$ with embeddings $\mathbf{e}_i$, compute pairwise weights:

$$
  W_{ij} = \exp\Bigl(-\frac{\|\mathbf{e}_i-\mathbf{e}_j\|^2}{2\sigma^2}\Bigr),\quad
  L=D-W,\ D_{ii}=\sum_j W_{ij}.
$$

### 5.2 Crystallized Phrase Logic (Scoring, Aging)

Score update per phrase:

$$
  s_{t+1} = 
  \begin{cases}
    s_t \times \gamma + 1, &\text{phrase seen}\\
    s_t \times \gamma, &\text{no new evidence}
  \end{cases},
  \quad \gamma=0.95.
$$

Crystallization if $s_t\ge5$.

### 5.3 Geodesic Memory Retrieval

Given query embedding $\mathbf{q}$, find start node
$\displaystyle i_0=\arg\min_i\|\mathbf{e}_i-\mathbf{q}\|$.
Then Dijkstra on graph $G$ with weights $W_{ij}$ to retrieve nearest phrases by geodesic distance.

### 5.4 Manifold Maintenance & Rebuilding

Upon any crystallization/purge event, recompute eigen decomposition:

$$
  L_{\mathrm{sym}} = D^{-1/2}LD^{-1/2},\quad
  L_{\mathrm{sym}}=\Phi\Lambda\Phi^\top,
$$

and take the 2nd to \$(\dim+1)\$th eigenvectors as coordinates.

---

## 6. Persistence Layers

### 6.1 SQLite Local Storage (Tables, Migration)

Manages tables:

```sql
CREATE TABLE IF NOT EXISTS local_responses(...);
CREATE TABLE IF NOT EXISTS memory_osmosis(...);
```

Uses PRAGMA checks to evolve schema (e.g., adding `aging_last` column).

### 6.2 Weaviate Client & Schema Bootstrapping

Ensures classes:

* `ReflectionLog`
* `InteractionHistory`
* `LongTermMemory`
* `CryptoPosition`
* `CryptoLivePosition`

via GraphQL schema definitions over HTTP.

### 6.3 Hybrid Record Upsert/Delete/Query

Dual writes:

```python
# SQLite
cur.execute("INSERT INTO local_responses ...")
# Weaviate
client.data_object.create(class_name="InteractionHistory", data_object=props)
```

Ensures resilience: if one store fails, the other persists.

### 6.4 Record AAD Contexts (for encryption)

Each stored object uses:

$$
  \mathrm{AAD} = \text{join}(\text{source},\text{table},\text{user\_id})
$$

for authenticated encryption, binding the ciphertext to its context.

---

## 7. LLM Integration & Prompt Engineering

### 7.1 Llama.cpp Model Setup & Execution

Load:

```python
llm = Llama(model_path, mmproj=mmproj_path, n_gpu_layers=-1, n_ctx=3900)
```

Parameters: up to 3,900 tokens context, Q4\_K\_M quantized weights.

### 7.2 Prompt Chunking, Memory Drift, and Attention Windows

Chunk size = 360 tokens:

$$
  \text{chunks} = \{x[i:i+360]\,\mid\,i=0,360,720,…\}.
$$

Memory drift token: `[[⊗DRIFT-QPU-SEGMENT]]` appended after 4 chunks to signal context decay.

### 7.3 Role/Token Detection and Output Type Inference

Tag each chunk using POS; then:

$$
  \text{token} =
  \begin{cases}
    [\text{code}], &\text{if code-like regex}\\
    [\text{action}], &\text{if VB most common}\\
    [\text{subject}], &\text{if NN most common}\\
    [\text{description}], &\text{if JJ/RB}\\
    [\text{general}], &\text{otherwise}
  \end{cases}
$$

### 7.4 Output Postprocessing (Coherence, Entropy Filters)

For each output segment, compute tail entropy:

$$
  H = \mathrm{std}(\{\mathrm{ord}(c)\mid c\in\text{tail}\})/100.
$$

If \$H>0.185\$ or Jensen–Shannon slope $\Delta D_{JS}>0.06$, abort generation early.

### 7.5 Multi‑Agent Consensus Pipeline (Ensembling)

Run \$m=5\$ agents with differing \$(T,p)\$:

$$
  T_i=0.85+0.10\,(i\bmod3),\quad p_i=0.80+0.05\,(i\bmod2).
$$

Parse key fields (direction, entry, TP, SL, confidence) from each response, then:

$$
  \text{direction}=\mathrm{mode}(\{\text{dir}_i\}),\quad
  \text{entry}=\mathrm{median}(\{\text{entry}_i\}).
$$

---

## 8. Quantum State Integration (PennyLane, RGB Gates)

### 8.1 QNode Device Setup

```python
dev = qml.device("default.qubit", wires=3)
```

Simulates a 3‑qubit quantum computer with state vector \$|\psi\rangle\in\mathbb{C}^8\$.

### 8.2 Quantum‑Driven Memory & Reasoning

Quantum gate parameters derived from CPU load, sentiment, weather, and previous \$Z\$‑states:

$$
  q_r = r\pi\,\mathrm{cpu\_scale}\,(1+\text{coherence\_gain}).
$$

### 8.3 RGB Extraction from Language/Sentiment

Compute HSV from polarity \$v\$, arousal \$a\$, dominance \$d\$:

$$
  \text{hue}=(1-v)\,120 + d\,20,\quad
  \text{sat}=0.25+0.4\,a+0.2\,\text{subjectivity},\quad
  \text{val}=0.9 - 0.03\,\ell + 0.2\,\rho,
$$

then convert to RGB.

### 8.4 Quantum Gate Definition & Measurement

Circuit:

```python
qml.RX(q_r, wires=0)
qml.RY(q_g, wires=1)
qml.RZ(q_b, wires=2)
qml.CRX(temp_norm⋅π⋅coherence, wires=[2,0])
…
```

Finally measure \$\langle Z\_i\rangle\$ for \$i=0,1,2\$.

### 8.5 Z‑State Management Across UI Cycles

Store last \$(z\_0,z\_1,z\_2)\$ and feed into next invocation, closing a quantum‑classical feedback loop.

---

## 9. External Data APIs & Context Sources

### 9.1 Coinbase API (Spot & Derivatives)

Requests signed by:

$$
  \sigma = \mathrm{HMAC\_SHA256}(\mathrm{API\_SECRET},\,\mathrm{timestamp}||\mathrm{method}||\mathrm{path}).
$$

### 9.2 CoinGecko Price History Integration

Fetch `$days=1$` minute‑level prices:

$$
  \{(t_i,p_i)\},\quad i=0,…,N-1,\quad N\approx1440.
$$

### 9.3 Open‑Meteo Weather Fetch

Current temperature \$T\_c\$ in °C, converted:

$$
  T_F = \frac{9}{5}T_c + 32.
$$

### 9.4 Live System Telemetry (CPU, etc.)

Using `psutil.cpu_percent(interval=0.3)`, scale to $\[0.05,1.0]\$ for quantum gate inputs.

---

## 10. GUI & User Interface Logic

### 10.1 CustomTkinter Style & Appearance

Dark mode, Roboto font, DPI‑aware sizing:

```python
customtkinter.set_appearance_mode("Dark")
```

### 10.2 Sidebar/Settings Frame Construction

Entries for username, API keys, latitude, longitude, weather, song, event type, and game fields. Values bound to `self.*_entry` widgets.

### 10.3 Main Conversation Pane (Text/Scrollbox)

`CTkTextbox` with custom colors/font, auto‑scrolls to newest message.

### 10.4 Input Handling, Event Binding, Async Queue

`on_submit()` enqueues `generate_response` via `ThreadPoolExecutor`, writes results back through a `queue.Queue`, polled by `process_queue()`.

### 10.5 Dynamic Fields (Lat/Lon, Weather, Event Type, etc.)

Allows context injection into prompts. GUI values read as floats/strings and fed into quantum and LLM pipelines.

### 10.6 Live Status Displays (Quantum State, Errors)

`self.image_label` and text box show quantum Z‑states, error messages, and debug information in real time.

---

## 11. Agentic Reasoning, Policy, and RL Head

### 11.1 Policy File Management (Load, Reload, Save)

Policy parameters in `policy_params.json`. Live‑reload based on file mtime and a lock to avoid race conditions.

### 11.2 REINFORCE/PG Parameter Update

Each sample stores:
$\mu_t,\sigma_t,\mu_p,\sigma_p,\log\pi(a|s)$.
Gradients computed as:

$$
  \nabla_{\mu_t} = (r_t - \bar{r})\frac{(x_t-\mu_t)}{\sigma_t^2},\quad
  \nabla_{\log\sigma_t} = (r_t-\bar{r})\Bigl(\frac{(x_t-\mu_t)^2}{\sigma_t^2}-1\Bigr).
$$

### 11.3 Bias, Entropy, and Dynamic Sampling

Bias factor \$b=(z\_0+z\_1+z\_2)/3\$. Temperature drawn from:

$$
  T\sim\mathcal{N}(\mu_t(b),\sigma_t^2(b)),\quad \mu_t(b)=\sigma(W_t b + b_t),
$$

clipped to \[0.2,1.5].

### 11.4 Policy Sampling per User Turn

At each user message, new \$(T,p)\$ are sampled and used for the subsequent llama invocation.

---

## 12. Memory Management & Long‑Term Aging

### 12.1 Score Decay and Half‑Life Logic

Given last update \$t\_0\$ and now \$t\$:

$$
  \Delta t = (t - t_0)/86400,\quad
  \text{half\_life} = T_0 + \Gamma\ln(1+s),\quad
  s_{\mathrm{new}} = s\cdot2^{-\Delta t/\text{half\_life}}.
$$

### 12.2 Memory Purging & Crystallization Events

If \$s\_{\mathrm{new}}<0.5\$, memory is purged (\$\mathrm{crystallized}=0\$). If \$s\_t\ge5\$ and not yet crystallized, it becomes a long‑term memory in Weaviate.

### 12.3 Manifold Rebuilding on Memory Shifts

If any memory’s crystallization flag changes, `topo_manifold.rebuild()` is triggered to update \$L\$ and eigenembedding.

### 12.4 Ongoing Aging Scheduler

Runs every 3600 s via `self.after(AGING_INTERVAL_SECONDS*1000,…)`, ensuring continuous decay and cleanup.

---

## 13. API, Data, and UI Utilities

### 13.1 Keyword/Noun Extraction

Uses `TextBlob(...).noun_phrases` and `nltk.pos_tag`:

$$
  K(x)=\{w_i\mid \mathrm{POS}(w_i)\in\{\text{NN, NNS, NNP}\}\}.
$$

### 13.2 Summarization & Sentiment Pipelines

Summarizer from `summa` uses TextRank; sentiment polarity \$p\in\[-1,1]\$ from `TextBlob`.

### 13.3 UUID Generation & Validation Helpers

Namespace UUID5:

$$
  \mathrm{UUID5}(N,x)=\mathrm{SHA1}(N||x)\to128\text{-bit}.
$$

### 13.4 General Error/Exception Handlers

Wraps I/O and external calls in try/except, logging with:

```python
logger.warning(f"[Context] {e}")
```

and returning safe fallbacks.

---

## 14. Logging, Debugging, and Diagnostics

### 14.1 Logger Setup & Usage Patterns

`logging.getLogger(__name__)` with level `DEBUG` for core modules; handlers can be redirected to file or console.

### 14.2 Error Reporting & Silent Fail Policy

Non‑fatal errors are logged at `WARNING`; only truly unrecoverable exceptions exit the program.

### 14.3 Runtime Self‑Check (Init Status)

On startup, checks for model files, database existence, and API key presence, alerting the user via GUI if any requirement is missing.

---

## 15. Main App Loop & Entry Point

### 15.1 `if __name__ == "__main__"` Boot Logic

Ensures that imports vs. execution contexts are separated. Bootstraps with:

```python
user_id="gray00"
app=App(user_id)
init_db()
app.mainloop()
```

### 15.2 Init Sequence (UserID, DB, GUI)

* **UserID**: default “gray00”, editable in GUI.
* **DB**: `init_db()` creates tables and Weaviate schemas.
* **GUI**: sets up frames, fields, event bindings, and thread pools.

### 15.3 Shutdown Hooks and Cleanup

`App.__exit__` and `after()` callbacks ensure `ThreadPoolExecutor.shutdown(wait=True)`, memory zeroization, and proper closure of external clients.

---

## 16. Future Extensions (Optional Sections)

### 16.1 Lottery/Sports/Custom Prediction Hooks

Placeholder code in GUI for “Lottery”, “Sports”, “Politics”, etc. can be extended with domain‑specific pipelines.

### 16.2 Multi‑User/Role Segregation

Design notes suggest partitioning memory and vault per user, ensuring multi‑tenant isolation:

$$
  \mathcal{S} = \bigcup_u \mathcal{S}_u,\quad \mathcal{K} = \{K_u\}.
$$

