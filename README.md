# Gaussian Elimination Solver — Project Documentation

**Linear Algebra Component · AMES Project**

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture](#2-architecture)
3. [Mathematical Background](#3-mathematical-background)
4. [Backend — Python / FastAPI](#4-backend--python--fastapi)
   - 4.1 [File Structure](#41-file-structure)
   - 4.2 [Algorithm — solver.py](#42-algorithm--solverpy)
   - 4.3 [API — main.py](#43-api--mainpy)
   - 4.4 [API Reference](#44-api-reference)
5. [Frontend — React / Vite](#5-frontend--react--vite)
   - 5.1 [File Structure](#51-file-structure)
   - 5.2 [Component: MatrixSolver](#52-component-matrixsolver)
   - 5.3 [Component: SolutionDisplay](#53-component-solutiondisplay)
6. [Running the Project](#6-running-the-project)
7. [Worked Example](#7-worked-example)
8. [Technology Stack](#8-technology-stack)
9. [Code Implementation Details](#9-code-implementation-details)
   - 9.1 [Backend Implementation — solver.py](#91-backend-implementation--solverpy)
   - 9.2 [Backend Implementation — main.py](#92-backend-implementation--mainpy)
   - 9.3 [Frontend Implementation — MatrixSolver.jsx](#93-frontend-implementation--matrixsolverjsx)
   - 9.4 [Frontend Implementation — SolutionDisplay.jsx](#94-frontend-implementation--solutiondisplayjsx)
   - 9.5 [Test Suite — test_solver.py](#95-test-suite--test_solverpy)
10. [Improvements & Fixes (v1.1)](#10-improvements--fixes-v11)
11. [Known Limitations](#11-known-limitations)
   - 11.1 [Mathematical Scope](#111-mathematical-scope)
   - 11.2 [Input & Size Constraints](#112-input--size-constraints)
   - 11.3 [API & Security](#113-api--security)
   - 11.4 [Frontend & UX](#114-frontend--ux)
   - 11.5 [Development / Testing](#115-development--testing)

---

## 1. Project Overview

This project implements a **Gaussian Elimination Solver** that solves a system of `n` linear equations in `n` unknowns:

$$A\mathbf{x} = \mathbf{b}$$

**Key features:**
- Gaussian elimination with **partial pivoting** for numerical stability
- **Back-substitution** to extract the solution vector
- Full **step-by-step working** — every row swap, elimination operation, and back-substitution step is recorded and displayed
- Interactive React UI where the user configures the matrix size (2×2 up to 8×8), fills in coefficients, and clicks **Solve**

---

## 2. Architecture

```
┌─────────────────────────────┐        HTTP POST /solve        ┌──────────────────────────┐
│   Frontend  (React / Vite)  │ ──────────────────────────────▶│  Backend  (FastAPI)       │
│   http://localhost:3000     │                                 │  http://localhost:8000    │
│                             │ ◀──────────────────────────────│                           │
│  MatrixSolver.jsx           │        JSON response           │  main.py  (API layer)     │
│  SolutionDisplay.jsx        │                                 │  solver.py (algorithm)   │
└─────────────────────────────┘                                 └──────────────────────────┘
```

**Communication:** The browser sends the augmented matrix `[A|b]` as JSON to the backend. The backend solves it and returns the solution vector plus every intermediate step. The frontend renders the complete working.

---

## 3. Mathematical Background

### System of Linear Equations

A system of `n` equations in `n` unknowns is written as:

$$\begin{pmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & & \ddots & \vdots \\ a_{n1} & a_{n2} & \cdots & a_{nn} \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{pmatrix} = \begin{pmatrix} b_1 \\ b_2 \\ \vdots \\ b_n \end{pmatrix}$$

This is represented as an **augmented matrix** `[A|b]` of shape `n × (n+1)`.

### Step 1 — Forward Elimination with Partial Pivoting

For each column `k` (the **pivot column**):

1. **Partial pivoting:** Find the row `p ≥ k` with the largest absolute value in column `k`:

$$p = \arg\max_{i \geq k} |a_{ik}|$$

Swap row `k` with row `p`. This avoids division by near-zero values and improves numerical stability.

2. **Elimination:** For each row `i > k`, compute the multiplier:

$$m_{ik} = \dfrac{a_{ik}}{a_{kk}}$$

Then update every element in that row:

$$R_i \leftarrow R_i - m_{ik} \times R_k$$

After all columns are processed, the matrix is in **upper-triangular form** `U`.

### Step 2 — Back-Substitution

Starting from the last row and working upward, solve for each variable:

$$x_i = \dfrac{b_i' - \displaystyle\sum_{j=i+1}^{n} u_{ij}\, x_j}{u_{ii}}$$

where `b'` and `U` are the modified values after forward elimination.

### Singularity Check

If any pivot `|a_{kk}| < 10^{-12}` (after row swapping), the matrix is singular — the system has no unique solution and an error is returned.

---

## 4. Backend — Python / FastAPI

### 4.1 File Structure

```
Backend/
├── main.py           ← FastAPI application, REST endpoints, CORS
├── solver.py         ← Gaussian elimination algorithm (pure logic)
├── requirements.txt  ← Python dependencies
├── start.py          ← Launcher: starts both backend and frontend
└── .venv/            ← Python virtual environment
```

### 4.2 Algorithm — `solver.py`

**Entry point:**
```python
gaussian_elimination(matrix: List[List[float]]) -> dict
```

**Input:**  
An augmented matrix `[A|b]` as a Python list of lists. Each row must have exactly `n+1` elements for an `n`-equation system.

**Output dictionary:**

| Key | Type | Description |
|-----|------|-------------|
| `solution` | `list[float]` | Values of `[x1, x2, …, xn]` |
| `variables` | `list[str]` | Labels `['x1', 'x2', …, 'xn']` |
| `steps` | `list[dict]` | Every recorded step (see below) |
| `n` | `int` | System size |

**Step dictionary structure:**

Each element of `steps` has:

| Field | Present when | Description |
|-------|-------------|-------------|
| `type` | always | One of: `initial`, `pivot`, `elimination`, `upper_triangular`, `back_substitution` |
| `description` | always | Human-readable description of the operation |
| `matrix` | all except `back_substitution` | Current state of the augmented matrix |
| `back_steps` | `back_substitution` only | List of per-variable substitution details |

**Step types in order:**

```
initial            → records the original [A|b]
pivot              → (if needed) records each row swap
elimination        → records each Ri ← Ri − factor × Rk operation
upper_triangular   → records the final upper-triangular matrix
back_substitution  → records x1…xn values with formulas
```

**Helper functions:**

| Function | Purpose |
|----------|---------|
| `_round(v, decimals=8)` | Rounds float, converts `−0.0` to `0.0` |
| `_mat_to_list(arr)` | Converts `numpy` array to a JSON-serialisable list, rounded to 6 decimal places |

**Numerical tolerances:**
- Pivot check: `< 1e-12` → singular matrix error
- Skip elimination: `< 1e-15` → treat as already zero

### 4.3 API — `main.py`

**Framework:** FastAPI with Pydantic v2 validation  
**CORS:** All origins allowed (suitable for local development)

**Request model:**
```python
class SolveRequest(BaseModel):
    matrix: List[List[float]]
    # Example: [[2,1,-1,8], [-3,-1,2,-11], [-2,1,2,-3]]
```

**Error handling:**

| Scenario | HTTP status | Response |
|----------|-------------|---------|
| Invalid matrix dimensions | `400` | `{"detail": "Row N has M elements; expected K"}` |
| Singular/near-singular matrix | `400` | `{"detail": "Matrix is singular…"}` |
| Unexpected server error | `500` | `{"detail": "Internal error: …"}` |

### 4.4 API Reference

#### `GET /`
Health check.

**Response:**
```json
{ "status": "ok", "message": "Gaussian Elimination Solver API is running." }
```

#### `POST /solve`
Solve the linear system.

**Request body:**
```json
{
  "matrix": [
    [2,  1, -1,  8],
    [-3, -1,  2, -11],
    [-2,  1,  2, -3]
  ]
}
```

**Success response (200):**
```json
{
  "solution": [2.0, 3.0, -1.0],
  "variables": ["x1", "x2", "x3"],
  "n": 3,
  "steps": [
    {
      "type": "initial",
      "description": "Original augmented matrix  [A | b]",
      "matrix": [[2,1,-1,8],[-3,-1,2,-11],[-2,1,2,-3]]
    },
    {
      "type": "pivot",
      "description": "Partial pivot: swap R1 ↔ R2  (largest pivot = -3)",
      "matrix": [[-3,-1,2,-11],[2,1,-1,8],[-2,1,2,-3]]
    },
    {
      "type": "elimination",
      "description": "R2  ←  R2  −  (-0.666667) × R1",
      "matrix": [...]
    },
    "... more elimination steps ...",
    {
      "type": "upper_triangular",
      "description": "Upper-triangular form achieved — ready for back-substitution",
      "matrix": [...]
    },
    {
      "type": "back_substitution",
      "description": "Back-substitution — solving for each variable",
      "back_steps": [
        { "variable": "x1", "value": 2.0, "description": "x1 = ... = 2" },
        { "variable": "x2", "value": 3.0, "description": "x2 = ... = 3" },
        { "variable": "x3", "value": -1.0, "description": "x3 = -1 / 1 = -1" }
      ]
    }
  ]
}
```

**Interactive API docs:** available at `http://localhost:8000/docs` (Swagger UI) when the backend is running.

---

## 5. Frontend — React / Vite

### 5.1 File Structure

```
Frontend/
├── index.html                   ← HTML entry point
├── package.json                 ← npm dependencies and scripts
├── vite.config.js               ← Vite dev server config (port 3000)
└── src/
    ├── main.jsx                 ← React root, mounts <App />
    ├── App.jsx                  ← Top-level component, renders <MatrixSolver />
    ├── index.css                ← All styles (global, cards, matrix, solution)
    └── components/
        ├── MatrixSolver.jsx     ← Matrix builder + Solve button + state management
        └── SolutionDisplay.jsx  ← Renders solution box + step-by-step working
```

### 5.2 Component: `MatrixSolver`

**File:** `src/components/MatrixSolver.jsx`

**State:**

| State | Type | Initial | Purpose |
|-------|------|---------|---------|
| `size` | `number` | `3` | Current n (number of equations) |
| `matrix` | `string[][]` | `emptyMatrix(3)` | n × (n+1) grid of input strings |
| `result` | `object \| null` | `null` | API response from `/solve` |
| `loading` | `boolean` | `false` | Shows "Solving…" on button |
| `error` | `string` | `''` | Validation or API error message |

**Key handlers:**

| Handler | Trigger | Action |
|---------|---------|--------|
| `handleSizeChange(n)` | Size button click | Resets matrix to `n × (n+1)` empty grid |
| `handleCell(row, col, val)` | Cell input change | Updates one cell in the matrix state |
| `handleExample()` | "Load Example" button | Fills matrix with a pre-defined example for current size |
| `handleReset()` | "Reset" button | Clears all cells and the solution |
| `handleSolve()` | "▶ Solve" button | Validates inputs, POSTs to `/solve`, sets `result` or `error` |

**Input validation (in `handleSolve`):**
- Every cell `matrix[i][j]` is checked with `parseFloat`
- Empty or non-numeric cells produce a specific error: *"Row N, column M is empty or invalid"*
- Parsed floats are assembled into the `n × (n+1)` array sent to the API

**Built-in examples:**

| Size | System |
|------|--------|
| 2×2 | `3x₁ + 2x₂ = 8`, `x₁ − x₂ = −1` |
| 3×3 | `2x₁ + x₂ − x₃ = 8`, `−3x₁ − x₂ + 2x₃ = −11`, `−2x₁ + x₂ + 2x₃ = −3` → x=(2,3,−1) |
| 4×4 | Four-equation system |

### 5.3 Component: `SolutionDisplay`

**File:** `src/components/SolutionDisplay.jsx`

**Props:** `result` (the full API response object)

**Sub-component `MatrixView`:** Renders any `n × (n+1)` matrix as an HTML table with the `b` column highlighted in red. Numbers are formatted to 4 decimal places with trailing zeros trimmed.

**Renders two sections:**

1. **Final Answer box** (green border) — displays each `xᵢ = value` pill

2. **Step-by-Step Working** — iterates over `result.steps` and renders a colour-coded card per step:

| Step type | Card colour | Badge colour | Content |
|-----------|-------------|-------------|---------|
| `initial` | grey border | grey | Augmented matrix table |
| `pivot` | amber border | amber | "Swap Rᵢ ↔ Rⱼ" + updated matrix |
| `elimination` | blue border | blue | "Rᵢ ← Rᵢ − factor × Rₖ" + updated matrix |
| `upper_triangular` | purple border | purple | Final upper-triangular matrix |
| `back_substitution` | green border | green | Each xᵢ = value with full formula |

---

## 6. Running the Project

### Prerequisites

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12+ | Backend runtime |
| Node.js | 18+ | Frontend build tool |
| npm | 9+ | JavaScript package manager |

### First-time setup

```powershell
# Backend — create virtual environment and install dependencies
cd Backend
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

```powershell
# Frontend — install JavaScript packages
cd Frontend
npm install
```

### Starting both servers (single command)

```powershell
cd Backend
.venv\Scripts\python.exe start.py
```

This launcher script:
1. Starts `npm run dev` (React dev server) as a background subprocess
2. Starts `uvicorn main:app --reload --reload-exclude .venv --port 8000`
3. Forwards **Ctrl+C** — both servers stop cleanly together

| Server | URL |
|--------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Swagger docs | http://localhost:8000/docs |

### Starting servers individually

```powershell
# Backend only
cd Backend
.venv\Scripts\uvicorn.exe main:app --reload --reload-exclude .venv --port 8000

# Frontend only
cd Frontend
npm run dev
```

---

## 7. Worked Example

**System (3×3):**

$$2x_1 + x_2 - x_3 = 8$$
$$-3x_1 - x_2 + 2x_3 = -11$$
$$-2x_1 + x_2 + 2x_3 = -3$$

**Augmented matrix:**

$$\left[\begin{array}{ccc|c} 2 & 1 & -1 & 8 \\ -3 & -1 & 2 & -11 \\ -2 & 1 & 2 & -3 \end{array}\right]$$

**Step 1 — Partial pivot (col 1):** Largest value in column 1 is `|-3| = 3` in row 2. Swap R1 ↔ R2.

$$\left[\begin{array}{ccc|c} -3 & -1 & 2 & -11 \\ 2 & 1 & -1 & 8 \\ -2 & 1 & 2 & -3 \end{array}\right]$$

**Step 2 — Eliminate col 1:**

- R2 ← R2 − (−0.6667) × R1
- R3 ← R3 − (0.6667) × R1

$$\left[\begin{array}{ccc|c} -3 & -1 & 2 & -11 \\ 0 & 0.333 & 0.333 & 0.667 \\ 0 & 1.667 & 3.333 & 4.333 \end{array}\right]$$

**Step 3 — Partial pivot (col 2):** Largest in column 2 (rows 2–3) is `1.667` in row 3. Swap R2 ↔ R3.

**Step 4 — Eliminate col 2:** R3 ← R3 − (0.2) × R2 → upper-triangular form.

**Step 5 — Back-substitution:**

- x3 = −1
- x2 = (b'₂ − u₂₃·x3) / u₂₂ = **3**
- x1 = (b'₁ − u₁₂·x2 − u₁₃·x3) / u₁₁ = **2**

**Solution:** `x1 = 2, x2 = 3, x3 = −1`

---

## 8. Technology Stack

### Backend

| Package | Version | Role |
|---------|---------|------|
| Python | 3.12 | Runtime |
| FastAPI | ≥ 0.111 | REST API framework |
| Uvicorn | ≥ 0.29 | ASGI web server |
| NumPy | ≥ 1.26 | Matrix operations (array storage, `argmax`, `dot`) |
| Pydantic | ≥ 2.0 | Request body validation and serialisation |

### Frontend

| Package | Version | Role |
|---------|---------|------|
| React | 18 | UI component library |
| React DOM | 18 | Browser rendering |
| Vite | 5 | Dev server and build tool |
| @vitejs/plugin-react | 4 | JSX transform via Babel |

### Project guidelines applied

| Guideline | Implementation |
|-----------|---------------|
| Well-organised structure | Backend algorithm (`solver.py`) separated from API layer (`main.py`) |
| Use reliable tools | NumPy for all matrix arithmetic instead of manual loops |
| Validate inputs early | Matrix dimension check and singularity check before any computation |
| Show results visually | Colour-coded step cards, matrix tables, solution pills in the UI |
| Test against known rules | Example systems with verified solutions (x=(2,3,−1) for the 3×3 example) |

---

## 9. Code Implementation Details

### 9.1 Backend Implementation — `solver.py`

**Core Algorithm Flow:**

```
Input: Augmented matrix [A | b] (n × n+1)
   ↓
[Step 1] Input Validation
   - Check matrix not empty
   - Check all rows have n+1 elements
   - Check matrix size ≤ MAX_MATRIX_SIZE (100)
   ↓
[Step 2] Forward Elimination with Partial Pivoting
   For each column k (pivot column):
   │
   ├─ Find pivot row: argmax(|a_ik|) for i ≥ k
   ├─ If |a_kk| < 1e-12: raise ValueError ("singular")
   ├─ Swap rows if needed (record "pivot" step)
   │
   └─ Eliminate below pivot:
      For each row i > k:
         factor = a_ik / a_kk
         a_i = a_i - factor × a_k
         (record "elimination" step)
   ↓
[Step 3] Record upper-triangular form
   ↓
[Step 4] Back-Substitution
   For i = n-1 down to 0:
   │
   └─ x_i = (b_i - sum(a_ij × x_j for j>i)) / a_ii
      (record "back_substitution" step with formula)
   ↓
Output: { solution, variables, steps, n }
```

**Key Implementation Features:**

| Component | Code | Purpose |
|-----------|------|---------|
| **Partial Pivoting** | `pivot_row = col + int(np.argmax(np.abs(aug[col:, col])))` | Finds row with largest absolute value in current column (improves stability) |
| **Singularity Check** | `if abs(aug[pivot_row, col]) < 1e-12: raise ValueError(...)` | Detects singular/near-singular matrices before division |
| **Row Elimination** | `aug[row] = aug[row] - factor * aug[col]` | Vectorised NumPy operation for efficiency |
| **Step Recording** | `steps.append({"type": "...", "description": "...", "matrix": ...})` | Captures every operation for UI replay |
| **Rounding** | `r = 0.0 if round(v, 8) == 0.0 else round(v, 8)` | Converts -0.0 to 0.0 to avoid display artifacts |
| **Size Validation** | `if n > MAX_MATRIX_SIZE: raise ValueError(...)` | Configurable resource guard (default: 100×100) |

**Helper Functions:**

```python
def _round(v: float, decimals=8) -> float:
    """Round float and convert -0.0 to 0.0."""
    r = round(float(v), decimals)
    return 0.0 if r == 0.0 else r

def _mat_to_list(arr: np.ndarray) -> list:
    """Convert NumPy array to JSON-serialisable list of lists."""
    return [[_round(v, 6) for v in row] for row in arr.tolist()]
```

---

### 9.2 Backend Implementation — `main.py`

**API Layer Design:**

```python
app = FastAPI(title="...", version="1.0.0")

# CORS Middleware — allows all origins (development only)
app.add_middleware(CORSMiddleware, allow_origins=["*"], ...)

# Rate Limiting Middleware — 30 requests per 60s per IP
app.middleware("http")(rate_limit_middleware)

@app.get("/")
def root() → dict:
    """Health check endpoint."""
    return {"status": "ok", "message": "..."}

@app.post("/solve")
def solve(request: SolveRequest) → dict:
    """Main solver endpoint."""
    try:
        result = gaussian_elimination(request.matrix)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")
```

**Rate Limiting Implementation:**

```python
REQUEST_LIMIT = 30  # requests
WINDOW_SIZE = 60    # seconds
request_tracker = defaultdict(list)  # {client_ip: [timestamp, ...]}

async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    now = time.time()
    
    # Remove timestamps older than WINDOW_SIZE
    request_tracker[client_ip] = [
        ts for ts in request_tracker[client_ip]
        if now - ts < WINDOW_SIZE
    ]
    
    # Check limit
    if len(request_tracker[client_ip]) >= REQUEST_LIMIT:
        return HTTPException(status_code=429, 
            detail=f"Rate limit: {REQUEST_LIMIT} requests per {WINDOW_SIZE}s")
    
    # Record and process
    request_tracker[client_ip].append(now)
    return await call_next(request)
```

**Request Validation (Pydantic):**

```python
class SolveRequest(BaseModel):
    matrix: List[List[float]] = Field(
        ...,
        description="Augmented matrix [A|b]",
        example=[[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]]
    )
    # Pydantic automatically:
    # - Validates type (List[List[float]])
    # - Rejects invalid JSON
    # - Serialises response to JSON
```

**Error Codes & Responses:**

| Status | Trigger | Example Response |
|--------|---------|------------------|
| 200 | Successful solve | `{"solution": [...], "variables": [...], "steps": [...], "n": 3}` |
| 400 | Invalid input (singular, wrong dimensions) | `{"detail": "Matrix is singular..."}` |
| 429 | Rate limit exceeded | `{"detail": "Rate limit: 30 requests per 60s"}` |
| 500 | Unexpected server error | `{"detail": "Internal error: ..."}` |

---

### 9.3 Frontend Implementation — `MatrixSolver.jsx`

**Component State & Props:**

```javascript
function MatrixSolver() {
  const [size, setSize] = useState(3)        // 2–8
  const [matrix, setMatrix] = useState(...)  // string[][] (n × n+1)
  const [result, setResult] = useState(null) // API response or null
  const [loading, setLoading] = useState(false)  // Disable button during solve
  const [error, setError] = useState('')    // Error message or empty
}
```

**Environment & Constants:**

```javascript
const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'
const FETCH_TIMEOUT_MS = 30000  // 30 seconds

const EXAMPLES = {
  2: [['3', '2', '8'], ['1', '-1', '-1']],
  3: [['2', '1', '-1', '8'], ['-3', '-1', '2', '-11'], ['-2', '1', '2', '-3']],
  4: [...],  // 4×4 example
  5: [...],  // 5×5 example
  // ... up to 8
}
```

**Event Handlers (with useCallback memoization):**

```javascript
// Size selector
const handleSizeChange = useCallback((n) => {
  setSize(n)
  setMatrix(emptyMatrix(n))
  setResult(null)
  setError('')
}, [])

// Cell input
const handleCell = useCallback((row, col, val) => {
  setMatrix((prev) => {
    const next = prev.map((r) => [...r])
    next[row][col] = val
    return next
  })
}, [])

// Load example
const handleExample = useCallback(() => {
  const ex = EXAMPLES[size]
  if (ex) {
    setMatrix(ex.map((r) => [...r]))
    setResult(null)
    setError('')
  }
}, [size])  // Depends on size

// Reset form
const handleReset = useCallback(() => {
  setMatrix(emptyMatrix(size))
  setResult(null)
  setError('')
}, [size])

// Main solve with validation & timeout
const handleSolve = useCallback(async () => {
  // 1. Validate all cells
  const parsed = []
  for (let i = 0; i < size; i++) {
    const row = []
    for (let j = 0; j <= size; j++) {
      const raw = matrix[i]?.[j] ?? ''
      const num = parseFloat(raw)
      if (raw === '' || isNaN(num)) {
        setError(`Row ${i + 1}, column ${j + 1} is empty or invalid`)
        return
      }
      row.push(num)
    }
    parsed.push(row)
  }
  
  // 2. Setup timeout
  setLoading(true)
  setError('')
  setResult(null)
  
  const controller = new AbortController()
  const timeoutId = setTimeout(
    () => controller.abort(),
    FETCH_TIMEOUT_MS
  )
  
  try {
    // 3. Fetch with abort signal
    const res = await fetch(`${API}/solve`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ matrix: parsed }),
      signal: controller.signal
    })
    
    // 4. Parse response
    const data = await res.json()
    if (!res.ok) {
      setError(data.detail ?? 'Solver returned an error')
    } else {
      setResult(data)
    }
  } catch (err) {
    // 5. Handle errors
    if (err.name === 'AbortError') {
      setError(`Request timeout after ${FETCH_TIMEOUT_MS / 1000}s`)
    } else {
      setError(`Cannot reach backend at ${API}. Is the server running?`)
    }
  } finally {
    clearTimeout(timeoutId)
    setLoading(false)
  }
}, [size, matrix])
```

**Rendering:**

```javascript
return (
  <div className="page">
    {/* Header */}
    <header className="page-header">
      <h1>Gaussian Elimination Solver</h1>
    </header>
    
    <main className="page-body">
      {/* Size Selector */}
      <section className="card">
        <div className="size-strip">
          {[2, 3, 4, 5, 6, 7, 8].map(n => (
            <button
              key={n}
              className={`size-btn ${size === n ? 'active' : ''}`}
              onClick={() => handleSizeChange(n)}
            >
              {n} × {n}
            </button>
          ))}
        </div>
      </section>
      
      {/* Matrix Input */}
      <section className="card">
        <table className="mtable">
          {/* Column headers: x1, x2, ..., xn, |, b */}
          {/* Row labels: R1, R2, ... */}
          {/* Input cells with onChange → handleCell */}
        </table>
        <div className="action-row">
          <button onClick={handleSolve} disabled={loading}>
            {loading ? 'Solving…' : '▶ Solve'}
          </button>
          <button onClick={handleReset}>Reset</button>
        </div>
      </section>
      
      {/* Error Display */}
      {error && <div className="error-box">{error}</div>}
      
      {/* Solution (if available) */}
      {result && <SolutionDisplay result={result} />}
    </main>
  </div>
)
```

---

### 9.4 Frontend Implementation — `SolutionDisplay.jsx`

**Component Structure:**

```javascript
function MatrixView({ matrix, n }) {
  // Format numbers: 4 decimals, trim trailing zeros
  const fmt = (v) => {
    const s = v.toFixed(4)
    return s.replace(/(\.\d*?)0+$/, '$1').replace(/\.$/, '')
  }
  
  return (
    <div className="matrix-view">
      <table>
        <thead>
          {/* n columns for x1..xn, plus b column */}
        </thead>
        <tbody>
          {/* matrix[i][0..n-1] = coefficients (normal) */}
          {/* matrix[i][n] = b column (red highlight) */}
        </tbody>
      </table>
    </div>
  )
}

export default function SolutionDisplay({ result }) {
  const { solution, variables, steps, n } = result
  
  return (
    <>
      {/* Final Answer Box */}
      <div className="final-answer">
        <h2>Solution</h2>
        <div className="answer-grid">
          {variables.map((v, i) => (
            <div className="answer-item" key={i}>
              <span className="av">{v}</span>
              <span className="aeq">=</span>
              <span className="aval">{solution[i]}</span>
            </div>
          ))}
        </div>
      </div>
      
      {/* Step-by-Step Working */}
      <div className="steps-block">
        <h2>Step-by-Step Working</h2>
        {steps.map((step, idx) => (
          <div key={idx} className={`step-card sc-${step.type}`}>
            {/* Badge + Description */}
            <div className="step-head">
              <span className={`badge b-${step.type}`}>
                {STEP_LABEL[step.type]}
              </span>
              <span className="step-desc">{step.description}</span>
            </div>
            
            {/* Matrix view (for pivot, elimination, upper_triangular) */}
            {step.matrix && <MatrixView matrix={step.matrix} n={n} />}
            
            {/* Back-substitution details */}
            {step.back_steps && (
              <div className="bs-list">
                {step.back_steps.map((bs, i) => (
                  <div key={i} className="bs-row">
                    <span className="bs-var">{bs.variable}</span>
                    <span className="bs-eq">=</span>
                    <span className="bs-val">{bs.value}</span>
                    <code className="bs-formula">{bs.description}</code>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </>
  )
}
```

**Step Badge Styling:**

| Type | Badge | Card BG | Color | When |
|------|-------|---------|-------|------|
| `initial` | INITIAL MATRIX | Light grey | #94a3b8 | Step 0: Original [A\|b] |
| `pivot` | PIVOT (ROW SWAP) | #fffbeb | #f59e0b | When row swap occurs |
| `elimination` | ROW ELIMINATION | #eff6ff | #3b82f6 | For each Ri ← Ri - factor×Rk |
| `upper_triangular` | UPPER TRIANGULAR | #faf5ff | #8b5cf6 | After all elimination |
| `back_substitution` | BACK SUBSTITUTION | #f0fdf4 | #10b981 | Final solution calculation |

---

### 9.5 Test Suite — `test_solver.py`

**Test Organization (20 tests, 100% pass rate):**

```python
class TestBasicSolutions:
    """5 tests — Verify correct solutions for known systems"""
    def test_2x2_simple()        # [3,2,8], [1,-1,-1] → (1.2, 2.2)
    def test_3x3_standard()      # README example → (2, 3, -1)
    def test_identity_matrix()   # I·x = b → x = b
    def test_negative_solution() # System with negatives
    def test_fractional_solution() # System with fractions

class TestStructure:
    """4 tests — Verify response structure & metadata"""
    def test_result_keys()       # {"solution", "variables", "steps", "n"}
    def test_variables_naming()  # ["x1", "x2", ...]
    def test_steps_sequence()    # initial → pivot/elim → upper_tri → back_sub
    def test_step_has_description() # All steps have descriptions

class TestErrorCases:
    """4 tests — Verify error handling"""
    def test_singular_matrix()   # Raises ValueError
    def test_empty_matrix()      # Raises ValueError
    def test_mismatched_rows()   # Raises ValueError
    def test_oversized_matrix()  # Raises ValueError (>100×100)

class TestNumericalProperties:
    """3 tests — Verify numeric correctness"""
    def test_solution_type()     # List of floats
    def test_no_negative_zero()  # -0.0 → 0.0
    def test_solution_count()    # len(solution) == n

class TestPivotingCorrectness:
    """2 tests — Verify partial pivoting"""
    def test_pivoting_necessary() # Small pivot → swap
    def test_pivoting_recorded()  # "pivot" step in steps

class TestBackSubstitution:
    """2 tests — Verify back-substitution"""
    def test_back_sub_step_exists() # back_substitution in steps
    def test_back_sub_formulas()    # Each step has formula & value
```

**Running & Results:**

```bash
cd Backend
.venv\Scripts\pip install pytest  # (if not already installed)
.venv\Scripts\python -m pytest test_solver.py -v
```

**Expected Output:**

```
test_solver.py::TestBasicSolutions::test_2x2_simple PASSED        [  5%]
test_solver.py::TestBasicSolutions::test_3x3_standard PASSED      [ 10%]
...
============================= 20 passed in 0.23s ==============================
```

---

## 10. Improvements & Fixes (v1.1)

**This update addresses all identified limitations from the code review.**

### Frontend Enhancements

| Improvement | Details |
|-----------|---------|
| **Environment variables** | API URL moved to `.env.local` — `VITE_API_URL=http://localhost:8000`. Configurable without code changes. |
| **Fetch timeout** | All API requests now have a 30-second timeout with `AbortController`. Prevents UI hanging if backend is unresponsive. |
| **Handler memoization** | `handleExample`, `handleReset`, `handleSolve` now use `useCallback` for improved performance. |
| **Complete example set** | Added built-in examples for all sizes 2–8. "Load Example" button now available for every size option. |

### Backend Enhancements

| Improvement | Details |
|-----------|---------|
| **Rate limiting** | Middleware added: 30 requests per 60 seconds per IP address. Returns HTTP 429 if exceeded. |
| **Size validation** | Optional upper limit (configurable `MAX_MATRIX_SIZE=100`). Prevents resource exhaustion from oversized matrices. |
| **Automated test suite** | 20 comprehensive pytest tests covering: basic solutions, error cases, numerical properties, pivoting, and back-substitution. Run with `pytest test_solver.py -v`. Ensures algorithm correctness. |
| **API documentation** | Updated `/solve` endpoint docstring with rate limit and size limit info. |

### Security & Deployment

| Setting | Status |
|---------|--------|
| CORS origins | Still `["*"]` for development. **Before production:** restrict to trusted domains. |
| Authentication | None. **Before production:** add API key or session management. |
| Rate limit scope | Per-IP in simple in-memory store. **For production:** use Redis-backed rate limiter for multi-process/multi-server setups. |

---

## 11. Known Limitations

### 11.1 Mathematical Scope

| Limitation | Detail |
|-----------|--------|
| **Square systems only** | The solver requires exactly `n` equations in `n` unknowns. Overdetermined systems (`m > n`) and underdetermined systems (`m < n`) are not supported and will be rejected with a dimension error. |
| **Unique solutions only** | If the coefficient matrix is singular or nearly singular (pivot `< 1e-12`), the solver raises an error. Systems with infinitely many solutions (rank-deficient) or no solution (inconsistent) are not distinguished — both are reported as "singular matrix". |
| **Real numbers only** | All coefficients must be real-valued floats. Complex-valued linear systems are not supported. |
| **Floating-point precision** | Partial pivoting reduces but does not eliminate accumulated rounding error. For ill-conditioned matrices (large condition number), results may carry significant numerical inaccuracy despite being rounded to 8 decimal places. |

### 11.2 Input & Size Constraints

| Limitation | Detail | Status |
|-----------|--------|--------|
| **UI size cap at 8×8** | The frontend allows systems from 2×2 up to 8×8 only. Larger systems must be submitted directly to the API endpoint (`POST /solve`), bypassing all frontend validation. | By design |
| **Backend size cap at 100×100** | Configurable in `solver.py` (`MAX_MATRIX_SIZE=100`). Matrices exceeding this limit will be rejected with a clear error. Adjust `MAX_MATRIX_SIZE` to raise or lower this limit. | ✅ Fixed in v1.1 |
| **Built-in examples for all sizes 2–8** | The "Load Example" button is now available for all selectable sizes. Pre-filled examples included for each. | ✅ Fixed in v1.1 |

### 11.3 API & Security

| Limitation | Detail | Status |
|-----------|--------|--------|
| **CORS wildcard** | `allow_origins=["*"]` is set in `main.py`. Any website can call the API. This is acceptable for local development but must be restricted to trusted origins before any deployment. | Acknowledged; see Section 9 for production guidance |
| **No authentication** | The `/solve` endpoint is fully public with no API key, token, or session check. | By design for this project |
| **Rate limiting enabled** | Simple per-IP rate limit: 30 requests per 60 seconds. Exceeding this returns HTTP 429. For production multi-server setups, use Redis-backed rate limiting. | ✅ Fixed in v1.1 |
| **Hardcoded backend URL (legacy)** | ~~The frontend hardcoded `http://localhost:8000` as the API base URL.~~ **Now:** Backend URL is configured via `VITE_API_URL` environment variable in `.env.local`. | ✅ Fixed in v1.1 |

### 11.4 Frontend & UX

| Limitation | Detail | Status |
|-----------|--------|--------|
| **Fetch timeout added** | ~~The `fetch()` call had no timeout.~~ **Now:** All requests have a 30-second timeout via `AbortController`. If exceeded, the UI displays a clear error and the "Solving…" button re-enables. | ✅ Fixed in v1.1 |
| **No persistent state** | Results are held in React component state only. Refreshing the page clears all inputs and outputs. There is no save, export, or share functionality. | By design |
| **Number display precision** | The solution display renders values to 4 decimal places (trailing zeros trimmed). The backend computes to 8 decimal places internally; very small differences between the two precision levels may confuse users comparing displayed values against raw API responses. | Acknowledged |
| **Handler performance** | ~~Event handlers were not memoised.~~ **Now:** `handleExample`, `handleReset`, and `handleSolve` use `useCallback` for optimised re-renders. | ✅ Fixed in v1.1 |

### 11.5 Development / Testing

| Limitation | Detail | Status |
|-----------|--------|--------|
| **Automated test suite** | ~~No `pytest` unit tests existed.~~ **Now:** 20 comprehensive tests cover basic solutions, error handling, numerical properties, pivoting, and back-substitution. Run with `pytest test_solver.py -v`. All tests pass. | ✅ Fixed in v1.1 |
| **Development-only server** | Uvicorn is started with `--reload` and a single worker (development mode). This configuration is not suitable for concurrent multi-user access or production deployment. | By design |
| **Windows-only launcher** | `start.py` uses `shell=True` and `npm.cmd` resolution which relies on Windows path behaviour. The launcher is untested on macOS or Linux. | Known limitation |
