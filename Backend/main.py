from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import time
from collections import defaultdict
from solver import gaussian_elimination

app = FastAPI(
    title="Gaussian Elimination Solver API",
    version="1.0.0",
    description="Solves n×n systems of linear equations using Gaussian elimination with partial pivoting.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Rate limiting (simple in-memory store) ──────────────────────────────────
# Tracks requests per client IP. Limit: 30 requests per minute.
REQUEST_LIMIT = 30
WINDOW_SIZE = 60  # seconds
request_tracker = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Simple rate limiting middleware — 30 requests per minute per IP."""
    client_ip = request.client.host if request.client else "unknown"
    now = time.time()

    # Clean old requests outside the window
    request_tracker[client_ip] = [
        req_time for req_time in request_tracker[client_ip]
        if now - req_time < WINDOW_SIZE
    ]

    # Check limit
    if len(request_tracker[client_ip]) >= REQUEST_LIMIT:
        return HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {REQUEST_LIMIT} requests per {WINDOW_SIZE}s.",
        )

    # Record this request
    request_tracker[client_ip].append(now)
    response = await call_next(request)
    return response


class SolveRequest(BaseModel):
    matrix: List[List[float]] = Field(
        ...,
        description="Augmented matrix [A|b] — each row has n+1 elements.",
        example=[
            [2, 1, -1, 8],
            [-3, -1, 2, -11],
            [-2, 1, 2, -3],
        ],
    )


@app.get("/", tags=["health"])
def root():
    return {"status": "ok", "message": "Gaussian Elimination Solver API is running."}


@app.get("/test", tags=["health"])
def test_endpoint():
    """
    Quick test endpoint — solves a simple identity matrix.
    If this returns [2, 3, 4], the solver is working correctly.
    """
    try:
        test_matrix = [
            [1, 0, 0, 2],
            [0, 1, 0, 3],
            [0, 0, 1, 4],
        ]
        result = gaussian_elimination(test_matrix)
        return {
            "status": "ok",
            "message": "Test passed — solver is working correctly",
            "test_input": test_matrix,
            "expected_solution": [2, 3, 4],
            "actual_solution": result["solution"],
            "match": result["solution"] == [2.0, 3.0, 4.0],
        }
    except Exception as exc:
        return {
            "status": "error",
            "message": str(exc),
        }


@app.post("/solve", tags=["solver"])
def solve(request: SolveRequest):
    """
    Solve the system Ax = b given as an augmented matrix [A|b].

    Returns the solution vector and every elimination/pivoting step
    so the client can display the full working.

    **Rate limit:** 30 requests per 60 seconds per IP.
    **Size limit:** Matrices larger than 100×100 may exceed CPU or memory constraints.
    """
    try:
        result = gaussian_elimination(request.matrix)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")
