"""
Launcher — starts both the FastAPI backend and the React frontend.

Usage (from the Backend folder):
    .venv\\Scripts\\python.exe start.py

Press Ctrl+C once to stop both servers cleanly.
"""

import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BACKEND_DIR.parent / "Frontend"


def main() -> None:
    print("=" * 55)
    print("  Gaussian Elimination Solver")
    print("  Backend  →  http://127.0.0.1:8000")
    print("  Frontend →  http://localhost:3000")
    print("=" * 55)
    print()

    # ── Check Frontend directory exists ────────────────────────────
    if not FRONTEND_DIR.exists():
        print(f"[launcher] ERROR: Frontend folder not found at {FRONTEND_DIR}")
        sys.exit(1)

    # ── Start React dev server (npm.cmd on Windows) ─────────────────
    print("[launcher] Starting React frontend  (npm run dev)…")
    fe = subprocess.Popen(
        "npm run dev",
        cwd=str(FRONTEND_DIR),
        shell=True,          # required on Windows so npm.cmd is found
        stdout=None,         # inherit so output shows in terminal
        stderr=None,
    )

    # ── Start FastAPI / uvicorn ──────────────────────────────────────
    # --reload-exclude .venv  → prevents uvicorn from reloading on every
    #   package install inside the virtual environment.
    print("[launcher] Starting FastAPI backend (uvicorn)…\n")
    be = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "main:app",
            "--reload",
            "--reload-exclude", str(BACKEND_DIR / ".venv"),
            "--port", "8000",
        ],
        cwd=str(BACKEND_DIR),
    )

    # Wait for uvicorn; forward Ctrl+C to both processes
    try:
        be.wait()
    except KeyboardInterrupt:
        print("\n[launcher] Shutting down both servers…")
    finally:
        fe.terminate()
        be.terminate()
        fe.wait(timeout=5)
        be.wait(timeout=5)
        print("[launcher] All servers stopped.")


if __name__ == "__main__":
    main()
