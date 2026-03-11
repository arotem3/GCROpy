#!/usr/bin/env python3
"""Test GCRO-DR on a shared 2D advection-diffusion benchmark."""

import argparse
import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import gmres, spilu
import matplotlib.pyplot as plt
from gcro import gcro


def create_2d_advection_diffusion(
    nx: int,
    ny: int,
    diffusion: float = 0.05,
    advection_x: float = 8.0,
    advection_y: float = 4.0,
):
    """Create the shared 2D advection-diffusion benchmark operator."""
    n = nx * ny
    hx = 1.0 / (nx + 1)
    hy = 1.0 / (ny + 1)

    Ix = eye(nx, format="csr")
    Iy = eye(ny, format="csr")

    Dxx = diags(  # type: ignore[arg-type]
        [np.ones(nx - 1), -2 * np.ones(nx), np.ones(nx - 1)],
        offsets=(-1, 0, 1),
        shape=(nx, nx),
        format="csr",
    ) / (hx**2)

    Dyy = diags(  # type: ignore[arg-type]
        [np.ones(ny - 1), -2 * np.ones(ny), np.ones(ny - 1)],
        offsets=(-1, 0, 1),
        shape=(ny, ny),
        format="csr",
    ) / (hy**2)

    laplacian = kron(Dxx, Iy, format="csr") + kron(Ix, Dyy, format="csr")

    Dx = diags(  # type: ignore[arg-type]
        [-np.ones(nx - 1), np.ones(nx - 1)],
        offsets=(-1, 1),
        shape=(nx, nx),
        format="csr",
    ) / (2.0 * hx)
    Dy = diags(  # type: ignore[arg-type]
        [-np.ones(ny - 1), np.ones(ny - 1)],
        offsets=(-1, 1),
        shape=(ny, ny),
        format="csr",
    ) / (2.0 * hy)

    Cx = kron(Dx, Iy, format="csr")
    Cy = kron(Ix, Dy, format="csr")

    A_sparse = (
        -diffusion * laplacian
        + advection_x * Cx
        + advection_y * Cy
        + eye(n, format="csr")
    )

    return A_sparse, n


def test_gcro_large_sparse_nonsym(plot=False):
    """Test GCRO-DR on the same matrix used by the preconditioning benchmark."""
    np.random.seed(456)
    nx, ny = 64, 64
    diffusion = 0.05
    advection_x = 8.0
    advection_y = 4.0
    A_sparse, n = create_2d_advection_diffusion(
        nx, ny, diffusion, advection_x, advection_y
    )

    def A_matvec(x):
        return A_sparse.dot(x)

    # Right-hand side
    b = np.random.randn(n)

    # Solve using GCRO-DR
    solver = gcro(n=n, A=A_matvec, kdim=20, edim=10)
    x_approx, residuals, niter, nrestart = solver.solve(b, tol=1e-6, maxiter=1000)

    # Check accuracy
    residual = np.linalg.norm(A_sparse.dot(x_approx) - b) / np.linalg.norm(b)

    print(f"Test 1: Large sparse 2D advection-diffusion {nx}×{ny} system")
    print(f"  Sparsity: {A_sparse.nnz / (n**2) * 100:.2f}% non-zeros")
    print(f"  Residual: {residual:.2e}")
    print(f"  Final residual norm / initial: {residual / np.linalg.norm(b):.2e}")
    print(f"  Iterations: {niter}")
    print(f"  Restarts: {nrestart}")
    assert residual < 1e-6, f"Residual too large: {residual}"
    print("  PASSED ✓\n")

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5), layout="constrained")
        ax.semilogy(residuals, ".-", linewidth=2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual Norm")
        ax.set_title(f"GCRO-DR Convergence ({nx}×{ny}, no preconditioning)")
        ax.grid(True, alpha=0.3)
        plt.show()

    return residuals


def test_with_preconditioning(plot=False):
    """Test GCRO-DR with and without preconditioning on the shared matrix."""
    print("=" * 70)
    print("GCRO-DR with Preconditioning Test")
    print("=" * 70)

    nx, ny = 64, 64
    diffusion = 0.05
    advection_x = 8.0
    advection_y = 4.0
    A_sparse, n = create_2d_advection_diffusion(
        nx, ny, diffusion, advection_x, advection_y
    )

    print("\nProblem: 2D Advection-Diffusion")
    print(f"  Grid: {nx} × {ny} = {n} unknowns")
    print(f"  Matrix sparsity: {A_sparse.nnz / (n**2) * 100:.2f}% non-zeros")

    def A_matvec(x):
        return A_sparse.dot(x)

    np.random.seed(42)
    b = np.random.randn(n)
    b_norm = np.linalg.norm(b)

    tol = 1e-6
    maxiter = 1000
    kdim = 20
    edim = 10

    print("\n--- Test 2: GCRO-DR (no preconditioning) ---")
    solver_noprecond = gcro(n=n, A=A_matvec, kdim=kdim, edim=edim)
    x_noprecond, residuals_noprecond, niter_noprecond, nrestart_noprecond = (
        solver_noprecond.solve(b, tol=tol, maxiter=maxiter)
    )
    residual_noprecond = np.linalg.norm(A_sparse.dot(x_noprecond) - b) / b_norm
    print(f"  Iterations: {niter_noprecond}")
    print(f"  Restarts: {nrestart_noprecond}")
    print(f"  Final residual: {residual_noprecond:.2e}")

    print("\n--- Test 3: GCRO-DR with ILU(0) preconditioning ---")
    ilu = spilu(A_sparse.tocsc(), drop_tol=0.0, fill_factor=1.0)

    def M_ilu(x):
        return ilu.solve(x)

    solver_ilu = gcro(n=n, A=A_matvec, kdim=kdim, edim=edim, M=M_ilu)
    x_ilu, residuals_ilu, niter_ilu, nrestart_ilu = solver_ilu.solve(
        b, tol=tol, maxiter=maxiter
    )
    residual_ilu = np.linalg.norm(A_sparse.dot(x_ilu) - b) / b_norm
    print(f"  Iterations: {niter_ilu}")
    print(f"  Restarts: {nrestart_ilu}")
    print(f"  Final residual: {residual_ilu:.2e}")

    print("\n--- Test 4: GCRO-DR with GMRES(5) preconditioning ---")

    def M_gmres(x):
        z, _ = gmres(A_sparse, x, restart=5, maxiter=5, atol=0, rtol=0)
        return z

    solver_gmres = gcro(n=n, A=A_matvec, kdim=kdim, edim=edim, M=M_gmres)
    x_gmres, residuals_gmres, niter_gmres, nrestart_gmres = solver_gmres.solve(
        b, tol=tol, maxiter=maxiter
    )
    residual_gmres = np.linalg.norm(A_sparse.dot(x_gmres) - b) / b_norm
    print(f"  Iterations: {niter_gmres}")
    print(f"  Restarts: {nrestart_gmres}")
    print(f"  Final residual: {residual_gmres:.2e}")

    assert (
        residual_noprecond < tol
    ), f"No-preconditioner residual too large: {residual_noprecond}"
    assert residual_ilu < tol, f"ILU residual too large: {residual_ilu}"
    assert residual_gmres < tol, f"GMRES residual too large: {residual_gmres}"

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6), layout="constrained")

        ax.semilogy(np.array(residuals_noprecond) / b_norm, "o-", label="No precond")
        ax.semilogy(np.array(residuals_ilu) / b_norm, "s-", label="ILU(0)")
        ax.semilogy(np.array(residuals_gmres) / b_norm, "^-", label="GMRES(5)")
        ax.axhline(y=tol, color="r", linestyle="--", label=f"Tolerance ({tol:.0e})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Relative Residual (||r|| / ||b||)")
        ax.set_title(f"Convergence Comparison ({nx}×{ny})")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.show()
    else:
        print("Plotting disabled (use --plot to enable)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GCRO-DR solver")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--plot", dest="plot", action="store_true", help="Enable plotting"
    )
    group.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        help="Disable plotting (default)",
    )
    parser.set_defaults(plot=False)
    args = parser.parse_args()

    print("=" * 70)
    print("GCRO-DR Solver Test Suite - Sparse Non-Symmetric Systems")
    print("=" * 70 + "\n")

    try:
        test_gcro_large_sparse_nonsym(plot=args.plot)
        test_with_preconditioning(plot=args.plot)

        print("=" * 70)
        print("All tests passed!")
        print("=" * 70 + "\n")

    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback

        traceback.print_exc()
