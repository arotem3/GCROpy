#!/usr/bin/env python3
"""
Example: GCRO solver for multiple right-hand sides.

Demonstrates solving a 2D advection-diffusion problem with multiple random
right-hand sides and plotting the convergence history across all solves.
"""

import numpy as np
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import gmres
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from gcro import gcro


def create_advection_diffusion_matrix_2d(
    nx, ny, diffusion=0.1, advection_x=1.0, advection_y=1.0
):
    """
    Create a 2D advection-diffusion operator:
    -ε*(d²u/dx² + d²u/dy²) + cx*du/dx + cy*du/dy + u = f
    on (0,1)x(0,1) with Dirichlet BCs.

    Parameters
    ----------
    nx : int
        Number of interior grid points in x-direction
    ny : int
        Number of interior grid points in y-direction
    diffusion : float
        Diffusion coefficient (ε)
    advection_x : float
        Advection coefficient in x-direction (cx)
    advection_y : float
        Advection coefficient in y-direction (cy)

    Returns
    -------
    A_sparse : scipy.sparse matrix
        Discretized operator
    """
    hx = 1.0 / (nx + 1)
    hy = 1.0 / (ny + 1)

    # 1D stencil in x for -ε*d²u/dx² + cx*du/dx
    diag_main_x = -2.0 * diffusion / (hx**2) * np.ones(nx)
    diag_upper_x = (diffusion / (hx**2) - advection_x / (2.0 * hx)) * np.ones(nx - 1)
    diag_lower_x = (diffusion / (hx**2) + advection_x / (2.0 * hx)) * np.ones(nx - 1)
    A_x = (
        diags(diag_lower_x, -1, shape=(nx, nx), format="csr")
        + diags(diag_main_x, 0, shape=(nx, nx), format="csr")
        + diags(diag_upper_x, 1, shape=(nx, nx), format="csr")
    )

    # 1D stencil in y for -ε*d²u/dy² + cy*du/dy
    diag_main_y = -2.0 * diffusion / (hy**2) * np.ones(ny)
    diag_upper_y = (diffusion / (hy**2) - advection_y / (2.0 * hy)) * np.ones(ny - 1)
    diag_lower_y = (diffusion / (hy**2) + advection_y / (2.0 * hy)) * np.ones(ny - 1)
    A_y = (
        diags(diag_lower_y, -1, shape=(ny, ny), format="csr")
        + diags(diag_main_y, 0, shape=(ny, ny), format="csr")
        + diags(diag_upper_y, 1, shape=(ny, ny), format="csr")
    )

    # Kronecker-sum structure for 2D operator plus reaction term +u.
    I_x = eye(nx, format="csr")
    I_y = eye(ny, format="csr")
    n_unknowns = nx * ny
    A_sparse = kron(I_y, A_x, format="csr") + kron(A_y, I_x, format="csr")
    A_sparse = A_sparse + eye(n_unknowns, format="csr")

    return A_sparse


def main():
    """Main example: solve 2D advection-diffusion with multiple random RHS."""

    # Problem setup
    nx, ny = 100, 150
    n_unknowns = nx * ny
    diffusion = 0.1
    advection_x = 4.0
    advection_y = 8.0
    num_rhs = 20  # Number of random right-hand sides

    print("=" * 70)
    print("GCRO-DR: Multiple Random Right-Hand Sides Example")
    print("=" * 70)
    print(f"Problem: 2D Advection-Diffusion")
    print(f"  Grid points: nx={nx}, ny={ny} (total={n_unknowns})")
    print(f"  Diffusion coefficient: {diffusion}")
    print(f"  Advection coefficients: cx={advection_x}, cy={advection_y}")
    print(f"  Number of RHS: {num_rhs}")
    print()

    # Create the matrix
    A_sparse = create_advection_diffusion_matrix_2d(
        nx, ny, diffusion, advection_x, advection_y
    )

    def A_matvec(x):
        return A_sparse.dot(x)

    # Storage for results
    all_residuals = []
    all_iterations = []
    all_restarts = []
    cumulative_iterations = [0]  # Track cumulative iteration count

    # GCRO-DR with frozen deflation space after first solve
    all_residuals_frozen = []
    all_iterations_frozen = []
    all_restarts_frozen = []
    cumulative_iterations_frozen = [0]

    # Storage for GMRES comparison
    gmres_iterations = []
    gmres_residuals_first = None  # Store residuals for first solve

    # Solver parameters
    kdim = 40
    edim = 20
    tol = 1e-6
    maxiter = 10000

    print(f"Solver parameters: kdim={kdim}, edim={edim}, tol={tol:.0e}")
    print()

    # Create solvers once.
    # solver_update: updates deflation space after each solve/restart.
    # solver_frozen: updates on the first solve, then keeps same space for remaining RHS.
    solver_update = gcro(n=n_unknowns, A=A_matvec, kdim=kdim, edim=edim)
    solver_frozen = gcro(n=n_unknowns, A=A_matvec, kdim=kdim, edim=edim)

    # Solve each right-hand side
    np.random.seed(42)
    for i in range(num_rhs):
        print(f"Solve {i+1}/{num_rhs}:")

        # Generate random right-hand side
        b = np.random.randn(n_unknowns)
        b_norm = np.linalg.norm(b)

        # Solve with GCRO-DR (continual deflation updates)
        x_approx, residuals, niter, nrestart = solver_update.solve(
            b, tol=tol, maxiter=maxiter
        )
        residual_final = np.linalg.norm(A_sparse.dot(x_approx) - b) / b_norm

        print(f"  GCRO-DR (updating) - Iterations: {niter}, Restarts: {nrestart}")
        print(f"    Final relative residual: {residual_final:.2e}")

        # Solve with GCRO-DR using a frozen deflation space after the first solve.
        x_frozen, residuals_frozen, niter_frozen, nrestart_frozen = solver_frozen.solve(
            b, tol=tol, maxiter=maxiter
        )
        residual_final_frozen = np.linalg.norm(A_sparse.dot(x_frozen) - b) / b_norm
        print(
            f"  GCRO-DR (frozen after first solve) - Iterations: {niter_frozen}, Restarts: {nrestart_frozen}"
        )
        print(f"    Final relative residual: {residual_final_frozen:.2e}")

        if i == 0:
            solver_frozen.set_update_deflation_space(False)
            print("    Deflation updates disabled for subsequent solves.")

        # Compare with GMRES(m) - restart every kdim iterations
        # Initial normalized residual is 1.0 for zero initial guess
        gmres_residuals_list = [1.0]  # Start with initial normalized residual

        def gmres_callback(pr_norm):
            # pr_norm is ||r|| / ||b|| computed by GMRES at each iteration
            gmres_residuals_list.append(pr_norm)

        x_gmres, info = gmres(
            A_sparse,
            b,
            restart=kdim,
            maxiter=maxiter,
            atol=0,
            rtol=tol,
            callback=gmres_callback,
            callback_type="pr_norm",
        )
        gmres_iter = len(gmres_residuals_list) - 1  # Don't count initial residual

        # Verify GMRES solution
        gmres_residual_final = np.linalg.norm(A_sparse.dot(x_gmres) - b) / b_norm
        print(f"  GMRES({kdim}) - Iterations: {gmres_iter}")
        print(f"  Final relative residual: {gmres_residual_final:.2e}")

        # Store results
        all_residuals.append(np.array(residuals) / b_norm)
        all_iterations.append(niter)
        all_restarts.append(nrestart)
        all_residuals_frozen.append(np.array(residuals_frozen) / b_norm)
        all_iterations_frozen.append(niter_frozen)
        all_restarts_frozen.append(nrestart_frozen)
        gmres_iterations.append(gmres_iter)
        cumulative_iterations.append(cumulative_iterations[-1] + len(residuals))
        cumulative_iterations_frozen.append(
            cumulative_iterations_frozen[-1] + len(residuals_frozen)
        )

        # Store residuals for first solve (already normalized by pr_norm)
        if i == 0:
            gmres_residuals_first = np.array(gmres_residuals_list)

    print()

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), constrained_layout=True)

    # Top-left: Residual histories side-by-side with cumulative iteration count
    ax1 = axes[0, 0]

    # Plot GCRO-DR with deflation updates across all solves
    for i in range(num_rhs):
        iter_start = cumulative_iterations[i]
        iter_count = len(all_residuals[i])
        iter_indices = np.arange(iter_start, iter_start + iter_count)

        ax1.semilogy(
            iter_indices,
            all_residuals[i],
            color="C0",
            linewidth=1.5,
            alpha=0.7,
        )

        # Keep label height consistent, slightly above the tolerance line.
        y_label = tol * 1.12
        ax1.text(
            iter_indices[-1] + 0.25,
            y_label,
            f"{i + 1}",
            color="C0",
            fontsize=8,
            alpha=0.85,
            ha="left",
            va="bottom",
        )

    ax1.axhline(y=tol, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax1.set_xlabel("Cumulative Iteration", fontsize=12)
    ax1.set_ylabel("Relative Residual ||r||/||b||", fontsize=12)
    ax1.set_title(
        f"GCRO-DR({kdim},{edim}) Convergence: All Solves Side-by-Side",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([1e-8, 2])

    # Top-right: First solve comparison (GCRO-DR vs GMRES)
    ax2 = axes[0, 1]
    ax2.semilogy(
        all_residuals[0],
        color="C0",
        linewidth=2,
        label=f"GCRO-DR({kdim},{edim})",
        alpha=0.8,
    )
    if gmres_residuals_first is not None:
        ax2.semilogy(
            gmres_residuals_first,
            color="C1",
            linewidth=2,
            label=f"GMRES({kdim})",
            alpha=0.8,
        )
    ax2.axhline(y=tol, color="k", linestyle="--", linewidth=1, alpha=0.5)
    ax2.set_xlabel("Iteration", fontsize=12)
    ax2.set_ylabel("Relative Residual ||r||/||b||", fontsize=12)
    ax2.set_title("First Solve: GCRO-DR vs GMRES", fontsize=13, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([1e-8, 2])

    # Zoomed inset: first m iterations of first solve
    if gmres_residuals_first is not None:
        m_zoom_target = kdim * 3
        m_zoom = min(m_zoom_target, len(all_residuals[0]), len(gmres_residuals_first))
        if m_zoom > 0:
            gcro_zoom = all_residuals[0][:m_zoom]
            gmres_zoom = gmres_residuals_first[:m_zoom]
            it_zoom = np.arange(1, m_zoom + 1)

            axins = inset_axes(
                ax2,
                width="34%",
                height="34%",
                loc="lower right",
                bbox_to_anchor=(0.0, 0.08, 1.0, 1.0),
                bbox_transform=ax2.transAxes,
                borderpad=0.8,
            )
            axins.patch.set_facecolor("white")
            axins.patch.set_alpha(0.55)
            axins.semilogy(
                it_zoom,
                gcro_zoom,
                color="C0",
                linewidth=1.5,
            )
            axins.semilogy(
                it_zoom,
                gmres_zoom,
                color="C1",
                linewidth=1.5,
                alpha=0.9,
            )
            axins.set_xlim(1, m_zoom)

            m_line = min(kdim, m_zoom)
            axins.axvline(
                m_line, color="0.25", linestyle="--", linewidth=1.0, alpha=0.8
            )

            y_min = max(1e-14, min(np.min(gcro_zoom), np.min(gmres_zoom)) * 0.8)
            y_max = min(2.0, max(np.max(gcro_zoom), np.max(gmres_zoom)) * 1.2)
            if y_min < y_max:
                axins.set_ylim(y_min, y_max)

            # Keep inset readable while still showing axis labels and ticks.
            xticks = sorted(set([1, m_line, max(1, m_zoom // 2), m_zoom]))
            axins.set_xticks(xticks)
            axins.set_xlabel("Iter", fontsize=7, labelpad=1)
            axins.tick_params(axis="both", labelsize=7)

    # Bottom-left: Iterations per RHS index - comparison with GMRES
    ax3 = axes[1, 0]
    rhs_indices = np.arange(1, num_rhs + 1)

    ax3.plot(
        rhs_indices,
        all_iterations,
        marker="o",
        markersize=8,
        linewidth=2,
        color="C0",
        label=f"GCRO-DR({kdim},{edim})",
    )

    ax3.plot(
        rhs_indices,
        all_iterations_frozen,
        marker="^",
        markersize=7,
        linewidth=2,
        color="C2",
        label=f"GCRO-DR({kdim},{edim}) frozen after first solve",
    )

    ax3.plot(
        rhs_indices,
        gmres_iterations,
        marker="s",
        markersize=8,
        linewidth=2,
        color="C1",
        label=f"GMRES({kdim})",
    )

    ax3.set_xlabel("RHS Index", fontsize=12)
    ax3.set_ylabel("Iterations to Convergence", fontsize=12)
    ax3.set_title(
        "GCRO-DR vs GMRES: Iterations per RHS", fontsize=13, fontweight="bold"
    )
    ax3.set_xticks(rhs_indices)
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Bottom-right: Cumulative iterations as a function of RHS index
    ax4 = axes[1, 1]
    cumulative_update = np.cumsum(all_iterations)
    cumulative_frozen = np.cumsum(all_iterations_frozen)
    cumulative_gmres = np.cumsum(gmres_iterations)

    ax4.plot(
        rhs_indices,
        cumulative_update,
        marker="o",
        markersize=8,
        linewidth=2,
        color="C0",
        label=f"GCRO-DR({kdim},{edim})",
    )

    ax4.plot(
        rhs_indices,
        cumulative_frozen,
        marker="^",
        markersize=7,
        linewidth=2,
        color="C2",
        label=f"GCRO-DR({kdim},{edim}) frozen after first solve",
    )

    ax4.plot(
        rhs_indices,
        cumulative_gmres,
        marker="s",
        markersize=8,
        linewidth=2,
        color="C1",
        label=f"GMRES({kdim})",
    )

    ax4.set_xlabel("RHS Index", fontsize=12)
    ax4.set_ylabel("Cumulative Iterations", fontsize=12)
    ax4.set_title("Cumulative Iterations vs RHS", fontsize=13, fontweight="bold")
    ax4.set_xticks(rhs_indices)
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3)

    # Summary statistics
    def compute_iteration_stats(values):
        arr = np.asarray(values, dtype=float)
        q0, q25, q50, q75, q100 = np.quantile(arr, [0.0, 0.25, 0.5, 0.75, 1.0])
        return {"min": q0, "q25": q25, "med": q50, "q75": q75, "max": q100}

    def fmt_iter(v):
        return f"{int(round(v))}"

    def make_boxplot_bar(stats, global_min, global_max, width):
        width = max(12, int(width))

        if global_max == global_min:
            chars = [" "] * (width + 1)
            center = width // 2
            chars[center] = "|"
            return "".join(chars)

        def pos(v):
            return int(round((v - global_min) / (global_max - global_min) * width))

        pmin = max(0, min(width, pos(stats["min"])))
        pmax = max(0, min(width, pos(stats["max"])))
        if pmax < pmin:
            pmin, pmax = pmax, pmin

        chars = [" "] * (width + 1)

        # Draw only inside each solver's [min, max] span on a shared global canvas.
        for i in range(pmin, pmax + 1):
            chars[i] = "-"

        if pmax == pmin:
            chars[pmin] = "|"
            return "".join(chars)

        local_low = pmin + 1
        local_high = pmax - 1

        p25 = max(local_low, min(local_high, pos(stats["q25"])))
        p50 = max(local_low, min(local_high, pos(stats["med"])))
        p75 = max(local_low, min(local_high, pos(stats["q75"])))

        if local_high - local_low >= 2:
            # Enforce p25 < p50 < p75 to avoid marker overwrites.
            p50 = max(p50, p25)
            p75 = max(p75, p50)
            if p50 <= p25:
                p50 = p25 + 1
            if p75 <= p50:
                p75 = p50 + 1

            if p75 > local_high:
                shift = p75 - local_high
                p25 -= shift
                p50 -= shift
                p75 -= shift
            if p25 < local_low:
                shift = local_low - p25
                p25 += shift
                p50 += shift
                p75 += shift

            p25 = min(local_high - 2, max(local_low, p25))
            p50 = min(local_high - 1, max(p25 + 1, p50))
            p75 = min(local_high, max(p50 + 1, p75))
        elif local_high - local_low == 1:
            p25 = local_low
            p50 = local_low
            p75 = local_high
        else:
            p25 = local_low
            p50 = local_low
            p75 = local_low

        for i in range(p25, p75 + 1):
            chars[i] = "="

        chars[pmin] = "|"
        chars[pmax] = "|"
        chars[p25] = "["
        chars[p75] = "]"
        chars[p50] = "M"

        return "".join(chars)

    solver_stats = [
        ("GCRO-DR", compute_iteration_stats(all_iterations)),
        (
            "GCRO-DR frozen",
            compute_iteration_stats(all_iterations_frozen),
        ),
        (f"GMRES({kdim})", compute_iteration_stats(gmres_iterations)),
    ]

    all_iter_values = np.concatenate(
        [
            np.asarray(all_iterations, dtype=float),
            np.asarray(all_iterations_frozen, dtype=float),
            np.asarray(gmres_iterations, dtype=float),
        ]
    )
    global_min = float(np.min(all_iter_values))
    global_max = float(np.max(all_iter_values))
    global_range = global_max - global_min

    # Choose total canvas width so every non-degenerate solver range gets
    # at least `min_plot_length` characters for its [min,max] span.
    min_plot_length = 16
    required_widths = []
    if global_range > 0:
        for _, stats in solver_stats:
            span = stats["max"] - stats["min"]
            if span > 0:
                required_widths.append(
                    int(np.ceil(min_plot_length * global_range / span))
                )

    total_plot_width = max([40, min_plot_length + 2] + required_widths)

    print("=" * 70)
    print("Summary:")
    print(f"  GCRO-DR:")
    print(
        f"    Iteration stats: med={fmt_iter(solver_stats[0][1]['med'])}, q25={fmt_iter(solver_stats[0][1]['q25'])}, q75={fmt_iter(solver_stats[0][1]['q75'])}, min={fmt_iter(solver_stats[0][1]['min'])}, max={fmt_iter(solver_stats[0][1]['max'])}"
    )
    print(f"    Min/Max iterations: {min(all_iterations)} / {max(all_iterations)}")
    print(f"    Average restarts: {np.mean(all_restarts):.1f}")
    print(f"    Total iterations: {sum(all_iterations)}")
    print(f"  GCRO-DR (frozen after first solve):")
    print(
        f"    Iteration stats: med={fmt_iter(solver_stats[1][1]['med'])}, q25={fmt_iter(solver_stats[1][1]['q25'])}, q75={fmt_iter(solver_stats[1][1]['q75'])}, min={fmt_iter(solver_stats[1][1]['min'])}, max={fmt_iter(solver_stats[1][1]['max'])}"
    )
    print(
        f"    Min/Max iterations: {min(all_iterations_frozen)} / {max(all_iterations_frozen)}"
    )
    print(f"    Average restarts: {np.mean(all_restarts_frozen):.1f}")
    print(f"    Total iterations: {sum(all_iterations_frozen)}")
    print(f"  GMRES({kdim}):")
    print(
        f"    Iteration stats: med={fmt_iter(solver_stats[2][1]['med'])}, q25={fmt_iter(solver_stats[2][1]['q25'])}, q75={fmt_iter(solver_stats[2][1]['q75'])}, min={fmt_iter(solver_stats[2][1]['min'])}, max={fmt_iter(solver_stats[2][1]['max'])}"
    )
    print(f"    Min/Max iterations: {min(gmres_iterations)} / {max(gmres_iterations)}")
    print(f"    Total iterations: {sum(gmres_iterations)}")

    print("  Iteration box plots (common scale across solvers):")
    print(f"    Global min/max: {fmt_iter(global_min)} / {fmt_iter(global_max)}")
    label_width = max(len(name) for name, _ in solver_stats)
    min_label_width = max(len(fmt_iter(stats["min"])) for _, stats in solver_stats) + 2
    max_label_width = max(len(fmt_iter(stats["max"])) for _, stats in solver_stats) + 2
    for name, stats in solver_stats:
        bar = make_boxplot_bar(stats, global_min, global_max, width=total_plot_width)
        min_text = fmt_iter(stats["min"])
        max_text = fmt_iter(stats["max"])
        print(
            f"    {name:<{label_width}}: {min_text:>{min_label_width}}  {bar}  {max_text:<{max_label_width}}"
        )

    speedup_update = sum(gmres_iterations) / sum(all_iterations)
    speedup_frozen = sum(gmres_iterations) / sum(all_iterations_frozen)
    print(f"  Speedup: {speedup_update:.2f}x (GCRO-DR updating vs GMRES)")
    print(f"  Speedup: {speedup_frozen:.2f}x (GCRO-DR frozen vs GMRES)")
    print("=" * 70)

    plt.show()


if __name__ == "__main__":
    np.random.seed(42)
    main()
