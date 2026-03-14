import numpy as np
from scipy.linalg import solve, ordqz, qr, norm


def compute_givens(x, y):
    rho = np.hypot(x, y)
    if rho == 0.0:
        return 1.0, 0.0
    return x / rho, y / rho


def apply_givens(x, y, c, s):
    x_new = c * x + s * y
    y_new = -s * x + c * y
    return x_new, y_new


def invariant_subspace(A, B, k):
    """Compute an invariant subspace of A with respect to B using the generalized Schur decomposition."""

    def select(alpha, beta):
        """Select eigenvalues based on magnitude, keeping conjugate pairs together."""
        n = len(alpha)

        # Use a relative threshold for generalized eigenvalues to avoid scale issues.
        scale = np.max(np.abs(beta)) if n > 0 else 1.0
        beta_tol = 1e-14 * max(1.0, scale)
        eigs = [a / b if abs(b) > beta_tol else np.inf for a, b in zip(alpha, beta)]
        blocks = []

        tol = 1e-12

        i = 0
        while i < n:
            mag = abs(eigs[i])

            if alpha[i].imag != 0:
                blocks.append((mag, i, 2))  # complex conjugate pair
                i += 2
            else:
                blocks.append((mag, i, 1))  # real eigenvalue
                i += 1

        blocks.sort(key=lambda x: x[0])

        count = 0
        J = np.zeros(n, dtype=bool)
        for _, idx, block_size in blocks:
            if count >= k:
                break

            if block_size == 2:
                # Keep conjugate pairs together; allow selecting k+1 only in this case.
                if count + 2 <= k or count == k - 1:
                    J[idx] = True
                    J[idx + 1] = True
                    count += 2
            else:
                J[idx] = True
                count += 1

        return J

    result = ordqz(A, B, sort=select, output="real")
    alpha, beta = result[2], result[3]
    selected_count = int(np.count_nonzero(select(alpha, beta)))
    assert (
        selected_count <= k + 1
    ), "Selected more eigenvalues than requested, this should not happen"

    return result[-1][:, :selected_count]


class gcro:
    """Flexible GCRO with Ritz deflation and optional (right) preconditioning.

    References:
    - Parks, M. L., de Sturler, E., Mackey, G., Johnson, D. D., & Maiti, S. (2006).
      Recycling Krylov Subspaces for Sequences of Linear Systems.
      SIAM Journal on Scientific Computing, 28(5), 1651–1674. https://doi.org/10.1137/040607277
    """

    def __init__(self, n: int, A, kdim: int = 40, edim: int = 20, M=None):
        self.n = n
        self.A = A
        self.kdim = kdim
        self.edim = edim
        self.M = (
            M if M is not None else lambda x: x
        )  # Identity preconditioner by default

        assert n >= 0
        assert kdim >= 0
        assert edim >= 0
        assert (
            edim + 1 < kdim
        ), "edim must be strictly less than kdim-1 to allow for conjugate eigenvalue pairs"

        # First n_active columns are deflation vectors, rest are additional Krylov vectors
        self.W = np.zeros((n, kdim + 1), dtype=np.float64)
        self.Z = np.zeros((n, kdim + 1), dtype=np.float64)
        self.n_active = 0
        self.update_deflation_space = True

        # Single Hessenberg/G work buffer (rows are rotated in-place by Givens).
        self.H = np.zeros(
            (kdim + 1, kdim), dtype=np.float64
        )  # upper Hessenberg from Arnoldi
        self.Hqr = np.zeros(
            (kdim + 1, kdim), dtype=np.float64
        )  # separate buffer for QR of H, we do this just so that we can update the residual at every iteration
        self.eta = np.zeros(kdim + 1, dtype=np.float64)  # RHS for least-squares problem

    def set_update_deflation_space(self, enabled):
        """Enable or disable updates of the recycled deflation space."""
        self.update_deflation_space = bool(enabled)

    def _arnoldi(self, x, r, m, target_tol):
        k = self.n_active
        m = min(m, self.kdim - k)

        W = self.W[:, : k + m + 1]
        Z = self.Z[:, : k + m]

        H = self.H[: k + m + 1, : k + m]
        Hqr = self.Hqr[: k + m + 1, : k + m]
        eta = self.eta[: k + m + 1]

        cs = np.zeros(m, dtype=np.float64)
        sn = np.zeros(m, dtype=np.float64)

        H[:] = 0.0
        Hqr[:] = 0.0
        eta[:] = 0.0

        # set the sub-block of H associated with the deflation space to identity
        if k > 0:
            H[:k, :k] = np.eye(k)
            Hqr[:k, :k] = np.eye(k)

        beta = norm(r)
        if beta < 1e-14:
            return x, W[:, : k + 1], Z[:, :k], H[: k + 1, :k], 0, []

        W[:, k] = r / beta
        eta[k] = beta
        res_hist = []

        # actual Krylov dimension incase we stop early
        m_actual = 0

        for j in range(m):
            # col: column index for W and Z offset by deflation space occupying first k columns
            col = k + j
            m_actual = j + 1

            Z[:, col] = self.M(W[:, col])
            w = self.A(Z[:, col])

            # modified Gram-Schmidt
            for i in range(0, col + 1):
                hij = np.vdot(W[:, i], w)

                H[i, col] = hij
                Hqr[i, col] = hij

                w -= hij * W[:, i]

            wnrm = norm(w)
            H[col + 1, col] = wnrm
            Hqr[col + 1, col] = wnrm

            if wnrm <= 1e-14 * beta:
                print("Warning: possible breakdown encountered.")
                m_actual -= 1  # discard this Krylov vector and stop early
                break

            W[:, col + 1] = w / wnrm

            # Update the QR factorization of H
            for i in range(j):
                row = k + i
                Hqr[row, col], Hqr[row + 1, col] = apply_givens(
                    Hqr[row, col], Hqr[row + 1, col], cs[i], sn[i]
                )

            cs[j], sn[j] = compute_givens(Hqr[col, col], Hqr[col + 1, col])
            Hqr[col, col], Hqr[col + 1, col] = apply_givens(
                Hqr[col, col], Hqr[col + 1, col], cs[j], sn[j]
            )
            eta[col], eta[col + 1] = apply_givens(eta[col], eta[col + 1], cs[j], sn[j])

            res_norm = abs(eta[col + 1])
            # print(f"||r||: {res_norm:10.2e}")
            res_hist.append(res_norm)

            if res_norm <= target_tol:
                break

        ncol = k + m_actual

        H = H[: ncol + 1, :ncol]
        eta = eta[: ncol + 1]

        W = W[:, : ncol + 1]
        Z = Z[:, :ncol]

        # solve least squares
        y = solve(
            Hqr[:ncol, :ncol], eta[:ncol], assume_a="upper triangular", overwrite_a=True
        )
        x += Z @ y

        return x, W, Z, H, m_actual, res_hist

    def _compute_ritz_invariant_space(self, H, W, Z, k):
        A = H.T @ H
        B = H.T @ (W.T @ Z)

        return invariant_subspace(A, B, k)

    def solve(self, b, x0=None, tol=1e-6, maxiter=1000):
        x = x0.copy() if x0 is not None else np.zeros(self.n, dtype=np.float64)

        r = b - self.A(x)

        beta = norm(r)
        bnrm = norm(b)
        if bnrm == 0:
            return x, [0.0], 0, 0
        target_tol = tol * bnrm

        if beta <= target_tol:
            return x, [beta], 0, 0

        nit = 0
        restarts = 0

        if self.n_active > 0:
            C = self.W[:, : self.n_active]
            Z = self.Z[:, : self.n_active]
            cr = C.T @ r
            x = x + Z @ cr
            r = r - C @ cr
            beta = norm(r)

        res = [beta]

        while beta > target_tol and nit < maxiter:
            m = min(self.kdim - self.n_active, maxiter - nit)
            x, W, Z, H, m, arnoldi_res = self._arnoldi(x, r, m, target_tol)

            res.extend(arnoldi_res)
            beta = res[-1]

            r = b - self.A(x)

            if self.update_deflation_space:
                P = self._compute_ritz_invariant_space(H, W, Z, self.edim)
                Q, R = qr(H @ P, mode="economic")
                P = solve(
                    R,
                    P.T,
                    assume_a="upper triangular",
                    transposed=True,
                    overwrite_a=True,
                    overwrite_b=True,
                ).T  # P <- P * inv(R)

                self.n_active = Q.shape[1]
                self.W[:, : self.n_active] = W @ Q
                self.Z[:, : self.n_active] = Z @ P  # Store preconditioned vectors

            nit += m
            restarts += 1

        return (
            x,
            res,
            nit,
            restarts,
        )
