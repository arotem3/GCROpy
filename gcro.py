import numpy as np
from numpy.linalg import qr, norm
from scipy.linalg import schur, qz, solve


def compute_givens(x, y):
    rho = np.hypot(x, y)
    if rho == 0.0:
        return 1.0, 0.0
    return x / rho, y / rho


def apply_givens(x, y, c, s):
    x_new = c * x + s * y
    y_new = -s * x + c * y
    return x_new, y_new


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

    def _fresh_ritz(self, H, k):
        m = H.shape[1]
        Hm = H[:m, :m]
        em = np.zeros(m, dtype=np.float64)
        em[-1] = 1.0
        hm = solve(Hm.T, em)
        Hm = Hm + H[m, m - 1] ** 2 * np.outer(hm, em)
        schur_result = schur(Hm, output="real")
        T, Z = schur_result[0], schur_result[1]

        # Extract eigenvalues from real Schur form (diagonal/2x2 blocks)
        evals = self._extract_real_evals(T)
        idx = np.argsort(np.abs(evals))[: k + 1]
        evecs = self._extract_real_evecs(Z, T, idx)

        return evecs

    def _compute_ritz_invariant_space(self, H, W, Z, k):
        A = H.T @ H
        B = H.T @ (W.T @ Z)

        # Generalized Schur form
        result = qz(A, B, output="real")
        S, T, Z = result[0], result[1], result[2]

        # Extract generalized eigenvalues
        evals = self._extract_generalized_evals(S, T)
        idx = np.argsort(np.abs(evals))[: k + 1]
        evecs = self._extract_real_evecs(Z, S, idx)

        return evecs

    def _extract_real_evals(self, T):
        """Extract eigenvalues from real Schur form."""
        evals = []
        i = 0
        while i < T.shape[0]:
            if i + 1 < T.shape[0] and abs(T[i + 1, i]) > 1e-14:
                # 2x2 block: complex conjugate pair
                trace = T[i, i] + T[i + 1, i + 1]
                det = T[i, i] * T[i + 1, i + 1] - T[i, i + 1] * T[i + 1, i]
                lam_real = trace / 2.0
                lam_imag = np.sqrt(det - (trace / 2.0) ** 2)
                lam = np.sqrt(lam_real**2 + lam_imag**2)
                evals.append(lam)
                evals.append(lam)
                i += 2
            else:
                # 1x1 block: real eigenvalue
                evals.append(abs(T[i, i]))
                i += 1
        return np.array(evals)

    def _extract_generalized_evals(self, S, T):
        """Extract generalized eigenvalues from real generalized Schur form."""
        evals = []
        i = 0
        while i < S.shape[0]:
            if i + 1 < S.shape[0] and abs(S[i + 1, i]) > 1e-14:
                # 2x2 complex pair
                s_trace = S[i, i] + S[i + 1, i + 1]
                t_trace = T[i, i] + T[i + 1, i + 1]
                eps = 1e-14
                if abs(t_trace) > eps:
                    lam = abs(s_trace / t_trace)
                else:
                    lam = abs(s_trace)
                evals.append(lam)
                evals.append(lam)
                i += 2
            else:
                # 1x1 real eigenvalue
                eps = 1e-14
                if abs(T[i, i]) > eps:
                    lam = abs(S[i, i] / T[i, i])
                else:
                    lam = abs(S[i, i])
                evals.append(lam)
                i += 1
        return np.array(evals)

    def _extract_real_evecs(self, Z, T, idx):
        """Extract real eigenvectors from real Schur form."""
        # Ensure complex conjugate pairs are not split
        idx_list = list(idx)
        i = 0
        block_idx = 0
        while i < T.shape[0]:
            if i + 1 < T.shape[0] and abs(T[i + 1, i]) > 1e-14:
                # Complex pair: if either index is selected, select both
                if block_idx in idx_list or block_idx + 1 in idx_list:
                    if block_idx not in idx_list:
                        idx_list.append(block_idx)
                    if block_idx + 1 not in idx_list:
                        idx_list.append(block_idx + 1)
                block_idx += 2
                i += 2
            else:
                block_idx += 1
                i += 1

        # Now extract with complete pairs
        evecs = []
        for j in sorted(idx_list):
            i = 0
            block_idx = 0
            while i < T.shape[0]:
                if i + 1 < T.shape[0] and abs(T[i + 1, i]) > 1e-14:
                    if block_idx == j or block_idx + 1 == j:
                        if block_idx == j:
                            evecs.append(Z[:, i])
                        else:
                            evecs.append(Z[:, i + 1])
                        break
                    block_idx += 2
                    i += 2
                else:
                    if block_idx == j:
                        evecs.append(Z[:, i])
                        break
                    block_idx += 1
                    i += 1
        return np.column_stack(evecs) if evecs else np.zeros((Z.shape[0], 0))

    def solve(self, b, x0=None, tol=1e-6, maxiter=1000):
        x = x0.copy() if x0 is not None else np.zeros(self.n, dtype=np.float64)

        r = b - self.A(x)

        beta = norm(r)
        bnrm = norm(b)
        if bnrm == 0:
            return x, [0.0], 0, 0
        target_tol = tol * bnrm
        print(target_tol)

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
                Q, R = qr(H @ P, mode="reduced")
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
