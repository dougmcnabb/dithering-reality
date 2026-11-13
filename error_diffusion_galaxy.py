"""
3D Isotropic Error-Diffusion Quantization of a Continuous Spherical Wave
========================================================================
Reproduces the 3D toy model accompanying "What Happens When You Dither a Galaxy?"
Requires only NumPy:
    python error_diffusion_galaxy.py
"""

import numpy as np


def run_3d_galaxy_error_diffusion(
    N: int = 65,
    r0: float = 3.25,
    A: float = 0.95,
    alpha: float = 0.75,
    seed: int = 7,
):
    rng = np.random.default_rng(seed)
    zz, yy, xx = np.mgrid[0:N, 0:N, 0:N]
    c = N // 2
    rr = np.sqrt((xx - c) ** 2 + (yy - c) ** 2 + (zz - c) ** 2)

    # Pure 3D spherical wave intensity |psi(R)|^2 = A / (1 + (R / r0)^2)
    rho = A / (1.0 + (rr / r0) ** 2)

    buf = rho.copy()
    knots = np.zeros_like(rho)
    rem = np.zeros_like(rho)
    visited = np.zeros(rho.shape, dtype=bool)

    nbrs = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]

    for idx in rng.permutation(rho.size):
        z, rem_idx = divmod(idx, N * N)
        y, x = divmod(rem_idx, N)
        visited[z, y, x] = True
        val = buf[z, y, x]
        k = 1.0 if val >= 0.5 else 0.0
        knots[z, y, x] = k
        err = val - k

        unvis = []
        for dz, dy, dx in nbrs:
            nz, ny, nx = z + dz, y + dy, x + dx
            if 0 <= nz < N and 0 <= ny < N and 0 <= nx < N and not visited[nz, ny, nx]:
                unvis.append((nz, ny, nx))

        m = len(unvis)
        if m > 0:
            d = (alpha * err) / m
            rem[z, y, x] = (1.0 - alpha) * err
            for nz, ny, nx in unvis:
                buf[nz, ny, nx] += d
        else:
            rem[z, y, x] = err

    tot = knots + rem
    sphere = rr <= (N // 2 - 1)
    print(f"Grid: {N}^3 voxels | r0={r0} | alpha={alpha} | seed={seed}")
    print(f"Exact Conservation Drift : {abs(rho.sum() - tot.sum()):.3e}")
    print(f"Virial Baryon (1-knots)  : {100 * knots[sphere].sum() / tot[sphere].sum():.2f}%")
    print(f"Virial Dark Residual     : {100 * rem[sphere].sum() / tot[sphere].sum():.2f}%")
    print("\n  R  | M_bary(<R) | M_dark(<R) | M_tot(<R) | v_baryon | v_total")
    print("-" * 66)
    for r in [3, 6, 9, 12, 16, 20, 24, 28, 31]:
        mask = rr <= r
        mb = knots[mask].sum()
        md = rem[mask].sum()
        mt = tot[mask].sum()
        vb = np.sqrt(max(mb, 0.0) / r)
        vt = np.sqrt(max(mt, 0.0) / r)
        print(f"r={r:2d} | {mb:10.1f} | {md:10.1f} | {mt:9.1f} |  {vb:6.3f}  |  {vt:6.3f}")


if __name__ == "__main__":
    run_3d_galaxy_error_diffusion()
