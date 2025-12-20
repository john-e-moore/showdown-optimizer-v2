from __future__ import annotations

import numpy as np

try:
    import numba as nb
except Exception as e:  # pragma: no cover
    raise ImportError(
        "numba is required for showdown lineup enumeration. Install with `pip install numba`."
    ) from e


@nb.njit(parallel=True, cache=True)
def count_valid_per_cpt(
    salary: np.ndarray,
    team01: np.ndarray,
    *,
    salary_cap: int,
) -> np.ndarray:
    """
    Count valid 6-player showdown lineups per CPT index using nested loops.

    Rules enforced:
    - CPT + 5 UTIL distinct players
    - salary_used <= salary_cap where CPT costs 1.5x
    - stack legality: reject 6-0 (all 6 from one team); team01 is 0/1.

    Returns counts_per_cpt: int64[n]
    """
    n = int(salary.shape[0])
    out = np.zeros(n, dtype=np.int64)

    for c in nb.prange(n):
        # integer 1.5x: DK salaries should be even multiples (typically 100s), so 3//2 is exact.
        cpt_salary = (salary[c] * 3) // 2
        if cpt_salary > salary_cap:
            continue
        count = 0

        for i in range(n - 4):
            if i == c:
                continue
            s_ci = cpt_salary + salary[i]
            if s_ci > salary_cap:
                continue
            for j in range(i + 1, n - 3):
                if j == c:
                    continue
                s_cij = s_ci + salary[j]
                if s_cij > salary_cap:
                    continue
                for k in range(j + 1, n - 2):
                    if k == c:
                        continue
                    s_cijk = s_cij + salary[k]
                    if s_cijk > salary_cap:
                        continue
                    for l in range(k + 1, n - 1):
                        if l == c:
                            continue
                        s_cijkl = s_cijk + salary[l]
                        if s_cijkl > salary_cap:
                            continue
                        for m in range(l + 1, n):
                            if m == c:
                                continue

                            salary_used = s_cijkl + salary[m]
                            if salary_used > salary_cap:
                                continue

                            t = (
                                int(team01[c])
                                + int(team01[i])
                                + int(team01[j])
                                + int(team01[k])
                                + int(team01[l])
                                + int(team01[m])
                            )
                            if t == 0 or t == 6:
                                continue

                            count += 1

        out[c] = count

    return out


@nb.njit(parallel=True, cache=True)
def fill_lineups(
    salary: np.ndarray,
    proj_points: np.ndarray,
    team01: np.ndarray,
    offsets: np.ndarray,
    counts_per_cpt: np.ndarray,
    *,
    salary_cap: int,
    cpt_out: np.ndarray,
    u1_out: np.ndarray,
    u2_out: np.ndarray,
    u3_out: np.ndarray,
    u4_out: np.ndarray,
    u5_out: np.ndarray,
    salary_used_out: np.ndarray,
    salary_left_out: np.ndarray,
    proj_points_out: np.ndarray,
    stack_code_out: np.ndarray,
    written_per_cpt_out: np.ndarray,
) -> None:
    """
    Fill preallocated output arrays for valid lineups.

    offsets[c] gives start index for CPT c in all output arrays.
    counts_per_cpt[c] must match the number of lineups generated for c.
    """
    n = int(salary.shape[0])

    for c in nb.prange(n):
        cpt_salary = (salary[c] * 3) // 2
        cpt_proj = 1.5 * float(proj_points[c])
        if cpt_salary > salary_cap:
            continue

        write_idx = int(offsets[c])
        written = 0

        for i in range(n - 4):
            if i == c:
                continue
            s_ci = cpt_salary + salary[i]
            if s_ci > salary_cap:
                continue
            p_ci = cpt_proj + float(proj_points[i])
            for j in range(i + 1, n - 3):
                if j == c:
                    continue
                s_cij = s_ci + salary[j]
                if s_cij > salary_cap:
                    continue
                p_cij = p_ci + float(proj_points[j])
                for k in range(j + 1, n - 2):
                    if k == c:
                        continue
                    s_cijk = s_cij + salary[k]
                    if s_cijk > salary_cap:
                        continue
                    p_cijk = p_cij + float(proj_points[k])
                    for l in range(k + 1, n - 1):
                        if l == c:
                            continue
                        s_cijkl = s_cijk + salary[l]
                        if s_cijkl > salary_cap:
                            continue
                        p_cijkl = p_cijk + float(proj_points[l])
                        for m in range(l + 1, n):
                            if m == c:
                                continue

                            salary_used = s_cijkl + salary[m]
                            if salary_used > salary_cap:
                                continue

                            t = (
                                int(team01[c])
                                + int(team01[i])
                                + int(team01[j])
                                + int(team01[k])
                                + int(team01[l])
                                + int(team01[m])
                            )
                            if t == 0 or t == 6:
                                continue

                            # stack_code: 0=3-3, 1=4-2, 2=5-1
                            if t == 3:
                                stack_code = 0
                            elif t == 2 or t == 4:
                                stack_code = 1
                            else:
                                stack_code = 2

                            proj_total = p_cijkl + float(proj_points[m])
                            salary_left = salary_cap - salary_used

                            cpt_out[write_idx] = c
                            u1_out[write_idx] = i
                            u2_out[write_idx] = j
                            u3_out[write_idx] = k
                            u4_out[write_idx] = l
                            u5_out[write_idx] = m
                            salary_used_out[write_idx] = salary_used
                            salary_left_out[write_idx] = salary_left
                            proj_points_out[write_idx] = np.float32(proj_total)
                            stack_code_out[write_idx] = np.uint8(stack_code)

                            write_idx += 1
                            written += 1

        written_per_cpt_out[c] = written


