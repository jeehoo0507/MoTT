"""Code-path test for collect_results.pooled_mod_table's P1 extension section (seed 0+1+2, held-out ip/pkg).
Uses FAKE records (copies of existing runs renamed), so the printed numbers mean nothing; only checks that the table
renders in the three states (extension runs missing / partly present / all present) without crashing."""
import copy
import sys

sys.path.insert(0, "scripts")
import collect_results as cr  # noqa: E402

recs = cr.load_evals()


def fake(src, dst):
    out = []
    for r in recs:
        if r["method"] == src:
            c = copy.deepcopy(r)
            c["method"] = dst
            out.append(c)
    assert out, src
    return out


base = recs + fake("lo_pcb_r2pk_rl", "lo_pcb_r2pk_s1_rl")
stages = [
    [],
    fake("r2pk_s1_rl", "r2pk_s2_rl"),
    fake("r2pk_s1_rl", "r2pk_s2_rl") + fake("lo_pcb_r2pk_rl", "lo_pcb_r2pk_s2_rl")
    + fake("lo_ip_r2_rl", "lo_ip_r2pk_rl") + fake("lo_pkg_r2_rl", "lo_pkg_r2pk_rl"),
]
for extra in stages:
    L = cr.pooled_mod_table(base + extra)
    i = next((k for k, x in enumerate(L) if x.startswith("### P1")), None)
    print("verdict line:", next(x for x in L if x.startswith("판정")).split(".")[0][:40], "| ext section:", i is not None)
    if i is not None:
        print("\n".join(L[i:i + 16]))
print("OK")
