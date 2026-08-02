"""How much of the source latent geometry is actually at risk from a per-frame loss.

``note.md`` argues that a cosine-only objective leaves pairwise distances free
within +/- 2*theta of the truth, where theta = arccos(c) is the cone half-angle at
the encoder's per-frame cosine. Whether that matters is an empirical question about
the *source* latents: it depends on how far apart the pairs actually are.

This script answers it from ``humanoidverse/data/g1_twist2_z`` (the frozen G1 BFM's
z* for all 40 LAFAN clips) and prints three things:

1. the separation distribution, split into pairs from the same clip and pairs from
   different behavior classes -- the second group is the one a tracker must keep apart;
2. the fraction of each group that falls under 2*theta, i.e. that the alignment loss
   would allow to be merged into a single point;
3. the floor each group can be compressed to, which is the claim that survives.

The headline finding is that (2) is essentially zero across behaviors and (3) is
severe, so the argument to make on the website is compression, not collapse.

Run with the cross_bfm environment:

  ~/Documents/Research/cross_bfm/.venv/bin/python tools/measure_pair_angles.py
"""

from __future__ import annotations

import argparse
import glob
import os
import pickle
import sys

import numpy as np

# Held-out per-frame cosine reached on each target robot; 2*theta is the width
# below which two source latents can be mapped to one point.
OPERATING_POINTS = [("M3", 0.9026), ("N1", 0.8870), ("T1", 0.8280)]

CROSS_STRIDE = 15  # 0.5 s at 30 fps -- keeps the full pair matrix tractable
WITHIN_STRIDE = 3  # 0.1 s; dense enough to include near-duplicate frames


def angles(zn: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(zn @ zn.T, -1.0, 1.0)))


def unit(z: np.ndarray) -> np.ndarray:
    return z / np.linalg.norm(z, axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.expanduser("~/Documents/Research/cross_bfm"))
    cli = ap.parse_args()
    repo = os.path.abspath(os.path.expanduser(cli.repo))
    sys.path.insert(0, repo)
    from humanoidverse.cross_embodiment.mjlab_bfm.tools.viz_latent_2d import categorize

    files = sorted(glob.glob(os.path.join(repo, "humanoidverse/data/g1_twist2_z", "*.pkl")))
    if not files:
        raise SystemExit("no z* clips found under humanoidverse/data/g1_twist2_z")

    # --- within-clip, sampled densely so adjacent frames are represented -------
    within, near_in_time = [], []
    for p in files:
        z = np.asarray(pickle.load(open(p, "rb"))["z"], dtype=np.float64)[::WITHIN_STRIDE]
        a = angles(unit(z))
        iu = np.triu_indices_from(a, k=1)
        within.append(a[iu])
        near_in_time.append(a[iu][np.abs(iu[0] - iu[1]) <= 10])  # within 1 s
    within = np.concatenate(within)
    near_in_time = np.concatenate(near_in_time)

    # --- across clips and behavior classes ------------------------------------
    Z, cat = [], []
    for p in files:
        z = np.asarray(pickle.load(open(p, "rb"))["z"], dtype=np.float64)[::CROSS_STRIDE]
        Z.append(z)
        cat += [categorize(os.path.basename(p)[:-4])] * len(z)
    Z, cat = np.concatenate(Z), np.array(cat)
    a = angles(unit(Z))
    iu = np.triu_indices_from(a, k=1)
    pairs = a[iu]
    cross = pairs[cat[iu[0]] != cat[iu[1]]]

    print(f"[pairs] {len(files)} clips | within-clip {within.size:,} pairs "
          f"| cross-behavior {cross.size:,} pairs")
    print(f"[within] median {np.median(within):.1f} deg, p5 {np.percentile(within, 5):.1f}, "
          f"p1 {np.percentile(within, 1):.1f}")
    print(f"[within, <=1 s apart] median {np.median(near_in_time):.1f} deg")
    print(f"[cross]  median {np.median(cross):.1f} deg, p5 {np.percentile(cross, 5):.1f}, "
          f"p1 {np.percentile(cross, 1):.1f}, min {cross.min():.1f}")
    print()

    # 51 deg is the scale of within-second variation inside one clip: a third of
    # those pairs sit below it. Compressing a cross-behavior pair under that line
    # makes two distinct behaviors as close as two neighbouring frames.
    ref = 51.0
    print(f"[reference] {100 * (near_in_time <= ref).mean():.0f}% of pairs less than a second "
          f"apart in one clip are already within {ref:.0f} deg")
    print()

    for name, c in OPERATING_POINTS:
        two = 2 * np.degrees(np.arccos(c))
        floor = np.maximum(0.0, cross - two)
        print(f"{name} (c={c}, 2*theta={two:.1f} deg)")
        print(f"    collapse to a point   : {100 * (within <= two).mean():.2f}% of within-clip pairs, "
              f"{100 * (cross <= two).mean():.2f}% of cross-behavior pairs")
        print(f"    median cross-behavior : {np.median(cross):.0f} deg -> floor {np.median(floor):.0f} deg "
              f"({100 * np.median(floor) / np.median(cross):.0f}% of true)")
        print(f"    can be pushed below {ref:.0f} deg: {100 * (floor < ref).mean():.1f}% "
              f"of cross-behavior pairs")


if __name__ == "__main__":
    main()
