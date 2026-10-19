# Submission mechanics

## 1. Code

Work on branch `solution/repro-ml` in **your fork**. Commit as you go with
meaningful messages. The PR must contain:

- the completed implementation (`train.py`, `configs/`);
- completed `REPORT.md`;
- no video files, no venvs, no `mlruns/` (see `.gitignore`).

## 2. Video (≤ 5 minutes)

Show the 4 checks from `docs/GRADING.md` in order. Screen recording briefly naming 
each check is enough; production quality is not graded. 
Upload to any host accessible by link and verify the link access window before
submitting.

## 3. Pull Request

On GitHub: Pull Requests → New → source: your fork's
`solution/repro-ml`, target: the original repo's `main`. In the
description include:

- link to the video + its duration;
- the reference-metrics table copied from your REPORT.md;
- the verification commands you ran (`python train.py`,
  `python train.py model=lr`, ...).

The link to the open PR is your submission. Submit it via [course channel]
before the announced deadline.

## 4. After review

If the reviewer (author/coordinator) requests changes, push fixes to the
same branch. If something in the *task
itself* is broken or unclear, open an Issue in the original repository.
