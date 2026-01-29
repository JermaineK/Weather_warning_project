# BRANCH_STATUS_knee

Last updated: 2026-01-29

Status summary:
- Slowtick fixed: lead filtering applied; hashes differ per lead.
  - corr(dcov_24h, dcov_240h) ≈ -0.047 (expected, not ~1).
- Knee leads produced: results/metrics/coral_sea_demo_knee_leads.csv

Current knee-lead metrics (target: y_knee_cross_240h):
- horizon_240h (no feature shift): AUC=0.726, PRAUC=0.914
- lead 24h: AUC=0.684, PRAUC=0.516
- lead 48h: AUC=0.687, PRAUC=0.691
- lead 72h: AUC=0.697, PRAUC=0.783
- lead 120h: AUC=0.718, PRAUC=0.867
- lead 240h: AUC=0.726, PRAUC=0.914

Interpretation note:
- "coincident" for y_knee_cross_240h is a horizon target (future window), not at-event.
