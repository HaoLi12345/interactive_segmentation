#!/usr/bin/env bash
# Sync raw datasets and PRISM splits from the lab machine to ACCRE.
# Run this on the lab machine (`ssh lab`), not on ACCRE.
#
# Layout on the lab side (verified 2026-04-29):
#   /media/hao/easystore/subcortical_segmentation/2023_nov/from_lab_server/0SAM_data/kist_update/data/   (32 GB, KiTS21)
#   /media/hao/easystore/subcortical_segmentation/2023_nov/from_lab_server/0SAM_data/Task10_Colon/       (4 GB, MSD-Colon)
#   /media/hao/easystore/subcortical_segmentation/2023_nov/promise/datafile/{kits,colon}/split.pkl       (PRISM splits)
#
# Layout on ACCRE after sync:
#   /data/h_oguz_lab/lih30/interactive_seg_data/kits21/        (case_00000/...)
#   /data/h_oguz_lab/lih30/interactive_seg_data/msd_colon/     (imagesTr/, labelsTr/, split.pkl)
#   /home/lih30/interactive_seg/data/splits/{kits.pkl,colon.pkl}  (small, committed to git)

set -euo pipefail

LAB_BASE="/media/hao/easystore/subcortical_segmentation/2023_nov"
ACCRE_DATA="accre:/data/h_oguz_lab/lih30/interactive_seg_data"
ACCRE_REPO="accre:/home/lih30/interactive_seg"

echo "=== KiTS21 → ACCRE ==="
rsync -az --info=progress2 \
  "${LAB_BASE}/from_lab_server/0SAM_data/kist_update/data/" \
  "${ACCRE_DATA}/kits21/"

echo "=== MSD-Colon → ACCRE ==="
rsync -az --info=progress2 \
  "${LAB_BASE}/from_lab_server/0SAM_data/Task10_Colon/" \
  "${ACCRE_DATA}/msd_colon/"

echo "=== PRISM splits (committed, small) → repo ==="
# Copy locally first, then commit + push from the lab side.
LOCAL_REPO_SPLITS="/home/hao/hao/interactive_seg/splits"
mkdir -p "${LOCAL_REPO_SPLITS}"
cp -v "${LAB_BASE}/promise/datafile/kits/split.pkl"  "${LOCAL_REPO_SPLITS}/kits.pkl"
cp -v "${LAB_BASE}/promise/datafile/colon/split.pkl" "${LOCAL_REPO_SPLITS}/colon.pkl"

# md5 manifest for verification (R-Method.PrismSplit).
md5sum "${LAB_BASE}/promise/datafile/kits/split.pkl"  > "${LOCAL_REPO_SPLITS}/SHA.txt"
md5sum "${LAB_BASE}/promise/datafile/colon/split.pkl" >> "${LOCAL_REPO_SPLITS}/SHA.txt"

echo "=== Done ==="
ssh accre "du -sh /data/h_oguz_lab/lih30/interactive_seg_data/* 2>/dev/null"
