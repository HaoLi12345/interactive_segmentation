#!/usr/bin/env bash
# Sync raw datasets and PRISM splits from the lab machine to ACCRE.
# The lab machine is the data source only; this project does not check out on lab.
# Run this script via ssh from anywhere that has the `lab` and `accre` aliases:
#   ssh lab 'bash -s' < scripts/sync_lab_to_accre.sh
# or run on the lab machine directly.
#
# Layout on the lab side (verified 2026-04-29):
#   /media/hao/easystore/subcortical_segmentation/2023_nov/from_lab_server/0SAM_data/kist_update/data/   (32 GB, KiTS21)
#   /media/hao/easystore/subcortical_segmentation/2023_nov/from_lab_server/0SAM_data/Task10_Colon/       (4 GB, MSD-Colon)
#   /media/hao/easystore/subcortical_segmentation/2023_nov/promise/datafile/{kits,colon}/split.pkl       (PRISM splits)
#
# Layout on ACCRE after sync:
#   /data/h_oguz_lab/lih30/interactive_seg_data/kits21/        (case_<NNNNN>/...)
#   /data/h_oguz_lab/lih30/interactive_seg_data/msd_colon/     (imagesTr/, labelsTr/, split.pkl)
#   /home/lih30/interactive_seg/splits/{kits.pkl, colon.pkl, SHA.txt}  (small, will be committed from ACCRE)

set -euo pipefail

LAB_BASE="/media/hao/easystore/subcortical_segmentation/2023_nov"
ACCRE_DATA="accre:/data/h_oguz_lab/lih30/interactive_seg_data"
ACCRE_REPO="accre:/home/lih30/interactive_seg"

echo "=== KiTS21 -> ACCRE ==="
rsync -az --progress \
  "${LAB_BASE}/from_lab_server/0SAM_data/kist_update/data/" \
  "${ACCRE_DATA}/kits21/"

echo "=== MSD-Colon -> ACCRE ==="
rsync -az --progress \
  "${LAB_BASE}/from_lab_server/0SAM_data/Task10_Colon/" \
  "${ACCRE_DATA}/msd_colon/"

echo "=== PRISM splits + md5 manifest -> ACCRE repo splits/ ==="
TMPDIR_LOCAL=$(mktemp -d)
trap "rm -rf '${TMPDIR_LOCAL}'" EXIT
cp -v "${LAB_BASE}/promise/datafile/kits/split.pkl"  "${TMPDIR_LOCAL}/kits.pkl"
cp -v "${LAB_BASE}/promise/datafile/colon/split.pkl" "${TMPDIR_LOCAL}/colon.pkl"
md5sum "${TMPDIR_LOCAL}/kits.pkl"  | sed "s|${TMPDIR_LOCAL}/|splits/|" >  "${TMPDIR_LOCAL}/SHA.txt"
md5sum "${TMPDIR_LOCAL}/colon.pkl" | sed "s|${TMPDIR_LOCAL}/|splits/|" >> "${TMPDIR_LOCAL}/SHA.txt"
rsync -av "${TMPDIR_LOCAL}/" "${ACCRE_REPO}/splits/"

echo "=== Done ==="
ssh accre "du -sh /data/h_oguz_lab/lih30/interactive_seg_data/* 2>/dev/null && ls -l /home/lih30/interactive_seg/splits/"
