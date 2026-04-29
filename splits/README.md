# splits/

PRISM split.pkl files, copied verbatim from the PRISM authors' working directory:

- `kits.pkl`   ← `lab:/media/hao/easystore/.../promise/datafile/kits/split.pkl`
- `colon.pkl`  ← `lab:/media/hao/easystore/.../promise/datafile/colon/split.pkl`

Both are 5-fold list-of-dict, where each dict has keys `{train, val, test}`. Values map case ids to their relative file paths.

## Reuse rule

Per `R-Method.PrismSplit`, these files are loaded verbatim. **Do not regenerate.** md5 sums are tracked in `SHA.txt` (also committed) so any divergence is detectable.

## How to populate

Run on the lab machine:
```
bash scripts/sync_lab_to_accre.sh
```

That script copies the split files into this directory and writes their md5s. Then `git add splits/`, `git commit`, `git push` — they are tiny (~100 KB total) and belong in version control.
