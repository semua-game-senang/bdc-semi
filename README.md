# BDC semifinal clustering code

Canonical whole-image electronic-waste clustering pipeline: verified manifest, frozen CLIP/KEC embeddings, native-pixel Gabor/LBP regions, fixed 12-prompt text bank, selective C2 similarity, and average-linkage 12/24-cluster outputs.

## Install

```powershell
pip install -r research/experiments/canonical_pipeline/requirements.txt
pip install modal
```

## Run from retained feature bundles

Retrieve these private `semua-game-senang/bdc` dataset artifacts:

- Revision `7ca2bd9bf25599795c262f7b4d5198a16a9f1072`: `outputs/canonical-v1-selected-v1/canonical-selected-v1.zip` for the verified manifest, CLIP/KEC vectors, and native-region descriptors.
- Revision `27e1bbc53c62108b1b86786a30d856d9977a241b`: `outputs/canonical-prompt-selective-c2-v2/prompt-selective-c2-v2.zip` for the fixed CLIP text vectors and selected outputs.

Place `manifest.json` beside `clip.npy`, `kec.npy`, and `surfaces.npz` in a run directory, retaining their JSON sidecars. Then run from this repository root:

```powershell
python -m research.experiments.canonical_pipeline.prompt_selective_c2 `
  --root data/canonical-v1/run `
  --text-vectors data/canonical-c2-v2/text/text_vectors.npy `
  --output-dir data/replayed-c2 `
  --prompt-temperature 0.05 --prompt-weight 0.25
```

The command saves the full C2 similarity matrix, 12/24-cut cluster labels, candidate edges, references, and run summary. `modal_runner.py` contains the manifest, semantic, region, and text-bank stages for runs from organizer images in the existing Modal volumes. Evaluation and audit entrypoints are in the same package.
