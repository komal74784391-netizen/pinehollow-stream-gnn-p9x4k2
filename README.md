# Stream Network GNN

Official implementation for **Graph Neural Priors for Stream Network
Delineation** (Proceedings of the 28th Conference on Environmental Machine
Learning).

Release: **v2.1.0** (2026-01-27). Contains the graph construction pipeline, the
message-passing model and the tile-level evaluation scripts.

```bash
pip install -r requirements.txt
python scripts/train.py --config configs/cemel.yaml
python scripts/eval_iou.py --checkpoint runs/best.ckpt
```
