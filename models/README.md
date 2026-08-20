# Models
## Final model: `yolov8n_neu_final.pt`

Trained on YOLOv8n, 50 epochs, `degrees=10` (gentler rotation — won hyperparameter
tuning), with generic augmentation + targeted oversampling for the two weakest
classes (crazing, then rolled-in_scale to rebalance).

**Validation results (270 images, 613 instances):**

| Class | mAP50 | mAP50-95 |
|---|---|---|
| **all (overall)** | **0.778** | 0.439 |
| crazing | 0.396 | 0.148 |
| inclusion | 0.861 | 0.424 |
| patches | 0.932 | 0.595 |
| pitted_surface | 0.870 | 0.535 |
| rolled-in_scale | 0.623 | 0.337 |
| scratches | 0.988 | 0.592 |

Inference speed: ~2.8ms/image on T4 GPU (~350+ FPS) — well within real-time
requirements for edge deployment.

## Training history (for reference)
1. Baseline (no augmentation): mAP50 = 0.763
2. + generic augmentation (25° rotation): mAP50 = 0.756 (crazing improved, rolled-in_scale dropped)
3. + hyperparameter tuning (10° rotation won) + 50 epochs: mAP50 = 0.769
4. + crazing oversampling (3x): mAP50 = 0.773 (crazing up, rolled-in_scale down)
5. + rolled-in_scale oversampling (2x, rebalance): **mAP50 = 0.778 (final)**