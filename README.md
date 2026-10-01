# Adversarial Robustness Evaluation & Defenses

EECS 5720 – Fundamentals of Cyber Security (Fall 2026), University of Toledo
Team: Uma Maram, Syam Sundar Nallamekala, Sushmitha Devi Saripalli, Hemasri Yarlagadda

## Goal
Evaluate how a pretrained image classifier breaks under adversarial attacks, apply defenses,
and benchmark every defense against the undefended baseline.

## Setup
| Item | Choice |
|---|---|
| Dataset | CIFAR-10 (test set for evaluation) |
| Baseline model | Pretrained CIFAR-10 ResNet-20 (`chenyaofo/pytorch-cifar-models`, ~92% clean acc) |
| Threat model | White-box, untargeted, L-inf; main budget eps = 8/255, sweep 2/4/8/16 /255 |
| Attacks | FGSM, PGD (10/20 steps), C&W-L2, AutoAttack |
| Defenses | 1) Adversarial fine-tuning (PGD / FGSM-RS) 2) Input preprocessing (JPEG, bit-depth) + adaptive BPDA attack |
| Metrics | Clean acc, robust acc per attack/eps, accuracy–robustness trade-off, training cost |
| Compute | Google Colab GPU (T4) |

## Roles (pick one)
| Role | Owns | Member |
|---|---|---|
| A – Baseline & Evaluation | Data pipeline, baseline model, `evaluate.py`, results table | |
| B – Attacks | FGSM, PGD, C&W, AutoAttack, eps sweep | |
| C – Adversarial Training | PGD / FGSM-RS fine-tuning of the baseline | |
| D – Preprocessing & Lit Review | JPEG / bit-depth defenses, BPDA adaptive attack, related-work lead | |

Everyone writes their own section of the report.

## Repo layout
```
src/
  data.py          # CIFAR-10 loaders (pixels in [0,1])
  models.py        # pretrained baseline + normalization wrapper
  attacks.py       # attack factory (torchattacks)
  evaluate.py      # clean + robust accuracy -> results/*.csv
  defenses/
    adv_train.py   # Role C
    preprocess.py  # Role D
notebooks/         # Colab notebooks
results/           # CSVs and plots (small files only)
report/            # IEEE report (Overleaf link / exported PDF)
```

## Quick start (Colab)
```bash
git clone <repo-url> && cd adversarial-robustness-eval
pip install -r requirements.txt
python -m src.evaluate --attack none          # clean accuracy
python -m src.evaluate --attack pgd --eps 8   # robust accuracy at 8/255
```

## Rules
- Work on a branch per role, open a PR into `main`.
- Do not commit datasets or model checkpoints (`.gitignore` covers them); share large files via OneDrive/Drive.
- Fix random seeds and log every run in `results/` so numbers in the report are reproducible.

## Milestones
- [ ] Week 7 – Midterm report (2-page IEEE): related work, threat model, methodology, baseline numbers
- [ ] Attacks complete on baseline
- [ ] Defenses trained and evaluated
- [ ] Final report
