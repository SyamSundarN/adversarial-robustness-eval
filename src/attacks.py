"""Attack factory. eps is given in /255 units (e.g. 8 -> 8/255)."""
import torchattacks


def get_attack(name, model, eps=8):
    e = eps / 255
    name = name.lower()
    if name == "fgsm":
        return torchattacks.FGSM(model, eps=e)
    if name == "pgd":
        return torchattacks.PGD(model, eps=e, alpha=e / 4, steps=10, random_start=True)
    if name == "pgd20":
        return torchattacks.PGD(model, eps=e, alpha=e / 4, steps=20, random_start=True)
    if name == "cw":
        return torchattacks.CW(model, c=1, kappa=0, steps=100, lr=0.01)  # L2, eps unused
    if name == "autoattack":
        return torchattacks.AutoAttack(model, norm="Linf", eps=e, version="standard", n_classes=10)
    raise ValueError(f"Unknown attack: {name}")
