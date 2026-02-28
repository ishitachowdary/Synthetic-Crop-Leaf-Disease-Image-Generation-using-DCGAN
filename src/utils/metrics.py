import torch
import numpy as np
import torch.nn.functional as F
from torchvision.models import inception_v3
from torchvision import transforms


@torch.no_grad()
def get_inception_model(device):
    model = inception_v3(pretrained=True, transform_input=False)
    model.fc = torch.nn.Identity()  # remove classifier
    model.eval().to(device)
    return model


@torch.no_grad()
def inception_score(images, device, splits=10):

    model = inception_v3(pretrained=True).to(device)
    model.eval()

    # Resize + normalize for Inception
    preprocess = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    preds = []

    for img in images:
        img = (img + 1) / 2  # [-1,1] → [0,1]
        img = preprocess(img).unsqueeze(0).to(device)
        p = F.softmax(model(img), dim=1)
        preds.append(p.cpu().numpy())

    preds = np.concatenate(preds, axis=0)

    scores = []
    N = preds.shape[0]
    split_size = N // splits

    for i in range(splits):
        part = preds[i * split_size:(i + 1) * split_size]
        py = np.mean(part, axis=0)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        scores.append(np.exp(np.mean(np.sum(kl, axis=1))))

    return float(np.mean(scores)), float(np.std(scores))
