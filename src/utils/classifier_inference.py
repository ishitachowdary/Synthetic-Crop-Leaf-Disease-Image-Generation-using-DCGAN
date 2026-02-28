import torch
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


# -------------------------------------------------
# Load classifier 
# -------------------------------------------------
def load_classifier(num_classes, device, checkpoint_path):
    model = resnet18(weights=None)  
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device)
    )
    model.to(device)
    model.eval()
    return model


# -------------------------------------------------
# Predict image 
# -------------------------------------------------
def predict_image(image_tensor, model, class_names, device, top_k=1):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])

    img = transform(image_tensor).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_idxs = torch.topk(probs, k=top_k)

    results = [
        (class_names[idx.item()], prob.item())
        for idx, prob in zip(top_idxs, top_probs)
    ]

    return results
