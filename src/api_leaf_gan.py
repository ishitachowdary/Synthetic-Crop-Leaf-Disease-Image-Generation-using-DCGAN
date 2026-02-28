from time import time
import torch
from fastapi import FastAPI
from torchvision import transforms
from PIL import Image
import io
import base64

from src.utils.monitoring import log_inference_latency, log_usage
from src.generator import Generator
from src.utils.config import Config

app = FastAPI(title="Leaf GAN API")

config = Config(
    "configs/data_config.yaml",
    "configs/train_config.yaml"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
start_time = time.time()
generator = Generator(
    latent_dim=config.train["dcgan"]["latent_dim"]
).to(device)

generator.load_state_dict(
    torch.load(config.checkpoints_dir / "G_epoch_150.pth", map_location=device)
)
generator.eval()

to_pil = transforms.ToPILImage()

@app.get("/generate")
def generate_image():
    with torch.no_grad():
        z = torch.randn(
            1, config.train["dcgan"]["latent_dim"], 1, 1, device=device
        )
        img = generator(z)[0]
        img = (img + 1) / 2
        img_pil = to_pil(img.cpu())

        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode()

    return {"image_base64": encoded}
log_inference_latency(start_time, num_images=1)
log_usage("api", 1, "unknown")
