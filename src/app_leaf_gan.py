import streamlit as st
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import pandas as pd
import math
from utils.logger import Logger

from generator import Generator
from utils.config import Config
from utils.classifier_inference import load_classifier, predict_image
from utils.monitoring import log_inference_latency, log_usage
import time
import io
import zipfile
import tempfile


st.set_page_config(
    page_title="Crop Leaf GAN",
    page_icon="🌱",
    layout="wide"
)

@st.cache_resource
def load_models():
    config = Config(
        "configs/data_config.yaml",
        "configs/train_config.yaml"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(
        latent_dim=config.train["dcgan"]["latent_dim"]
    ).to(device)

    generator.load_state_dict(
        torch.load(
            config.checkpoints_dir / "G_epoch_150.pth",
            map_location=device
        )
    )
    generator.eval()

    dataset = ImageFolder(config.data_dir / "Train")
    class_names = dataset.classes

    classifier = load_classifier(
        num_classes=len(class_names),
        device=device,
        checkpoint_path=config.checkpoints_dir / "classifier_augmented.pth"
    )
    logger = Logger(log_dir="logs")
    return generator, classifier, class_names, device, config, logger


generator, classifier, class_names, device, config, logger = load_models()
to_pil = transforms.ToPILImage()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
with st.sidebar:
    st.header("Model Summary")

    st.metric("Baseline Accuracy", "62.98%")
    st.metric("Augmented Accuracy", "78.24%")
    st.metric("Inception Score", "3.065 ± 0.223")

    st.markdown("---")
    st.markdown(
        """
**Pipeline**
- DCGAN → Synthetic images  
- Classifier → Semantic interpretation  
- Goal → Mitigate data scarcity
"""
    )

# -------------------------------------------------
# Main UI
# -------------------------------------------------
st.title("🌱 Crop Leaf Disease Image Generator")
st.markdown(
    """
Generate **synthetic crop leaf disease images** using a trained **DCGAN**,  
and **interpret them automatically** using a disease classifier.
"""
)

num_images = st.slider(
    "Number of images to generate",
    min_value=1,
    max_value=200,
    value=12
)

top_k = st.selectbox(
    "Number of predictions per image",
    options=[1, 3],
    index=1
)
start_time = time.time()

generate_btn = st.button("Generate Images")

# -------------------------------------------------
# Image generation + interpretation
# -------------------------------------------------
if generate_btn:
    st.subheader("Generated Images & Model Interpretation")

    n_cols = 8
    rows = math.ceil(num_images / n_cols)
    generated_images = [] 
    predictions = []

    for r in range(rows):
        cols = st.columns(n_cols)

        for c in range(n_cols):
            idx = r * n_cols + c
            if idx >= num_images:
                break

            with torch.no_grad():
                z = torch.randn(
                    1,
                    config.train["dcgan"]["latent_dim"],
                    1,
                    1,
                    device=device
                )

                img = generator(z)[0]
                img = (img + 1) / 2  # [-1,1] → [0,1]

                preds = predict_image(
                    img, classifier, class_names, device, top_k=top_k
                )

                predictions.append(preds[0][0])  # top-1 for stats

                caption = "\n".join(
                    [f"{cls}: {p*100:.1f}%" for cls, p in preds]
                )

                img_pil = to_pil(img.cpu())

                filename = f"synthetic_{idx:03d}.png"
                generated_images.append((img_pil, filename))

                cols[c].image(
                    img_pil,
                    caption=caption,
                    width="stretch"
                )

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for img, name in generated_images:
            img_bytes = io.BytesIO()
            img.save(img_bytes, format="PNG")
            zipf.writestr(name, img_bytes.getvalue())

    zip_buffer.seek(0)

    st.download_button(
        label="📥 Download Generated Images (ZIP)",
        data=zip_buffer,
        file_name="synthetic_leaf_images.zip",
        mime="application/zip"
    )
    # -------------------------------------------------
    # Analytics
    # -------------------------------------------------
    st.subheader("Generated Class Distribution")

    df = pd.DataFrame(predictions, columns=["Predicted Class"])
    counts = df["Predicted Class"].value_counts()
    most_common_class = counts.idxmax()

    logger.log_inference(
        num_images=num_images,
        top_class=most_common_class
    )
    log_inference_latency(start_time, num_images)
    log_usage(
        source="streamlit",
        num_images=num_images,
        top_class=most_common_class
    )

    st.bar_chart(counts)

    st.info(
        "💡 This distribution shows how a trained classifier interprets GAN-generated images. "
        "It helps identify diversity, dominance of certain diseases, and possible GAN bias."
    )


    st.subheader("What Did the GAN Learn?")

    st.markdown(
        """
- The GAN successfully learns **leaf shape and texture**
- Disease patterns emerge as **classifier-recognizable features**
- Some images show **hybrid or ambiguous traits**, reflected by lower confidence
- This behavior is expected in **unconditional GANs**
"""
    )

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption(
    "DCGAN-based Synthetic Crop Disease Image Generator | "
    "Data Scarcity Mitigation Project"
)
