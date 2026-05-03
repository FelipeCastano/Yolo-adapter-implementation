import io
import base64
import logging
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

logging.basicConfig(level=logging.INFO)

API_URL = "http://api:8000"

# ── Helpers ───────────────────────────────────────────────────────────────────

def image_to_base64(img: Image.Image) -> str:
	"""Convert a PIL Image to a Base64-encoded JPEG string.

	:param img: Input image.
	:type img: PIL.Image.Image
	:return: Base64-encoded JPEG string.
	:rtype: str
	"""
	buffer = io.BytesIO()
	img.save(buffer, format="JPEG")
	return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(b64: str) -> Image.Image:
	"""Decode a Base64 string to a PIL Image.

	:param b64: Base64-encoded image string.
	:type b64: str
	:return: Decoded PIL Image.
	:rtype: PIL.Image.Image
	"""
	return Image.open(io.BytesIO(base64.b64decode(b64)))


def draw_boxes(img: Image.Image, detections: list) -> Image.Image:
    """Draw bounding boxes and labels on a PIL Image.

    :param img: Input image.
    :type img: PIL.Image.Image
    :param detections: List of detection dicts with ``bbox``, ``class_name``
        and ``confidence`` keys.
    :type detections: list
    :return: Annotated image.
    :rtype: PIL.Image.Image
    """
    annotated = img.copy()
    draw      = ImageDraw.Draw(annotated)

    # Escalar texto y boxes proporcionalmente al tamaño de la imagen
    img_w, img_h   = img.size
    scale          = max(img_w, img_h) / 800
    font_size      = max(12, int(16 * scale))
    box_thickness  = max(2, int(3 * scale))
    label_height   = font_size + 6
    char_width     = font_size * 0.65

    colors = {
        "Capacitor":       "#3B8BD4",
        "Inductor":        "#1D9E75",
        "Resistor":        "#D85A30",
        "DC_VS":           "#7F77DD",
        "AC_VS":           "#D4537E",
        "Connection_Node": "#EF9F27",
        "Gnd":             "#888780",
    }

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            size=font_size,
        )
    except OSError:
        font = ImageFont.load_default(size=font_size)

    for det in detections:
        bbox            = det["bbox"]
        label           = f"{det['class_name']} {det['confidence']:.2f}"
        color           = colors.get(det["class_name"], "#E24B4A")
        x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_thickness)

        text_w = int(len(label) * char_width)
        draw.rectangle([x1, y1 - label_height, x1 + text_w, y1], fill=color)
        draw.text((x1 + 2, y1 - label_height + 2), label, fill="white", font=font)

    return annotated
    
# ── App ───────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Blood Cell Detector", layout="wide")
st.title("Blood Cell Detector")

tab_predict, tab_augment, tab_docs = st.tabs(["Detection", "Augmentation", "Documentation"])


# ── Tab 1: Detection ──────────────────────────────────────────────────────────

with tab_predict:
	st.header("Blood Cell Detection")
	st.write("Upload a circuit schematic to detect blood cells.")
	st.markdown(
		"To explore training experiments visit "
		"[MLflow](http://127.0.0.1:5000)",
		unsafe_allow_html=False,
	)

	uploaded = st.file_uploader(
		"Upload image", type=["jpg", "jpeg", "png"], key="predict_upload"
	)
	detect_btn = st.button("Detect blood cells")  # ← movido aquí

	if uploaded:
		img      = Image.open(uploaded).convert("RGB")
		col1, col2 = st.columns(2)

		with col1:
			st.subheader("Original")
			st.image(img, use_container_width=True)

		if detect_btn:
			with st.spinner("Running detection..."):
				try:
					b64      = image_to_base64(img)
					response = requests.post(
						f"{API_URL}/predict",
						json    = {"image": b64},
						timeout = 30,
					)
					response.raise_for_status()
					detections = response.json().get("detections", [])

					with col2:
						st.subheader("Detections")
						annotated = draw_boxes(img, detections)
						st.image(annotated, use_container_width=True)

				except requests.RequestException as e:
					st.error(f"API error: {e}")


with tab_augment:
	st.header("Data Augmentation")
	st.write("Upload an image to generate augmented variants.")

	uploaded_aug = st.file_uploader(
		"Upload image", type=["jpg", "jpeg", "png"], key="augment_upload"
	)
	n_variants  = st.slider("Number of variants", min_value=1, max_value=10, value=5)
	augment_btn = st.button("Generate augmentations")  # ← movido aquí

	if uploaded_aug:
		img_aug = Image.open(uploaded_aug).convert("RGB")
		st.subheader("Original")
		st.image(img_aug, width=300)

		if augment_btn:
			with st.spinner("Generating variants..."):
				try:
					b64      = image_to_base64(img_aug)
					response = requests.post(
						f"{API_URL}/augment",
						json    = {"image": b64, "n_variants": n_variants},
						timeout = 30,
					)
					response.raise_for_status()
					variants = response.json().get("variants", [])

					st.subheader(f"Generated {len(variants)} variant(s)")
					cols = st.columns(min(len(variants), 5))
					for i, b64_variant in enumerate(variants):
						with cols[i % 5]:
							variant_img = base64_to_image(b64_variant)
							st.image(variant_img, caption=f"Variant {i + 1}",
									 use_container_width=True)

				except requests.RequestException as e:
					st.error(f"API error: {e}")

# ── Tab 3: Documentation ──────────────────────────────────────────────────────

with tab_docs:
	st.header("Documentation")
	st.components.v1.iframe("http://localhost:8889", height=800, scrolling=True)