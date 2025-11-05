# streamlit_app.py
import io, os, re
from typing import List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, ImageOps
import easyocr  # OCR sans tesseract

# ---------- Dossiers / Config ----------
FONTS_DIR = os.path.join("app", "fonts")
os.makedirs(FONTS_DIR, exist_ok=True)

st.set_page_config(page_title="Image Editing (Streamlit)", layout="centered")
st.title("🖼️ Modif d’images — Streamlit (OCR + inpainting)")

# ---------- Utils ----------
def pil_to_cv(im: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGBA2BGR if im.mode == "RGBA" else cv2.COLOR_RGB2BGR)

def cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

def load_font(font_name: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    if font_name:
        path = os.path.join(FONTS_DIR, font_name)
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
    # Fallback
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

def list_candidate_fonts() -> List[str]:
    if not os.path.isdir(FONTS_DIR):
        return []
    return sorted([f for f in os.listdir(FONTS_DIR) if f.lower().endswith((".ttf", ".otf"))])

def measure_text_bbox(text: str, font: ImageFont.FreeTypeFont) -> Tuple[int,int]:
    im = Image.new("RGB", (1,1))
    d = ImageDraw.Draw(im)
    x0,y0,x1,y1 = d.textbbox((0,0), text, font=font)
    return (x1-x0, y1-y0)

def estimate_font_size_for_box(text: str, font_name: Optional[str], target_w: int, target_h: int) -> int:
    lo, hi = 6, max(12, target_h*3)
    best = lo
    for _ in range(16):
        mid = (lo + hi)//2
        font = load_font(font_name, mid)
        w,h = measure_text_bbox(text, font)
        if h <= target_h and w <= int(target_w*1.15):
            best = mid; lo = mid + 1
        else:
            hi = mid - 1
    return max(6, best)

def choose_best_font_by_fit(text: str, target_w: int, target_h: int, candidates: List[str]) -> Tuple[Optional[str],int]:
    best_name, best_size, best_err = None, 0, 1e9
    for name in candidates or [None]:
        size = estimate_font_size_for_box(text, name, target_w, target_h)
        w,h = measure_text_bbox(text, load_font(name, size))
        err = abs(h - target_h) + 0.6*abs(w - target_w)
        if err < best_err:
            best_err, best_name, best_size = err, name, size
    return (best_name, best_size)

def hex_to_rgba(x: str, alpha: int = 255):
    x = x.lstrip("#")
    if len(x) == 6:
        return (int(x[:2],16), int(x[2:4],16), int(x[4:],16), alpha)
    return (255,255,255,alpha)

def estimate_text_color(im: Image.Image, box: Tuple[int,int,int,int]) -> Tuple[int,int,int,int]:
    x,y,w,h = box
    crop = im.crop((x,y,x+w,y+h)).convert("RGB")
    arr = np.asarray(crop).reshape(-1, 3)
    if arr.size == 0: return (255,255,255,255)
    lum = 0.2126*arr[:,0] + 0.7152*arr[:,1] + 0.0722*arr[:,2]
    k = max(1, int(0.2*len(lum)))
    idx = np.argpartition(lum, k)[:k]
    r,g,b = arr[idx].mean(axis=0).astype(int)
    return (int(r),int(g),int(b),255)

def inpaint_rect(cv_img: np.ndarray, box: Tuple[int,int,int,int], inflate: int = 4) -> np.ndarray:
    x,y,w,h = box
    x1 = max(0, x - inflate); y1 = max(0, y - inflate)
    x2 = min(cv_img.shape[1]-1, x + w + inflate); y2 = min(cv_img.shape[0]-1, y + h + inflate)
    mask = np.zeros(cv_img.shape[:2], dtype=np.uint8); mask[y1:y2, x1:x2] = 255
    return cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_TELEA)

# ---------- OCR (EasyOCR) ----------
@st.cache_resource(show_spinner=False)
def get_reader():
    # Français + Anglais
    return easyocr.Reader(["fr","en"], gpu=False)

def ocr_with_boxes_easyocr(im: Image.Image):
    """Retourne [(texte, (x,y,w,h))]"""
    cv = pil_to_cv(im)
    # result: list of [box, text, conf], box = 4 points
    result = get_reader().readtext(cv, detail=1)
    out = []
    for (pts, txt, conf) in result:
        if not txt: continue
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        x, y, w, h = int(min(xs)), int(min(ys)), int(max(xs)-min(xs)), int(max(ys)-min(ys))
        if w*h <= 0: continue
        out.append((txt.strip(), (x,y,w,h)))
    return out

def find_box_for_text(query: str, boxes) -> Optional[Tuple[int,int,int,int]]:
    norm_q = query.lower().replace(" ", "").replace(",", "").replace("'", "").replace("’", "")
    best = None
    # simple recherche "contient"
    for txt, box in boxes:
        norm_t = txt.lower().replace(" ", "").replace(",", "").replace("'", "").replace("’", "")
        if norm_q in norm_t or norm_t in norm_q:
            best = box; break
    return best

# ---------- Parsing d'instruction (FR minimal) ----------
RE_NUMBER = r"[0-9][0-9 .,’',]*[0-9]*"
POS_KEYS = {"haut gauche":"topleft","en haut à gauche":"topleft","en haut a gauche":"topleft",
            "topleft":"topleft","haut droite":"topright","en haut à droite":"topright",
            "bottomleft":"bottomleft","bottomright":"bottomright","centre":"center","center":"center"}

def parse_instruction(t: str):
    t = t.lower().strip()
    actions = []
    # remplacer X par Y
    m = re.findall(rf"(?:remplacer|replace|changer|change)\s+({RE_NUMBER}|\S+)\s+(?:par|en)\s+({RE_NUMBER}|\S+)", t)
    for old,new in m:
        actions.append({"type":"replace_text","old":old.strip(),"new":new.strip()})
    # mettre "..." en position
    m2 = re.findall(r'(?:(?:mettre|place|placer)\s+(?:la\s+)?(?:date|texte)?\s*"([^"]+)"|"([^"]+)")\s*(?:\s|,|;)*(?:en|à|a)?\s*([a-zàâéèêîôûç\s]+)?', t)
    for a,b,pos_raw in m2:
        content = a or b
        pos = POS_KEYS.get((pos_raw or "").strip(), None)
        actions.append({"type":"place_text","text":content,"pos_key":pos})
    return actions

# ---------- UI ----------
st.caption("Colle ce projet dans GitHub, puis déploie sur Streamlit Cloud. Polices optionnelles dans `app/fonts/`.")

uploaded = st.file_uploader("Choisir une image", type=["png","jpg","jpeg"])
instruction = st.text_area("Instruction (ex. « Remplacer 11,00 par 13 500 et mettre la date \"Mer 28 oct. 17:32\" en haut à gauche »)", height=120)

col1, col2 = st.columns(2)
with col1:
    color_mode = st.selectbox("Couleur du texte", ["auto", "blanc (#FFFFFF)", "noir (#000000)", "custom hex"])
with col2:
    custom_hex = st.text_input("Si custom hex → #RRGGBB", value="#FFFFFF")

apply_watermark = st.checkbox("Ajouter watermark « Mockup »", value=False)

if uploaded:
    im_src = Image.open(uploaded)
    im = ImageOps.exif_transpose(im_src).convert("RGBA")
    st.image(im, caption="Image d’entrée", use_container_width=True)

    if st.button("🚀 Exécuter"):
        boxes = ocr_with_boxes_easyocr(im)
        actions = parse_instruction(instruction)
        cv_img = pil_to_cv(im)

        W, H = im.size
        positions = {
            "topleft": (20, 20),
            "topright": (W - 300, 20),
            "bottomleft": (20, H - 80),
            "bottomright": (W - 300, H - 80),
            "center": (W // 2 - 50, H // 2 - 10),
        }

        for act in actions:
            if act["type"] == "replace_text":
                box = find_box_for_text(act["old"], boxes) or find_box_for_text(act["old"].replace(" ", ""), boxes)
                if box is not None:
                    # efface
                    cv_img = inpaint_rect(cv_img, box, inflate=6)
                    x,y,w,h = box
                    # police/size automatiques
                    candidates = list_candidate_fonts()
                    best_font, best_size = choose_best_font_by_fit(act["new"], w, h, candidates)
                    pil_after = cv_to_pil(cv_img).convert("RGBA")
                    d = ImageDraw.Draw(pil_after)
                    # couleur
                    if color_mode == "auto":
                        fill = estimate_text_color(pil_after, box)
                    elif color_mode.startswith("blanc"):
                        fill = hex_to_rgba("#FFFFFF")
                    elif color_mode.startswith("noir"):
                        fill = hex_to_rgba("#000000")
                    else:
                        fill = hex_to_rgba(custom_hex or "#FFFFFF")
                    d.text((x, y), act["new"], font=load_font(best_font, best_size), fill=fill)
                    cv_img = pil_to_cv(pil_after)
                    # rafraîchir OCR si nécessaire
                    boxes = ocr_with_boxes_easyocr(pil_after)

            elif act["type"] == "place_text":
                pos = positions.get(act.get("pos_key"), (20,20))
                target_h = max(12, int(0.04 * H))
                candidates = list_candidate_fonts()
                best_font, best_size = choose_best_font_by_fit(act["text"], int(0.5*W), target_h, candidates)
                pil_after = cv_to_pil(cv_img).convert("RGBA")
                d = ImageDraw.Draw(pil_after)
                if color_mode == "auto":
                    fill = hex_to_rgba("#FFFFFF")
                elif color_mode.startswith("blanc"):
                    fill = hex_to_rgba("#FFFFFF")
                elif color_mode.startswith("noir"):
                    fill = hex_to_rgba("#000000")
                else:
                    fill = hex_to_rgba(custom_hex or "#FFFFFF")
                d.text(pos, act["text"], font=load_font(best_font, best_size), fill=fill)
                cv_img = pil_to_cv(pil_after)

        out = cv_to_pil(cv_img).convert("RGBA")

        if apply_watermark:
            d = ImageDraw.Draw(out)
            font = load_font(None, max(18, out.size[0]//40))
            x0,y0,x1,y1 = d.textbbox((0,0), "Mockup", font=font)
            w,h = x1-x0, y1-y0
            x,y = out.size[0]-w-16, out.size[1]-h-16
            d.rectangle([x-8,y-4,x+w+8,y+h+4], fill=(0,0,0,100))
            d.text((x,y), "Mockup", font=font, fill=(255,255,255,180))

        st.success("✅ Terminé")
        st.image(out, caption="Résultat", use_container_width=True)
        buf = io.BytesIO()
        fmt = (im_src.format or "PNG")
        out.save(buf, format=fmt)
        st.download_button("⬇️ Télécharger l’image", data=buf.getvalue(),
                           file_name=f"result.{fmt.lower()}", mime=f"image/{fmt.lower()}")
else:
    st.info("➡️ Dépose une image pour commencer.")
