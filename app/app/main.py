import io
import json
import os
from typing import List, Optional, Tuple

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image, ImageDraw, ImageFont, ImageOps
import pytesseract
import cv2

from .nlp_rules import parse_instruction

app = FastAPI(title="Image Editing App Starter", version="0.2.0-auto-adapt")

FONTS_DIR = os.path.join(os.path.dirname(__file__), "fonts")

# --- Utils ---

def pil_to_cv(im: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGBA2BGR if im.mode == "RGBA" else cv2.COLOR_RGB2BGR)

def cv_to_pil(arr: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def list_candidate_fonts() -> List[str]:
    if not os.path.isdir(FONTS_DIR):
        return []
    files = [f for f in os.listdir(FONTS_DIR) if f.lower().endswith((".ttf", ".otf"))]
    return sorted(files)

def load_font(font_name: Optional[str], size: int) -> ImageFont.FreeTypeFont:
    # Essaye police demandée, sinon DejaVuSans fallback
    if font_name:
        try:
            return ImageFont.truetype(os.path.join(FONTS_DIR, font_name), size)
        except Exception:
            pass
    # Fallback system
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()

def hex_to_rgba(x: str, alpha: int = 255):
    x = x.lstrip('#')
    if len(x) == 6:
        r = int(x[0:2], 16); g = int(x[2:4], 16); b = int(x[4:6], 16)
        return (r, g, b, alpha)
    return (255,255,255,alpha)

def add_watermark(im: Image.Image, text: str = "Mockup") -> Image.Image:
    draw = ImageDraw.Draw(im)
    W, H = im.size
    font = load_font(None, max(18, W // 40))
    bbox = draw.textbbox((0,0), text, font=font)
    w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    x, y = W - w - 16, H - h - 16
    draw.rectangle([x-8, y-4, x+w+8, y+h+4], fill=(0,0,0,100))
    draw.text((x, y), text, font=font, fill=(255,255,255,180))
    return im

# --- OCR ---

def ocr_with_boxes(im: Image.Image):
    cv = pil_to_cv(im)
    gray = cv2.cvtColor(cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 5, 75, 75)
    data = pytesseract.image_to_data(gray, output_type=pytesseract.Output.DICT, lang='eng+fra')
    results = []
    for i in range(len(data['text'])):
        txt = (data['text'][i] or '').strip()
        if not txt: continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        if w*h <= 0: continue
        results.append((txt, (x, y, w, h)))
    return results

# --- Inpainting ---

def inpaint_rect(cv_img: np.ndarray, box: Tuple[int,int,int,int], inflate: int = 4) -> np.ndarray:
    x, y, w, h = box
    x1 = max(0, x - inflate); y1 = max(0, y - inflate)
    x2 = min(cv_img.shape[1]-1, x + w + inflate); y2 = min(cv_img.shape[0]-1, y + h + inflate)
    mask = np.zeros(cv_img.shape[:2], dtype=np.uint8); mask[y1:y2, x1:x2] = 255
    return cv2.inpaint(cv_img, mask, 3, cv2.INPAINT_TELEA)

# --- Auto style/size/color estimation ---

def estimate_text_color(im: Image.Image, box: Tuple[int,int,int,int]):
    x, y, w, h = box
    crop = im.crop((x, y, x+w, y+h)).convert('RGB')
    arr = np.asarray(crop).reshape(-1, 3)
    if arr.size == 0: return (255,255,255,255)
    lum = 0.2126*arr[:,0] + 0.7152*arr[:,1] + 0.0722*arr[:,2]
    k = max(1, int(0.2 * len(lum)))
    idx = np.argpartition(lum, k)[:k]
    mean_col = arr[idx].mean(axis=0)
    r,g,b = [int(x) for x in mean_col]
    return (r,g,b,255)

def measure_text_bbox(text: str, font: ImageFont.FreeTypeFont):
    dummy = Image.new('RGB', (1,1)); d = ImageDraw.Draw(dummy)
    bbox = d.textbbox((0,0), text, font=font)
    return (bbox[2]-bbox[0], bbox[3]-bbox[1])

def estimate_font_size_for_box(text: str, font_name: Optional[str], target_w: int, target_h: int) -> int:
    lo, hi = 6, max(12, target_h*3); best = lo
    for _ in range(16):
        mid = (lo + hi)//2
        font = load_font(font_name, mid)
        w,h = measure_text_bbox(text, font)
        if h <= target_h and w <= int(target_w*1.15):
            best = mid; lo = mid + 1
        else: hi = mid - 1
    return max(6, best)

def choose_best_font_by_fit(text: str, target_w: int, target_h: int, candidates: List[str]):
    best_name, best_size, best_err = None, 0, 1e9
    for name in candidates or [None]:
        size = estimate_font_size_for_box(text, name, target_w, target_h)
        font = load_font(name, size)
        w,h = measure_text_bbox(text, font)
        err = abs(h - target_h) + 0.6*abs(w - target_w)
        if err < best_err:
            best_err, best_name, best_size = err, name, size
    return (best_name, best_size)

# --- Schemas ---

class OverlayItem(BaseModel):
    text: str
    x: int
    y: int
    size: int = 32
    color: str = "#FFFFFF"
    font_name: Optional[str] = None

class OverlayPayload(BaseModel):
    items: List[OverlayItem]
    font_name: Optional[str] = None
    apply_watermark: bool = False
    watermark: Optional[str] = "Mockup"

# --- Endpoints ---

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/overlay")
async def overlay(image: UploadFile = File(...), fields: str = Form(...)):
    try:
        payload = OverlayPayload.model_validate(json.loads(fields))
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid fields JSON: {e}"})
    raw = await image.read()
    im_src = Image.open(io.BytesIO(raw))
    im = ImageOps.exif_transpose(im_src).convert("RGBA")
    out_format = (im_src.format or 'PNG')
    draw = ImageDraw.Draw(im)
    for it in payload.items:
        font = load_font(it.font_name or payload.font_name, it.size)
        draw.text((it.x, it.y), it.text, font=font, fill=hex_to_rgba(it.color))
    if payload.apply_watermark and payload.watermark:
        im = add_watermark(im, payload.watermark)
    buf = io.BytesIO(); im.save(buf, format=out_format); buf.seek(0)
    media = f"image/{out_format.lower()}" if out_format else "image/png"
    return StreamingResponse(buf, media_type=media)

@app.post("/process_nl")
async def process_nl(
    image: UploadFile = File(...),
    instruction: str = Form(...),
    font_name: Optional[str] = Form(default=None),
    text_size: int = Form(default=32),
    text_color: str = Form(default="#FFFFFF"),
    apply_watermark: bool = Form(default=False),
):
    raw = await image.read()
    im_src = Image.open(io.BytesIO(raw))
    im = ImageOps.exif_transpose(im_src).convert("RGBA")
    out_format = (im_src.format or 'PNG')
    boxes = ocr_with_boxes(im)
    actions = parse_instruction(instruction)
    cv_img = pil_to_cv(im)
    W, H = im.size
    positions = {"topleft": (20, 20), "topright": (W - 300, 20), "bottomleft": (20, H - 80), "bottomright": (W - 300, H - 80), "center": (W // 2 - 50, H // 2 - 10)}
    def find_box_for_text(query: str):
        norm_q = query.lower().replace(" ", "").replace(",", "").replace("'", "").replace("’", "")
        best = None
        for txt, box in boxes:
            norm_t = txt.lower().replace(" ", "").replace(",", "").replace("'", "").replace("’", "")
            if norm_q in norm_t or norm_t in norm_q:
                best = box; break
        return best
    for act in actions:
        if act["type"] == "replace_text":
            box = find_box_for_text(act["old"]) or find_box_for_text(act["old"].replace(" ", ""))
            if box is not None:
                cv_img = inpaint_rect(cv_img, box, inflate=6)
                x, y, w, h = box
                candidates = list_candidate_fonts()
                best_font, best_size = choose_best_font_by_fit(act["new"], w, h, candidates)
                if font_name:
                    best_font = font_name
                    best_size = estimate_font_size_for_box(act["new"], best_font, w, h)
                color_rgba = estimate_text_color(cv_to_pil(cv_img), box) if text_color == "auto" else hex_to_rgba(text_color)
                pil_after = cv_to_pil(cv_img).convert("RGBA")
                d = ImageDraw.Draw(pil_after)
                font = load_font(best_font, best_size)
                d.text((x, y), act["new"], font=font, fill=color_rgba)
                cv_img = pil_to_cv(pil_after)
        elif act["type"] == "place_text":
            pos_key = act.get("pos_key"); x, y = positions.get(pos_key, (20, 20))
            target_h = max(12, int(0.04 * H))
            candidates = list_candidate_fonts()
            best_font, best_size = choose_best_font_by_fit(act["text"], int(0.5*W), target_h, candidates)
            font = load_font(best_font, best_size)
            pil_after = cv_to_pil(cv_img).convert("RGBA")
            d = ImageDraw.Draw(pil_after)
            color_rgba = hex_to_rgba(text_color if text_color != "auto" else "#FFFFFF")
            d.text((x, y), act["text"], font=font, fill=color_rgba)
            cv_img = pil_to_cv(pil_after)
    out = cv_to_pil(cv_img).convert("RGBA")
    if apply_watermark: out = add_watermark(out, "Mockup")
    buf = io.BytesIO(); out.save(buf, format=out_format); buf.seek(0)
    media = f"image/{out_format.lower()}" if out_format else "image/png"
    return StreamingResponse(buf, media_type=media)

@app.post("/smart_replace")
async def smart_replace(
    image: UploadFile = File(...),
    old_text: str = Form(...),
    new_text: str = Form(...),
    force_font_name: Optional[str] = Form(default=None),
    color_mode: str = Form(default="auto"),
    apply_watermark: bool = Form(default=False),
):
    # Remplace un texte existant par un autre (auto police/taille/couleur + format conservé)
    raw = await image.read()
    im_src = Image.open(io.BytesIO(raw))
    im = ImageOps.exif_transpose(im_src).convert("RGBA")
    out_format = (im_src.format or 'PNG')
    boxes = ocr_with_boxes(im)
    def find_box_for_text(query: str):
        norm_q = query.lower().replace(" ", "").replace(",", "").replace("'", "").replace("’", "")
        for txt, box in boxes:
            norm_t = txt.lower().replace(" ", "").replace(",", "").replace("'", "").replace("’", "")
            if norm_q in norm_t or norm_t in norm_q:
                return box
        return None
    target_box = find_box_for_text(old_text)
    if target_box is None:
        return JSONResponse(status_code=404, content={"error": "Ancien texte introuvable via OCR"})
    x, y, w, h = target_box
    cv_img = pil_to_cv(im)
    cv_img = inpaint_rect(cv_img, target_box, inflate=6)
    candidates = list_candidate_fonts()
    best_font, best_size = choose_best_font_by_fit(new_text, w, h, candidates)
    if force_font_name:
        best_font = force_font_name
        best_size = estimate_font_size_for_box(new_text, best_font, w, h)
    color_rgba = estimate_text_color(im, target_box) if color_mode == "auto" else hex_to_rgba(color_mode)
    pil_after = cv_to_pil(cv_img).convert("RGBA")
    d = ImageDraw.Draw(pil_after)
    font = load_font(best_font, best_size)
    d.text((x, y), new_text, font=font, fill=color_rgba)
    if apply_watermark:
        pil_after = add_watermark(pil_after, "Mockup")
    buf = io.BytesIO(); pil_after.save(buf, format=out_format); buf.seek(0)
    media = f"image/{out_format.lower()}" if out_format else "image/png"
    return StreamingResponse(buf, media_type=media)
