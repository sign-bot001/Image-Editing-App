import re
from typing import List, Dict, Any

RE_NUMBER = r"""[0-9][0-9 .,’',]*[0-9]*"""

POSITION_KEYWORDS = {
    "haut gauche": "topleft",
    "en haut à gauche": "topleft",
    "en haut a gauche": "topleft",
    "topleft": "topleft",
    "haut droite": "topright",
    "en haut à droite": "topright",
    "bottomleft": "bottomleft",
    "bottomright": "bottomright",
    "centre": "center",
    "center": "center",
}

def parse_instruction(text: str) -> List[Dict[str, Any]]:
    t = text.lower().strip()
    actions: List[Dict[str, Any]] = []

    # Remplacer X par Y
    m = re.findall(rf"(?:remplacer|replace|changer|change)\s+({RE_NUMBER}|\S+)\s+(?:par|en)\s+({RE_NUMBER}|\S+)", t)
    for old, new in m:
        actions.append({"type": "replace_text", "old": old.strip(), "new": new.strip()})

    # Mettre un texte "..." à une position
    m2 = re.findall(r'(?:(?:mettre|place|placer)\s+(?:la\s+)?(?:date|texte)?\s*"([^"]+)"|"([^"]+)")\s*(?:\s|,|;)*(?:en|à|a)?\s*([a-zàâéèêîôûç\s]+)?', t)
    for a, b, pos_raw in m2:
        content = a or b
        pos = POSITION_KEYWORDS.get((pos_raw or "").strip(), None)
        actions.append({"type": "place_text", "text": content, "pos_key": pos})

    return actions
