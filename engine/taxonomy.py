"""Bi-cultural (Indian + Western + Fusion), climate-aware zero-shot anchor bank.

Each label is embedded with several prompt templates (averaged + re-normalised) under the
classification task prefix, and stored in data/taxonomy_anchors.bin (float32, N x 256) with a
JSON sidecar describing rows. Images are scored by dot product against the bank.
"""
import hashlib
import json
from dataclasses import dataclass, field

import numpy as np

from engine import config

VIBES = ["Work", "Casual", "Festive", "Travel"]
SEASONS = ["Summer/Humid", "Monsoon", "Mild/Winter"]

W, C, F, T = "Work", "Casual", "Festive", "Travel"
HOT, WET, MILD = "Summer/Humid", "Monsoon", "Mild/Winter"
ALL = (HOT, WET, MILD)


@dataclass(frozen=True)
class Garment:
    label: str
    style: str                      # 'indian' | 'western' | 'fusion'
    vibes: tuple[str, ...]
    seasons: tuple[str, ...] = ALL
    aliases: tuple[str, ...] = field(default_factory=tuple)


# --------------------------------------------------------------- categories (whole looks)
CATEGORIES: list[Garment] = [
    # Indian ethnic - menswear
    Garment("kurta pajama", "indian", (F, C), ALL, ("kurta with pyjama",)),
    Garment("cotton kurta", "indian", (C, W), (HOT, WET)),
    Garment("linen kurta", "indian", (W, C), (HOT,)),
    Garment("silk kurta", "indian", (F,), (MILD, WET)),
    Garment("short kurta with jeans", "fusion", (C, T), (HOT, WET)),
    Garment("nehru jacket over kurta", "indian", (F, W), (MILD, WET)),
    Garment("bandhgala suit", "indian", (F, W), (MILD,), ("jodhpuri suit",)),
    Garment("sherwani", "indian", (F,), (MILD,)),
    Garment("indo-western sherwani", "fusion", (F,), (MILD,)),
    Garment("silk dhoti with kurta", "indian", (F,), (HOT, MILD), ("veshti", "panche")),
    Garment("mundu", "indian", (F, C), (HOT,), ("kerala mundu",)),
    Garment("pathani suit", "indian", (C, F), ALL),
    Garment("achkan", "indian", (F,), (MILD,)),
    Garment("lungi", "indian", (C,), (HOT, WET)),
    # Indian ethnic - womenswear
    Garment("silk saree", "indian", (F,), (MILD, HOT), ("kanjeevaram saree", "banarasi saree")),
    Garment("cotton saree", "indian", (W, C), (HOT, WET), ("handloom saree",)),
    Garment("chiffon saree", "indian", (F, C), (HOT, WET), ("georgette saree",)),
    Garment("linen saree", "indian", (W,), (HOT,)),
    Garment("pre-draped saree", "fusion", (F,), ALL),
    Garment("salwar kameez", "indian", (C, W), ALL, ("salwar suit",)),
    Garment("anarkali suit", "indian", (F,), (MILD, HOT)),
    Garment("churidar kurta", "indian", (W, C), ALL),
    Garment("kurti with leggings", "indian", (W, C, T), (HOT, WET)),
    Garment("straight kurta with palazzo", "fusion", (W, C), (HOT, WET)),
    Garment("sharara set", "indian", (F,), (MILD,)),
    Garment("lehenga choli", "indian", (F,), (MILD,), ("ghagra choli",)),
    Garment("half saree", "indian", (F,), ALL, ("langa voni",)),
    Garment("patiala suit", "indian", (C, F), (MILD,)),
    Garment("kaftan", "fusion", (C, T), (HOT,)),
    Garment("indo-western gown", "fusion", (F,), (MILD,)),
    Garment("dhoti pants with crop top", "fusion", (F, C), (HOT,)),
    Garment("kurta with denim jacket", "fusion", (C, T), (MILD, WET)),
    Garment("angrakha kurta", "indian", (F, C), (HOT, MILD)),
    Garment("kurta churidar for men", "indian", (F, W), ALL),
    Garment("palazzo suit with dupatta", "indian", (C, W), (HOT, WET)),
    Garment("mekhela chador", "indian", (F,), ALL),
    Garment("kasavu saree", "indian", (F,), (HOT, MILD), ("kerala set-mundu",)),
    Garment("paithani saree", "indian", (F,), (MILD,)),
    Garment("ikat handloom saree", "indian", (W, F), (HOT, WET)),
    Garment("saree with belt and blazer", "fusion", (W, F), (MILD,)),
    Garment("kurta with formal trousers", "fusion", (W,), (HOT, WET)),
    Garment("nehru jacket with shirt and trousers", "fusion", (W, F), (MILD, WET)),
    Garment("dhoti salwar with short kurti", "fusion", (C, F), (HOT,)),
    Garment("ethnic skirt with crop top", "fusion", (F, C), (HOT, MILD)),
    Garment("cape-style indo-western outfit", "fusion", (F,), (MILD,)),
    # Western - work
    Garment("formal shirt with trousers", "western", (W,), ALL),
    Garment("shirt and chinos", "western", (W, C), ALL, ("smart casual shirt and chinos",)),
    Garment("two-piece business suit", "western", (W, F), (MILD,)),
    Garment("blazer with trousers", "western", (W,), (MILD, WET)),
    Garment("blazer with jeans", "western", (W, C), (MILD,)),
    Garment("polo t-shirt with chinos", "western", (W, C, T), (HOT, WET)),
    Garment("sheath dress", "western", (W,), ALL),
    Garment("blouse with pencil skirt", "western", (W,), ALL),
    Garment("pantsuit", "western", (W,), (MILD,)),
    Garment("formal waistcoat with shirt", "western", (W, F), (MILD,)),
    # Western - casual
    Garment("t-shirt and jeans", "western", (C, T), ALL),
    Garment("graphic tee with shorts", "western", (C, T), (HOT,)),
    Garment("linen shirt with shorts", "western", (C, T), (HOT,)),
    Garment("henley with jeans", "western", (C,), (MILD,)),
    Garment("hoodie and joggers", "western", (C, T), (MILD,)),
    Garment("sweater with jeans", "western", (C,), (MILD,)),
    Garment("denim jacket with t-shirt", "western", (C, T), (MILD,)),
    Garment("leather jacket outfit", "western", (C, F), (MILD,)),
    Garment("summer sundress", "western", (C, T), (HOT,)),
    Garment("maxi dress", "western", (C, F, T), (HOT,)),
    Garment("crop top with jeans", "western", (C,), (HOT,)),
    Garment("jumpsuit", "western", (C, W), ALL),
    Garment("co-ord set", "western", (C, T, W), (HOT, WET)),
    Garment("tracksuit", "western", (C, T), (MILD,)),
    Garment("gym activewear", "western", (C,), ALL),
    Garment("rain jacket outfit", "western", (T, C), (WET,), ("windcheater",)),
    Garment("cargo pants with t-shirt", "western", (C, T), ALL),
    Garment("oxford shirt with jeans", "western", (C, W), ALL),
    Garment("knit sweater with chinos", "western", (W, C), (MILD,)),
    Garment("trench coat outfit", "western", (W, T), (MILD, WET)),
    Garment("puffer jacket outfit", "western", (T, C), (MILD,)),
    Garment("shirt dress", "western", (W, C), (HOT, WET)),
    Garment("wrap dress", "western", (W, C), ALL),
    Garment("a-line skirt with top", "western", (C, W), (HOT, WET)),
    Garment("linen trousers with tank top", "western", (C, T), (HOT,)),
    Garment("bermuda shorts with polo", "western", (C, T), (HOT,)),
    Garment("turtleneck with trousers", "western", (W,), (MILD,)),
    Garment("cardigan over a dress", "western", (C, W), (MILD,)),
    Garment("denim overalls", "western", (C,), (MILD, HOT)),
    Garment("athleisure joggers with tee", "western", (C, T), ALL),
    Garment("beach shorts and resort shirt", "western", (C, T), (HOT,)),
    # Western - festive / party
    Garment("tuxedo", "western", (F,), (MILD,)),
    Garment("cocktail dress", "western", (F,), ALL),
    Garment("evening gown", "western", (F,), (MILD,)),
    Garment("party shirt with trousers", "western", (F, C), ALL),
    Garment("velvet blazer outfit", "western", (F,), (MILD,)),
    Garment("sequin party dress", "western", (F,), (MILD,)),
]

FABRICS: list[Garment] = [
    Garment("linen", "any", (W, C), (HOT,)),
    Garment("cotton", "any", (W, C, T), (HOT, WET)),
    Garment("khadi", "any", (W, C), (HOT,)),
    Garment("silk", "any", (F,), (MILD,)),
    Garment("raw silk", "any", (F,), (MILD,)),
    Garment("chiffon", "any", (F, C), (HOT, WET)),
    Garment("georgette", "any", (F, C), (WET, HOT)),
    Garment("denim", "any", (C, T), (MILD, WET)),
    Garment("wool", "any", (W, F), (MILD,)),
    Garment("tweed", "any", (W,), (MILD,)),
    Garment("velvet", "any", (F,), (MILD,)),
    Garment("polyester", "any", (W, T), (WET,)),
    Garment("rayon", "any", (C,), (HOT, WET)),
    Garment("knit jersey", "any", (C, T), ALL),
    Garment("leather", "any", (C, F), (MILD,)),
    Garment("brocade", "any", (F,), (MILD,)),
    Garment("organza", "any", (F,), (MILD, HOT)),
    Garment("chikankari embroidered cotton", "any", (F, C), (HOT,)),
]

COLORS = [
    "white", "off-white cream", "black", "charcoal grey", "light grey", "navy blue",
    "royal blue", "sky blue", "teal", "olive green", "emerald green", "mint green",
    "mustard yellow", "saffron orange", "maroon", "red", "pink", "pastel pink",
    "lavender", "purple", "beige", "khaki", "brown", "gold", "multicolour",
    "peach", "rust", "turquoise",
]

PATTERNS = [
    "solid plain", "vertical stripes", "checked plaid", "floral print", "paisley print",
    "block print", "bandhani tie-dye", "ikat", "zari embroidery", "mirror work",
    "polka dots", "geometric print",
]

VIBE_PROMPTS = {
    W: ["office work attire", "business formal outfit", "smart professional workwear"],
    C: ["relaxed casual everyday outfit", "weekend casual clothes", "laid-back streetwear"],
    F: ["festive traditional celebration outfit", "wedding guest attire", "dressy party outfit"],
    T: ["comfortable travel outfit", "airport travel clothes", "practical outfit for sightseeing"],
}

SEASON_PROMPTS = {
    HOT: ["breathable outfit for hot humid weather", "light summer clothes for heat"],
    WET: ["monsoon outfit for rainy weather", "quick-dry clothes for rain"],
    MILD: ["layered outfit for cool winter weather", "warm clothes for mild cold weather"],
}

TEMPLATES = [
    "a photo of a person wearing {}",
    "an outfit consisting of {}",
    "{} clothing",
]


def anchor_rows() -> list[dict]:
    """Flat list of (axis, label, prompts) rows; row order == .bin row order."""
    rows = []
    for g in CATEGORIES:
        names = (g.label, *g.aliases)
        rows.append({"axis": "category", "label": g.label, "style": g.style,
                     "vibes": list(g.vibes), "seasons": list(g.seasons),
                     "prompts": [t.format(n) for n in names for t in TEMPLATES]})
    for g in FABRICS:
        rows.append({"axis": "fabric", "label": g.label, "vibes": list(g.vibes), "seasons": list(g.seasons),
                     "prompts": [f"clothes made of {g.label} fabric", f"a {g.label} outfit"]})
    for c in COLORS:
        rows.append({"axis": "color", "label": c,
                     "prompts": [f"a person wearing {c} clothes", f"a {c} coloured outfit"]})
    for p in PATTERNS:
        rows.append({"axis": "pattern", "label": p,
                     "prompts": [f"clothing with a {p} pattern", f"a {p} outfit"]})
    for v, ps in VIBE_PROMPTS.items():
        rows.append({"axis": "vibe", "label": v, "prompts": ps})
    for s, ps in SEASON_PROMPTS.items():
        rows.append({"axis": "season", "label": s, "prompts": ps})
    return rows


def taxonomy_hash(rows: list[dict] | None = None) -> str:
    rows = rows or anchor_rows()
    payload = json.dumps([config.MODEL_ID, config.EMBED_DIM, rows], sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class AnchorBank:
    def __init__(self, rows: list[dict], matrix: np.ndarray):
        self.rows = rows
        self.matrix = matrix.astype(np.float32)
        self.axes: dict[str, np.ndarray] = {}
        for axis in {r["axis"] for r in rows}:
            self.axes[axis] = np.array([i for i, r in enumerate(rows) if r["axis"] == axis])

    def __len__(self):
        return len(self.rows)

    def by_label(self, axis: str, label: str) -> dict:
        return next(r for r in self.rows if r["axis"] == axis and r["label"] == label)

    def scores(self, vec: np.ndarray, axis: str) -> list[tuple[str, float]]:
        idx = self.axes[axis]
        sims = self.matrix[idx] @ vec
        order = np.argsort(-sims)
        return [(self.rows[idx[i]]["label"], float(sims[i])) for i in order]

    def classify(self, vec: np.ndarray, top_k: int = 3) -> dict:
        """Zero-shot labels per axis: {'category': [(label, cos), ...], ...}."""
        return {axis: self.scores(vec, axis)[:top_k] for axis in self.axes}


def build_anchor_bank(embedder, force: bool = False) -> AnchorBank:
    rows = anchor_rows()
    h = taxonomy_hash(rows)
    if not force and config.TAXONOMY_BIN.exists() and config.TAXONOMY_META.exists():
        meta = json.loads(config.TAXONOMY_META.read_text())
        if meta.get("hash") == h and type(embedder).__name__ == meta.get("embedder"):
            mat = np.fromfile(config.TAXONOMY_BIN, dtype=np.float32).reshape(len(rows), config.EMBED_DIM)
            return AnchorBank(rows, mat)

    flat, owner = [], []
    for i, r in enumerate(rows):
        flat += r["prompts"]
        owner += [i] * len(r["prompts"])
    vecs = embedder.embed_texts(flat, task="classification")
    owner = np.array(owner)
    mat = np.stack([vecs[owner == i].mean(axis=0) for i in range(len(rows))])
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat.astype(np.float32)

    config.TAXONOMY_BIN.parent.mkdir(parents=True, exist_ok=True)
    mat.tofile(config.TAXONOMY_BIN)
    config.TAXONOMY_META.write_text(json.dumps({
        "hash": h, "model_id": config.MODEL_ID, "dim": config.EMBED_DIM,
        "embedder": type(embedder).__name__, "count": len(rows),
        "rows": [{k: v for k, v in r.items() if k != "prompts"} for r in rows],
    }, indent=1))
    return AnchorBank(rows, mat)


_BANK = None


def get_anchor_bank(embedder=None) -> AnchorBank:
    global _BANK
    if _BANK is None:
        from engine.embedder import get_embedder

        _BANK = build_anchor_bank(embedder or get_embedder())
    return _BANK


# ---------------------------------------------------------------- tag derivation

SOFTMAX_T = 0.02  # cosine gaps between anchors are small (~0.01-0.05); sharpen before voting
FABRIC_NAME_MIN_P = 0.5


def _softmax(axis_scores: list[tuple[str, float]]) -> list[tuple[str, float]]:
    s = np.array([v for _, v in axis_scores], dtype=np.float64)
    p = np.exp((s - s.max()) / SOFTMAX_T)
    p /= p.sum()
    return [(lab, float(pi)) for (lab, _), pi in zip(axis_scores, p)]


def _prior_votes(axis_scores, bank: AnchorBank, axis: str, attr: str) -> dict[str, float]:
    """Distribution over vibes/seasons implied by the (soft) top labels of an axis."""
    votes: dict[str, float] = {}
    for label, p in _softmax(axis_scores):
        targets = bank.by_label(axis, label).get(attr, [])
        for t in targets:
            votes[t] = votes.get(t, 0.0) + p / len(targets)
    return votes


def _combine(*weighted: tuple[float, dict[str, float]]) -> str:
    total: dict[str, float] = {}
    for w, dist in weighted:
        for k, v in dist.items():
            total[k] = total.get(k, 0.0) + w * v
    return max(total, key=total.get)


def derive_tags(labels: dict, bank: AnchorBank, exif_season: str | None = None) -> dict:
    """Turns raw per-axis zero-shot scores (best-first, top-k) into the outfit's stored tags."""
    cat, cat_s = labels["category"][0]
    color = labels["color"][0][0]
    fabric = labels["fabric"][0][0]
    # Human-readable whole-look name, e.g. 'Navy Linen Kurta'. Fabric is the hardest axis to read
    # from pixels, so it only joins the name when the zero-shot vote is confident.
    fabric_word = fabric.split()[-1]
    fabric_p = _softmax(labels["fabric"])[0][1]
    noun = cat if fabric_word in cat or fabric_p < FABRIC_NAME_MIN_P else f"{fabric_word} {cat}"
    primary = f"{color.split()[0]} {noun}".title()

    vibe = _combine((1.0, dict(_softmax(labels["vibe"]))),
                    (1.5, _prior_votes(labels["category"], bank, "category", "vibes")))
    season = _combine((1.0, dict(_softmax(labels["season"]))),
                      (0.6, _prior_votes(labels["category"], bank, "category", "seasons")),
                      (0.6, _prior_votes(labels["fabric"], bank, "fabric", "seasons")),
                      # the user actually wore it in this season - strong evidence
                      (0.8, {exif_season: 1.0} if exif_season else {}))

    return {
        "primary_category": primary,
        "category": cat,
        "category_score": cat_s,
        "fabric_tag": fabric,
        "color_tag": color,
        "vibe_tag": vibe,
        "season_tag": season,
        "style": bank.by_label("category", cat)["style"],
    }


if __name__ == "__main__":
    from engine.embedder import get_embedder

    bank = build_anchor_bank(get_embedder(), force=True)
    counts = {a: len(i) for a, i in bank.axes.items()}
    print(f"Wrote {len(bank)} anchors -> {config.TAXONOMY_BIN}  {counts}")
