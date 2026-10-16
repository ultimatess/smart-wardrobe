"""Gate 3 - recommendation math backtest (CLI, no UI).

    uv run python tests/gate3_backtest.py             # mock bi-cultural wardrobe (text proxies)
    uv run python tests/gate3_backtest.py --db data/wardrobe.db   # your real ingested wardrobe

Feeds 5 daily scenarios through the decay-weighted scorer. Pass: for every scenario the top-3
are weather-appropriate (no rain/heat fabric warnings), match the vibe pill (>= 2 of 3), and
respect the Anti-Repeat Radar (nothing worn in the last 3 days; no same-venue repeat at #1).

Mock mode embeds each outfit's description as an EmbeddingGemma 2 *document* vector - the same
shared space the image crops live in - so the scoring math is exercised end-to-end with the real model.
"""
import argparse
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine import db  # noqa: E402
from engine.embedder import get_embedder  # noqa: E402
from engine.recommender import HEAT_UNSAFE, RAIN_UNSAFE, DayContext, Recommender  # noqa: E402

HOT, WET, MILD = "Summer/Humid", "Monsoon", "Mild/Winter"
# (description, vibe, season, fabric, wears as [(days_before_anchor, venue)])
MOCK = [
    ("white linen kurta with cotton pajama, breathable office wear", "Work", HOT, "linen", [(40, "office")]),
    ("light blue cotton formal shirt with grey trousers", "Work", HOT, "cotton", [(1, "office")]),
    ("navy linen shirt with beige chinos, smart casual", "Work", HOT, "linen", [(25, "office")]),
    ("olive polo t-shirt with khaki chinos", "Work", WET, "cotton", [(10, "office")]),
    ("charcoal wool two-piece business suit with tie", "Work", MILD, "wool", [(60, "office")]),
    ("black polyester quick-dry shirt with dark trousers, rain-ready", "Work", WET, "polyester", [(30, "office")]),
    ("cream khadi kurta with nehru jacket", "Work", HOT, "khadi", [(15, "office")]),
    ("beige cotton kurta with white churidar", "Work", HOT, "cotton", [(55, "office")]),
    ("pastel pink cotton shirt with navy trousers", "Work", WET, "cotton", [(48, "office")]),
    ("lavender handloom cotton saree for the office", "Work", WET, "cotton", [(70, "office")]),
    ("white cotton shirt with olive chinos", "Work", HOT, "cotton", [(33, "office"), (80, "office")]),
    ("maroon raw silk sherwani with churidar for a wedding", "Festive", MILD, "raw silk", [(120, "hall")]),
    ("red kanjeevaram silk saree with gold zari border", "Festive", MILD, "silk", [(200, "hall")]),
    ("pink georgette anarkali suit with dupatta", "Festive", WET, "georgette", [(90, "hall")]),
    ("yellow chikankari embroidered cotton kurta pajama for a puja", "Festive", HOT, "chikankari embroidered cotton", [(35, "hall")]),
    ("teal chiffon saree with mirror work blouse", "Festive", WET, "chiffon", [(2, "hall")]),
    ("emerald georgette sharara set with dupatta", "Festive", WET, "georgette", [(150, "hall")]),
    ("mustard cotton kurta with printed nehru jacket for a function", "Festive", WET, "cotton", [(60, "hall")]),
    ("peach georgette lehenga choli", "Festive", MILD, "georgette", [(240, "hall")]),
    ("blue denim jeans with white graphic t-shirt", "Casual", MILD, "denim", [(5, "home")]),
    ("grey hoodie and black joggers", "Casual", MILD, "knit jersey", [(12, "home")]),
    ("floral rayon summer sundress", "Casual", HOT, "rayon", [(20, "home")]),
    ("linen shirt with cotton shorts for the weekend", "Casual", HOT, "linen", [(8, "home")]),
    ("white cotton t-shirt with olive cargo shorts", "Casual", HOT, "cotton", [(30, "home")]),
    ("striped cotton shirt with blue jeans", "Casual", MILD, "cotton", [(45, "cafe")]),
    ("grey knit sweater with dark jeans", "Casual", MILD, "knit jersey", [(90, "cafe")]),
    ("comfortable cotton co-ord set for travel", "Travel", HOT, "cotton", [(50, "airport")]),
    ("cargo pants with breathable t-shirt and sneakers for sightseeing", "Travel", WET, "cotton", [(70, "airport")]),
    ("rain jacket over quick-dry t-shirt and track pants", "Travel", WET, "polyester", [(100, "airport")]),
]

ANCHOR = datetime(2026, 8, 12, 8, 30)  # a Wednesday
SCENARIOS = [
    ("34C humid Wednesday + Work", DayContext("Work", ANCHOR, 34, 80, False, "office")),
    ("Rainy Sunday evening + Festive", DayContext("Festive", ANCHOR + timedelta(days=4, hours=10), 26, 92, True, "hall")),
    ("Mild 18C Saturday + Casual", DayContext("Casual", ANCHOR + timedelta(days=3, hours=2), 18, 45, False, "cafe")),
    ("Warm 29C Friday + Travel", DayContext("Travel", ANCHOR + timedelta(days=2, hours=-3), 29, 60, False, "airport")),
    ("Monsoon Thursday + Work (office radar)", DayContext("Work", ANCHOR + timedelta(days=1), 27, 88, True, "office")),
]


VENUE_NAMES = {"home": "Home", "office": "Office", "hall": "Function hall", "cafe": "Cafe", "airport": "Airport"}


def build_mock(conn, embedder) -> None:
    vecs = embedder.embed_texts([m[0] for m in MOCK], task="document")
    with conn:
        conn.executemany("INSERT INTO venue_names VALUES (?, ?)", VENUE_NAMES.items())
        for (desc, vibe, season, fabric, wears), v in zip(MOCK, vecs):
            last = max(ANCHOR - timedelta(days=d) for d, _ in wears).isoformat()  # refined by wear_logs
            oid = db.insert_outfit(conn, {
                "primary_category": desc.split(",")[0], "vibe_tag": vibe, "season_tag": season,
                "fabric_tag": fabric, "labels_json": "{}", "wear_count": len(wears), "last_worn_timestamp": last,
            }, v)
            for d, venue in wears:
                # photos are taken mid-day / evening at the venue (drives Home/Office inference)
                worn = ANCHOR - timedelta(days=d) + timedelta(hours=2 if venue == "office" else 11)
                conn.execute("INSERT INTO wear_logs (outfit_id, worn_at, location_cluster) VALUES (?,?,?)",
                             (oid, worn.isoformat(), venue))


def check(ctx: DayContext, top: list[dict]) -> list[str]:
    fails = []
    if sum(r["vibe_tag"] == ctx.vibe for r in top) < 2:
        fails.append("vibe: fewer than 2/3 match the pill")
    for r in top:
        fab = (r["fabric_tag"] or "").lower()
        if (ctx.rain and fab in RAIN_UNSAFE) or (ctx.season == HOT and fab in HEAT_UNSAFE):
            fails.append(f"weather: {r['primary_category']} ({fab})")
        if r["days_since_worn"] is not None and r["days_since_worn"] < 3:
            fails.append(f"anti-repeat: {r['primary_category']} worn {r['days_since_worn']}d ago")
    if top and top[0]["breakdown"]["venue_radar"] < 1:
        fails.append(f"venue radar: #1 {top[0]['primary_category']} repeats at this venue")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, help="backtest an existing wardrobe.db instead of the mock wardrobe")
    args = ap.parse_args()
    emb = get_embedder()
    if args.db:
        conn = db.get_db(args.db)
    else:
        path = Path(tempfile.mkdtemp(prefix="gate3-")) / "mock.db"
        db.init_db(path, verbose=False)
        conn = db.get_db(path)
        build_mock(conn, emb)
    rec = Recommender(conn, embedder=emb)
    from engine.context import venue_labels
    print("venues:", venue_labels(conn))

    passed = 0
    for name, ctx in SCENARIOS:
        full = rec.recommend(ctx, top_k=10_000)
        res = {**full, "results": full["results"][:3]}
        fails = check(ctx, res["results"])
        flagged = [r for r in full["results"] if r["breakdown"]["venue_radar"] < 1]
        if not args.db and ctx.venue_cluster in ("office", "hall") and not flagged:
            fails.append("venue radar never fired (expected recent repeats at this venue)")
        passed += not fails
        print(f"\n### {name}  [{ctx.when:%a %d %b %H:%M}, season={res['season']}]")
        print(f"    query: \"{res['context']}\"")
        for i, r in enumerate(res["results"], 1):
            b = r["breakdown"]
            print(f"  {i}. {r['primary_category']:<58} {r['vibe_tag']:<8} score={r['score']:.3f} "
                  f"(sim {b['similarity']:.3f} x rec {b['recency']:.2f} x fit {b['context_fit']:.2f} x radar {b['venue_radar']})")
            for w in r["warnings"]:
                print(f"       ! {w}")
        if flagged:
            print(f"    radar held back: {', '.join(r['primary_category'] for r in flagged)}")
        print("    ->", "PASS" if not fails else "FAIL: " + "; ".join(fails))
    print(f"\n== GATE 3: {'PASS' if passed == len(SCENARIOS) else 'FAIL'} ({passed}/{len(SCENARIOS)} scenarios)")
    return 0 if passed == len(SCENARIOS) else 1


if __name__ == "__main__":
    sys.exit(main())
