"""Ədalətli Kirayə — Streamlit UI with LangGraph human-in-the-loop."""
import uuid

import pandas as pd
import streamlit as st

from src import db
from src.analysis import sample_listing
from src.area import build_area_report
from src.charts import district_bar_chart, market_range_chart
from src.config import COMMERCIAL_TYPES, MAX_AREA_DISTRICTS, MIN_COMPARABLES, PROPERTY_TYPES, llm_available
from src.graph import REMOVE_REASONS, build_graph, resume, start
from src.models import AreaQuery, Listing, MarketStats, Verdict
from src.scraper.normalize import DISTRICTS, is_listing_url

st.set_page_config(page_title="Ədalətli Kirayə", page_icon="🏠", layout="wide")
db.init_db()

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html { font-size: 17px; }
html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif; }

/* Fon */
.stApp {
    background:
        radial-gradient(900px 420px at 8% -5%, rgba(99,102,241,.16), transparent 60%),
        radial-gradient(800px 400px at 100% 0%, rgba(6,182,212,.14), transparent 60%);
}
.block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1200px; }
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

/* ===== HERO ===== */
.hero {
    background: linear-gradient(120deg, #4338CA 0%, #6366F1 45%, #06B6D4 100%);
    border-radius: 28px; padding: 38px 42px; margin-bottom: 22px; color: #fff;
    box-shadow: 0 18px 40px rgba(67,56,202,.28); position: relative; overflow: hidden;
}
.hero::after {
    content: "🏠"; position: absolute; right: 36px; top: 10px; font-size: 9rem; opacity: .16;
}
.hero-title { font-size: 2.6rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1.1; }
.hero-sub { font-size: 1.12rem; opacity: .92; margin-top: 10px; max-width: 640px; }
.chips { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 22px; }
.chip {
    background: rgba(255,255,255,.18); border: 1px solid rgba(255,255,255,.3);
    backdrop-filter: blur(6px); padding: 8px 16px; border-radius: 999px;
    font-weight: 600; font-size: .95rem;
}
.chip span { font-size: 1.25rem; margin-right: 6px; vertical-align: -2px; }

/* ===== STEPPER ===== */
.stepper { display: flex; gap: 12px; margin: 6px 0 22px; }
.step {
    flex: 1; display: flex; align-items: center; gap: 12px; padding: 12px 16px;
    border-radius: 16px; border: 1px solid rgba(127,127,127,.22);
    background: rgba(127,127,127,.06); font-weight: 600;
}
.step .dot {
    width: 34px; height: 34px; border-radius: 50%; display: grid; place-items: center;
    font-weight: 800; background: rgba(127,127,127,.2); font-size: 1rem;
}
.step.active { border-color: #6366F1; background: rgba(99,102,241,.12); box-shadow: 0 6px 18px rgba(99,102,241,.2); }
.step.active .dot { background: linear-gradient(135deg, #4F46E5, #06B6D4); color: #fff; }
.step.done .dot { background: #10B981; color: #fff; }
.step.todo { opacity: .55; }

/* ===== SECTION BAŞLIQLARI ===== */
.sec { display: flex; align-items: center; gap: 12px; margin: 26px 0 10px; }
.sec .ic {
    width: 44px; height: 44px; border-radius: 14px; display: grid; place-items: center; font-size: 1.5rem;
    background: linear-gradient(135deg, rgba(99,102,241,.2), rgba(6,182,212,.2));
}
.sec .tt { font-size: 1.3rem; font-weight: 700; letter-spacing: -0.01em; }

/* ===== HÖKM KARTI ===== */
.verdict {
    display: flex; align-items: center; gap: 22px; padding: 26px 30px; border-radius: 24px;
    border: 1px solid var(--vc); background: linear-gradient(120deg, var(--vbg), rgba(127,127,127,.04));
    box-shadow: 0 12px 30px var(--vsh); margin: 8px 0 14px;
}
.verdict .vicon { font-size: 3.6rem; line-height: 1; }
.verdict .vlabel { font-size: 2rem; font-weight: 800; letter-spacing: -0.02em; color: var(--vc); }
.verdict .vsub { font-size: 1.02rem; opacity: .85; margin-top: 4px; }
.verdict.ədalətli { --vc: #059669; --vbg: rgba(16,185,129,.16); --vsh: rgba(16,185,129,.18); }
.verdict.baha { --vc: #DC2626; --vbg: rgba(239,68,68,.16); --vsh: rgba(239,68,68,.18); }
.verdict.ucuz { --vc: #D97706; --vbg: rgba(245,158,11,.18); --vsh: rgba(245,158,11,.2); }
.verdict.azdir { --vc: #64748B; --vbg: rgba(100,116,139,.16); --vsh: rgba(100,116,139,.15); }

/* ===== ƏN SƏRFƏLİ BÖLGƏ ===== */
.best {
    display: flex; align-items: center; gap: 18px; padding: 22px 28px; border-radius: 22px; margin: 14px 0;
    background: linear-gradient(120deg, rgba(16,185,129,.2), rgba(6,182,212,.14));
    border: 1px solid rgba(16,185,129,.5); box-shadow: 0 10px 26px rgba(16,185,129,.16);
}
.best .bicon { font-size: 3rem; }
.best .bsmall { font-size: .9rem; opacity: .8; font-weight: 600; text-transform: uppercase; letter-spacing: .06em; }
.best .bname { font-size: 1.8rem; font-weight: 800; letter-spacing: -0.02em; }

/* ===== METRİKLƏR / KARTLAR ===== */
[data-testid="stMetric"] {
    background: rgba(127,127,127,.08); border: 1px solid rgba(127,127,127,.2);
    border-radius: 18px; padding: 16px 18px;
}
[data-testid="stMetricLabel"] { font-weight: 600; opacity: .8; }
[data-testid="stMetricValue"] { font-size: 1.6rem; font-weight: 800; }

[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 22px; box-shadow: 0 6px 22px rgba(0,0,0,.06);
}

/* ===== DÜYMƏLƏR ===== */
.stButton > button, [data-testid^="stBaseButton"] {
    border-radius: 14px; font-weight: 700; padding: .65rem 1.1rem; font-size: 1rem;
    transition: transform .1s ease, box-shadow .2s ease;
}
.stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 20px rgba(79,70,229,.28); }
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(120deg, #4F46E5, #06B6D4); border: none; color: #fff;
}

/* ===== TABLAR ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px; background: rgba(127,127,127,.08); padding: 6px; border-radius: 16px; border-bottom: none;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 12px; padding: 12px 24px; font-weight: 700; font-size: 1.05rem; height: auto;
}
.stTabs [aria-selected="true"] { background: linear-gradient(120deg, #4F46E5, #06B6D4); color: #fff !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ===== İNPUTLAR ===== */
[data-baseweb="input"], [data-baseweb="select"] > div, [data-baseweb="base-input"] { border-radius: 14px !important; }
.stTextInput input { font-size: 1.05rem; padding: 14px 16px; }

/* ===== CƏDVƏL / ALERT ===== */
[data-testid="stAlert"] { border-radius: 16px; }
[data-testid="stDataFrame"], [data-testid="stDataEditor"] { border-radius: 16px; overflow: hidden; }
[data-testid="stExpander"] { border-radius: 16px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

VERDICT_STYLE = {
    "ədalətli": ("🟢", "Ədalətli qiymət", "ədalətli"),
    "baha": ("🔴", "Baha", "baha"),
    "ucuz": ("🟡", "Ucuz", "ucuz"),
    "məlumat azdır": ("⚪", "Məlumat azdır", "azdir"),
}


@st.cache_resource
def get_graph():
    # One compiled graph + MemorySaver per server process; threads are separated by thread_id.
    return build_graph()


def reset() -> None:
    for k in ("step", "review", "final", "thread_id"):
        st.session_state.pop(k, None)


ss = st.session_state
ss.setdefault("step", 1)
ss.setdefault("thread_id", str(uuid.uuid4()))


def section(icon: str, title: str) -> None:
    st.markdown(f"<div class='sec'><div class='ic'>{icon}</div><div class='tt'>{title}</div></div>",
                unsafe_allow_html=True)


def stepper(current: int) -> None:
    labels = ["Link", "Yoxla", "Nəticə"]
    parts = []
    for i, label in enumerate(labels, start=1):
        cls = "done" if i < current else "active" if i == current else "todo"
        dot = "✓" if i < current else str(i)
        parts.append(f"<div class='step {cls}'><div class='dot'>{dot}</div><div>{label}</div></div>")
    st.markdown(f"<div class='stepper'>{''.join(parts)}</div>", unsafe_allow_html=True)


def listing_card(x: Listing) -> None:
    with st.container(border=True):
        st.markdown(f"### 🏷️ {x.title or 'Elan'}")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Qiymət", f"{x.price_azn} AZN")
        c2.metric("📐 Sahə", f"{x.area_m2:g} m²")
        c3.metric("🚪 Otaq", x.rooms if x.rooms else "—")
        c4.metric("🏢 Növ", x.property_type)
        extra = [f"📍 {x.district}"]
        if x.metro:
            extra.append(f"🚇 {x.metro} m.")
        if x.floor:
            extra.append(f"🪜 {x.floor}/{x.total_floors} mərtəbə")
        extra.append("✨ təmirli" if x.renovated else "🧱 təmirsiz" if x.renovated is False else "❔ təmir naməlum")
        st.caption(" · ".join(extra) + f" · [elana bax ↗]({x.url})")


def begin(url: str | None = None, sample: Listing | None = None, guess: int | None = None) -> None:
    ss.thread_id = str(uuid.uuid4())
    with st.status("Analiz gedir…", expanded=True) as status:
        try:
            payload = start(get_graph(), ss.thread_id, url=url, sample=sample, user_guess=guess or None,
                            on_step=lambda s: status.write(s))
        except ValueError as exc:
            status.update(label="Xəta", state="error")
            st.error(str(exc))
            return
        except Exception as exc:  # network etc.
            status.update(label="Xəta", state="error")
            st.error(f"Gözlənilməz xəta: {exc}")
            return
        status.update(label="Oxşar elanlar tapıldı — yoxla", state="complete", expanded=False)
    ss.review = payload
    ss.step = 2
    st.rerun()


# ---------- Hero ----------
st.markdown("""
<div class="hero">
  <div class="hero-title">🏠 Ədalətli Kirayə</div>
  <div class="hero-sub">Ev, ofis və ya obyekt — razılaşmazdan əvvəl qiymətin bazara uyğun olub-olmadığını öyrən.</div>
  <div class="chips">
    <div class="chip"><span>🔍</span>Elanı yoxla</div>
    <div class="chip"><span>📊</span>Bazar müqayisəsi</div>
    <div class="chip"><span>🗺️</span>Rayonları müqayisə et</div>
    <div class="chip"><span>✉️</span>Danışıq mesajı</div>
  </div>
</div>
""", unsafe_allow_html=True)

TAB_CHECK, TAB_AREA = "🔗 Elanı yoxla", "🗺️ Bölgə üzrə bax"
tab_check, tab_area = st.tabs([TAB_CHECK, TAB_AREA], key="main_tabs", on_change="rerun")


def go_check(url: str) -> None:
    """Callback (runs before widgets are created): switch to the check tab and analyze url."""
    reset()
    ss.pending_url = url
    ss.main_tabs = TAB_CHECK


def theme_type() -> str:
    return getattr(getattr(st.context, "theme", None), "type", None) or "dark"


with tab_check:
    stepper(ss.step)

    # ---------- Step 1: input ----------
    if ss.step == 1 and ss.get("pending_url"):
        # Came from "Bu elanı yoxla" in the area tab.
        begin(url=ss.pop("pending_url"))
    elif ss.step == 1:
        with st.container(border=True):
            section("🔗", "Elanın linkini yapışdır")
            url = st.text_input("bina.az elanının linki", placeholder="https://bina.az/items/1234567")
            guess = st.number_input("🤔 Sənə görə bu yerin ədalətli qiyməti neçədir? (AZN, istəyə bağlı)",
                                    min_value=0, step=50, value=0)
            c1, c2 = st.columns([2, 1])
            if c1.button("🔍 Analiz et", type="primary", use_container_width=True):
                if not is_listing_url(url):
                    st.error("Link düzgün deyil. Nümunə: https://bina.az/items/1234567")
                else:
                    begin(url=url, guess=guess)
            if c2.button("🧪 Nümunə elanla sına", use_container_width=True):
                x = sample_listing()
                if x is None:
                    st.error("Bazada nümunə elan yoxdur. Əvvəlcə seed_data.py-ı işə sal.")
                else:
                    begin(sample=x, guess=guess)

    # ---------- Step 2: human review ----------
    elif ss.step == 2:
        review = ss.review
        target = Listing.model_validate(review["listing"])
        listing_card(target)
        cands = review["candidates"]
        section("🧐", f"Oxşar elanlar ({len(cands)})")
        st.caption("Uyğun olmayanların işarəsini götür və səbəbini seç. Hökm yalnız seçilmiş elanlarla hesablanacaq.")
        df = pd.DataFrame([
            {
                "✅": True,
                "Qiymət": c["price_azn"],
                "m²": c["area_m2"],
                "Otaq": c.get("rooms"),
                "Mərtəbə": f"{c['floor']}/{c['total_floors']}" if c.get("floor") else "—",
                "Rayon": c["district"],
                "Təmir": "✅" if c.get("renovated") else "—",
                "Niyə oxşar": c.get("reason", ""),
                "Link": c["url"],
                "Çıxarma səbəbi": "—",
                "id": c["id"],
            }
            for c in cands
        ])
        edited = st.data_editor(
            df,
            hide_index=True,
            use_container_width=True,
            disabled=["Qiymət", "m²", "Otaq", "Mərtəbə", "Rayon", "Təmir", "Niyə oxşar", "Link", "id"],
            column_order=["✅", "Qiymət", "m²", "Otaq", "Mərtəbə", "Rayon", "Təmir", "Niyə oxşar", "Çıxarma səbəbi", "Link"],
            column_config={
                "✅": st.column_config.CheckboxColumn("✅", help="Müqayisəyə daxil et"),
                "Link": st.column_config.LinkColumn(display_text="aç"),
                "Çıxarma səbəbi": st.column_config.SelectboxColumn(options=["—", *REMOVE_REASONS], required=True),
                "Niyə oxşar": st.column_config.TextColumn(width="medium"),
            },
            key=f"editor_{ss.thread_id}",
        )
        approved = edited[edited["✅"]]["id"].tolist()
        removed = edited[~edited["✅"]]
        reasons = {r["id"]: r["Çıxarma səbəbi"] if r["Çıxarma səbəbi"] in REMOVE_REASONS else "digər"
                   for _, r in removed.iterrows()}
        if len(approved) < MIN_COMPARABLES:
            st.warning(f"{len(approved)} elan seçilib. {MIN_COMPARABLES}-dən az olsa, hökm \"məlumat azdır\" olacaq.")
        b1, b2 = st.columns([2, 1])
        if b1.button(f"✅ Təsdiqlə və hesabla ({len(approved)} elan)", type="primary", use_container_width=True):
            with st.status("Hesablanır…", expanded=True) as status:
                ss.final = resume(get_graph(), ss.thread_id, approved, reasons, on_step=lambda s: status.write(s))
                status.update(label="Hazırdır", state="complete", expanded=False)
            ss.step = 3
            st.rerun()
        if b2.button("↩️ Geri", use_container_width=True):
            reset()
            st.rerun()

    # ---------- Step 3: result ----------
    else:
        final = ss.final
        target = Listing.model_validate(final["listing"])
        v = Verdict.model_validate(final["verdict"])
        s = MarketStats.model_validate(final["stats"]) if final.get("stats") else None
        listing_card(target)

        icon, label, cls = VERDICT_STYLE[v.label]
        sign = "+" if v.diff_azn > 0 else ""
        sub = ""
        if s and v.label != "məlumat azdır":
            sub = (f"Bazardan <b>{sign}{v.diff_azn} AZN ({sign}{v.diff_pct:.0f}%)</b> · "
                   f"ədalətli aralıq: <b>{v.fair_range[0]}–{v.fair_range[1]} AZN</b>")
        st.markdown(
            f"<div class='verdict {cls}'><div class='vicon'>{icon}</div>"
            f"<div><div class='vlabel'>{label}</div><div class='vsub'>{sub}</div></div></div>",
            unsafe_allow_html=True,
        )
        if v.label == "ucuz":
            st.caption("⚠️ Qiymət bazardan xeyli aşağıdırsa, elanı diqqətlə yoxla (fırıldaq ehtimalı).")
        if v.explanation:
            st.write(v.explanation)

        if s:
            m1, m2, m3 = st.columns(3)
            m1.metric("📊 Median qiymət", f"{s.median_price:.0f} AZN")
            m2.metric("📏 1 m² qiyməti", f"{s.median_price_per_m2:.1f} AZN")
            m3.metric("🧮 Müqayisə edilən elan", s.n)
            if v.user_guess:
                st.write(f"🤔 Sənin təxminin bazar medianına **{abs(v.user_guess - s.median_price):.0f} AZN** yaxın idi.")

        approved = final.get("approved") or []
        if s and approved:
            section("📈", "Bazar aralığı")
            theme = theme_type()
            st.altair_chart(
                market_range_chart(s, v, target.price_azn, [c["price_azn"] for c in approved], theme=theme),
                use_container_width=True, theme=None,
            )
            with st.expander("📋 Cədvəl görünüşü"):
                st.dataframe(
                    pd.DataFrame([{"Qiymət": c["price_azn"], "m²": c["area_m2"], "Otaq": c.get("rooms"),
                                   "Rayon": c["district"], "Link": c["url"]} for c in approved]),
                    hide_index=True, column_config={"Link": st.column_config.LinkColumn(display_text="aç")},
                )

        if v.negotiation_message:
            section("✉️", "Ev sahibinə mesaj")
            st.code(v.negotiation_message, language=None, wrap_lines=True)

        removed = final.get("removed_reasons") or {}
        if removed:
            st.caption(f"🗑️ Sən {len(removed)} elanı çıxardın: " + ", ".join(sorted(set(removed.values()))))

        if st.button("🔄 Yeni analiz", type="primary"):
            reset()
            st.rerun()

with tab_area:
    st.caption("Sahibkar və ya ev axtaran üçün: hansı rayon büdcənə daha sərfəlidir?")
    with st.container(border=True):
        section("🎛️", "Axtarış parametrləri")
        ptype = st.segmented_control(
            "Əmlak növü", PROPERTY_TYPES, default="obyekt", required=True,
            format_func=lambda t: {"mənzil": "🏢 Mənzil", "həyət evi": "🏡 Həyət evi", "ofis": "💼 Ofis", "obyekt": "🏪 Obyekt"}[t],
            key="area_ptype",
        )
        counts = {d: n for t, d, n in db.count_by() if t == ptype}
        options = sorted(DISTRICTS, key=lambda d: -counts.get(d, 0))
        districts = st.multiselect(
            f"📍 Rayonlar (1–{MAX_AREA_DISTRICTS})", options, default=[d for d in options[:3] if counts.get(d)],
            max_selections=MAX_AREA_DISTRICTS, format_func=lambda d: f"{d} ({counts.get(d, 0)} elan)", key="area_districts",
        )
        a1, a2 = st.columns(2)
        area_rng = a1.slider("📐 Sahə (m²)", 10, 400, (40, 120) if ptype in COMMERCIAL_TYPES else (50, 120), step=5,
                             key=f"area_rng_{ptype}")
        budget = a2.number_input("💰 Aylıq büdcə (AZN, istəyə bağlı)", min_value=0, step=100, value=0, key="area_budget")
        rooms = None
        if ptype not in COMMERCIAL_TYPES:
            rooms = st.select_slider("🚪 Otaq sayı", ["fərq etməz", 1, 2, 3, 4, 5], value="fərq etməz", key="area_rooms")
            rooms = None if rooms == "fərq etməz" else rooms
        live = st.toggle("🔄 bina.az-dan yenilə (rayon başına ~5 s)", value=False, key="area_live",
                         help="Söndürülü olanda yalnız yığılmış bazadan istifadə olunur.")

        if st.button("📊 Bazara bax", type="primary", use_container_width=True, disabled=not districts):
            q = AreaQuery(property_type=ptype, districts=districts, area_min=area_rng[0], area_max=area_rng[1],
                          rooms=rooms, budget_azn=budget or None)
            with st.status("Bazar analiz olunur…", expanded=True) as status:
                ss.area = build_area_report(q, live=live, on_step=lambda s: status.write(s))
                status.update(label="Hazırdır", state="complete", expanded=False)

    ar = ss.get("area")
    if ar:
        rep = ar.report
        q = rep.query
        if rep.best_value_district:
            st.markdown(
                f"<div class='best'><div class='bicon'>🏆</div><div>"
                f"<div class='bsmall'>Ən sərfəli bölgə</div><div class='bname'>{rep.best_value_district}</div>"
                f"</div></div>",
                unsafe_allow_html=True,
            )
        st.write(rep.summary)

        section("📊", "Rayonların müqayisəsi")
        tbl = pd.DataFrame([
            {
                "Rayon": ("🏆 " if s.district == rep.best_value_district else "") + s.district,
                "Median qiymət": f"{s.median_price:.0f} AZN" if s.median_price is not None else "—",
                "1 m² qiyməti": f"{s.median_price_per_m2:.1f} AZN" if s.median_price_per_m2 is not None else "—",
                "Aralıq (p25–p75)": f"{s.p25:.0f}–{s.p75:.0f}" if s.p25 is not None else "—",
                "Elan sayı": s.n,
                "Büdcəyə uyğun": s.within_budget if q.budget_azn else "—",
                "Qeyd": "" if s.enough_data else "məlumat azdır",
            }
            for s in rep.districts
        ])
        st.dataframe(tbl, hide_index=True, use_container_width=True)
        if any(s.n for s in rep.districts):
            st.altair_chart(district_bar_chart(rep.districts, q.budget_azn, rep.best_value_district, theme_type()),
                            use_container_width=True, theme=None)

        if rep.matching_listings:
            title = "Büdcəyə uyğun elanlar" if q.budget_azn else "Elanlar"
            section("🏘️", f"{title} ({len(rep.matching_listings)})")
            for x in rep.matching_listings:
                with st.container(border=True):
                    c1, c2 = st.columns([5, 1.4], vertical_alignment="center")
                    desc = f"**💰 {x.price_azn} AZN** · 📐 {x.area_m2:g} m² · 📍 {x.district}"
                    if x.rooms:
                        desc += f" · 🚪 {x.rooms} otaq"
                    if x.floor:
                        desc += f" · 🪜 {x.floor}/{x.total_floors}"
                    if x.metro:
                        desc += f" · 🚇 {x.metro} m."
                    desc += f" · {x.price_per_m2:.1f} AZN/m² · [aç ↗]({x.url})"
                    c1.markdown(desc)
                    c2.button("🔍 Bu elanı yoxla", key=f"chk_{x.id}", use_container_width=True,
                              on_click=go_check, args=(x.url,))
        elif q.budget_azn:
            st.info("Büdcəyə uyğun elan tapılmadı. Büdcəni və ya sahə aralığını genişləndir.")

st.divider()
updated = db.last_updated()
footer = f"🕒 Məlumat tarixi: {updated:%d.%m.%Y %H:%M} · " if updated else ""
final = ss.get("final")
area_res = ss.get("area")
if area_res and area_res.used_llm:
    footer += f"Bölgə xülasəsinin xərci: ${area_res.cost_usd:.4f} · "
if final and ss.step == 3:
    if final.get("used_agent"):
        footer += f"Analizin xərci: ${final.get('cost_usd', 0):.4f} ({final.get('input_tokens', 0) + final.get('output_tokens', 0)} token) · "
    else:
        footer += "Sadə rejim (AI agentsiz) · "
elif not llm_available():
    footer += "AI açarı qurulmayıb — sadə rejim · "
st.caption(footer + "Bu, elanlar əsasında təxmindir, rəsmi qiymətləndirmə deyil.")
