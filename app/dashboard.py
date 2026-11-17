"""NIFTYTAIL dashboard (D55, D58, D59): a thin Streamlit UI over niftytail.dashboard_data.

Run from the repository root in .venv-dashboard on Windows (README, "Dashboard"). Four sections:
Today (local only), Track record, How it works, Proof. Every number is tracked JSON, the local
forecast of `forecast`, the owner's own data, or closed-form arithmetic on those. Any deployment
must set NIFTYTAIL_PUBLIC=1, which removes Today, "Day by day" and every reports/runs and
reports/live access.
"""

from __future__ import annotations

import streamlit as st

from niftytail import dashboard_data as dd

ROOT = dd.HERE
LABEL_NOTE = (
    "Labels per D49 T6: *borderline* = the deciding p-value lies in [0.01, 0.10]; "
    "*pass-dependent* = the verdict at 5% differs across PR W1000, PR W2000 and TRI W1000. "
    "≤1e-4 is the Monte Carlo floor 1/(B+1)."
)
MAIN_FINDING = "Main finding on the price index (pass-dependent, borderline)"  # D62
LAST_SESSION = "Last session in your data (end of day). Intraday prices are not shown."
DEMO_SESSION = "Last session (synthetic demo)"  # D65 (i)
NSE_BUTTON = "Today's market on NSE ↗"
CHECK_LINE = "{k}/{n} past days recomputed from scratch match the published backtest"
# Mode-aware surface colours (D59 (r)); the dark values are the shipped theme.
SURFACE = {
    "dark": {"card": "#1e2329", "border": "#2b3139", "muted": "#929aa5", "figure": "#ffffff"},
    "light": {"card": "#f5f5f5", "border": "#eaecef", "muted": "#5e6673", "figure": "#ffffff"},
}
CSS = """
<style>
.block-container {{padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px;}}
h1 {{font-size: 2rem; font-weight: 600; letter-spacing: -0.02em;}}
h2, h3 {{font-weight: 600; letter-spacing: -0.01em;}}
[data-testid="stMetric"] {{background: {card}; border: 1px solid {border}; border-radius: 12px;
  padding: 16px; box-shadow: none;}}
[data-testid="stMetricValue"] {{font-size: 40px; font-weight: 700;
  font-variant-numeric: tabular-nums;}}
[data-testid="stMetricLabel"], [data-testid="stCaptionContainer"] {{color: {muted};}}
[data-testid="stTable"], [data-testid="stDataFrame"], table {{font-variant-numeric: tabular-nums;}}
.nt-card {{background: {card}; border: 1px solid {border}; border-radius: 12px; padding: 16px;
  margin-bottom: 1rem; font-variant-numeric: tabular-nums;}}
.nt-kpi-label {{color: {muted}; font-size: 0.875rem;}}
.nt-kpi-small {{font-size: 22px; font-weight: 700; white-space: normal; overflow-wrap: anywhere;
  line-height: 1.25; margin-top: 0.35rem;}}
.nt-figure {{background: {figure}; border-radius: 12px; padding: 16px; margin: 0.5rem 0;}}
.nt-figure img {{width: 100%; height: auto;}}
[data-testid="stNumberInputStepUp"]:hover, [data-testid="stNumberInputStepDown"]:hover
  {{color: #0b0e11 !important;}}
.nt-banner {{padding: 0.75rem 1rem; border-radius: 12px; background: {card};
  border: 1px solid {border}; border-left: 4px solid #f6465d; font-weight: 600;
  margin-bottom: 1rem;}}
</style>
"""


# --- cached loaders (keyed on the mode and paths, so a test can switch them) -------------------
@st.cache_data(show_spinner=False)
def headline() -> dict[str, str]:
    return dd.headline(ROOT)


@st.cache_data(show_spinner=False)
def metrics() -> list[dict[str, str]]:
    return dd.metrics(ROOT)


@st.cache_data(show_spinner=False)
def svg(fid: str) -> str:
    return dd.svg_data_uri(ROOT, fid)


@st.cache_data(show_spinner=False)
def coverage_rows(pid: str, level: str) -> list[dict[str, str]]:
    return dd.coverage_rows(ROOT, pid, level)


@st.cache_data(show_spinner=False)
def dm_rows(pid: str) -> list[dict[str, str]]:
    return dd.dm_rows(ROOT, pid)


@st.cache_data(show_spinner=False)
def sensitivity_rows(level: str) -> list[dict[str, str]]:
    return dd.sensitivity_rows(ROOT, level)


@st.cache_data(show_spinner=False)
def provenance() -> dict[str, object]:
    return dd.provenance(ROOT)


@st.cache_data(show_spinner=False)
def local_figures(public: bool, run_dir: str) -> list[dict[str, str]] | None:
    return dd.local_figures(ROOT, {"NIFTYTAIL_PUBLIC": "1"} if public else {})


@st.cache_data(show_spinner=False, ttl=60)
def live_bundle(public: bool, live_dir: str) -> dict[str, object] | None:
    return dd.live_bundle(ROOT, {"NIFTYTAIL_PUBLIC": "1"} if public else {})


# --- helpers -------------------------------------------------------------------------------------
def mode() -> str:
    return "light" if st.context.theme.type == "light" else "dark"


def figure(uri: str, caption: str) -> None:
    st.markdown(
        f'<div class="nt-figure"><img src="{uri}" alt="{caption}"></div>', unsafe_allow_html=True
    )
    st.caption(caption)


def card(html: str) -> None:
    st.markdown(f'<div class="nt-card">{html}</div>', unsafe_allow_html=True)


def chart(spec: dict[str, object]) -> None:
    st.vega_lite_chart(spec=spec, width="stretch", theme=None)


def banner(text: str = dd.BANNER) -> None:
    st.markdown(f'<div class="nt-banner">{text}</div>', unsafe_allow_html=True)


def secret_public() -> bool:
    """D62 (b): NIFTYTAIL_PUBLIC = "1" as a top-level key of st.secrets (Community Cloud)."""
    try:
        return str(st.secrets.get("NIFTYTAIL_PUBLIC", "")) == "1"
    except Exception:  # no secrets file: not set
        return False


@st.cache_data(show_spinner=False)
def demo_bundle() -> dict[str, object] | None:
    return dd.demo_bundle(ROOT)


def pass_and_level(key: str) -> tuple[str, str]:
    left, right = st.columns(2)
    labels = {label: pid for pid, label, _ in dd.PASSES}
    pid = labels[left.selectbox("Pass", list(labels), key=f"{key}-pass")]
    level = right.selectbox("VaR level", list(dd.LEVELS), index=1, key=f"{key}-level")
    return pid, level


# --- Today (local only) --------------------------------------------------------------------------
def today(bundle: dict[str, object] | None, demo: bool = False) -> None:
    banner(dd.DEMO_BANNER if demo else dd.BANNER)
    if bundle is None:
        st.info(dd.NO_FORECAST)
        return
    doc, ser, m = bundle["doc"], bundle["series"], mode()
    title = (lambda name: f"{name} ({dd.DEMO_TITLE})") if demo else (lambda name: None)
    fr = dd.freshness(doc, ser)
    card(
        f"<b>{fr['text']}</b> · last session from the {fr['source']} · "
        f"forecast commit <code>{bundle['commit']}</code>"
    )
    for w in (None if demo else fr["warning"], bundle["warning"]):
        if w is not None:
            st.warning(w)

    st.subheader("Market")
    left, right = st.columns([3, 1])
    rng = left.radio("Range", list(dd.RANGES), index=3, horizontal=True, key="range")
    log = right.radio("Scale", ["Log", "Linear"], horizontal=True, key="scale") == "Log"
    *nums, src = dd.kpis(ser, rng)
    cols = st.columns(4)
    for col, k in zip(cols, nums, strict=False):
        col.metric(k["label"], k["value"])
    with cols[3]:  # D61 (b): a smaller, wrapping value so the source is never truncated
        card(
            f'<div class="nt-kpi-label">{src["label"]}</div>'
            f'<div class="nt-kpi-small">{src["value"]}</div>'
        )
    chart(dd.market_spec(doc, ser, rng, log, m, title("Market")))
    inside = dd.window_caption(doc, rng)
    if inside is not None:
        st.caption(inside)
    chart(dd.drawdown_spec(ser, rng, m, title("Drawdown")))
    last = dd.last_session(ser)
    label = DEMO_SESSION if demo else LAST_SESSION
    card(
        f"<b>{label}</b><br>{last['date']} · open {last['open']} · high {last['high']} · "
        f"low {last['low']} · close {last['close']} · {last['change vs the previous close']} "
        "vs the previous close"
    )
    if not demo:  # D65 (i): no NSE link beside synthetic numbers
        st.link_button(NSE_BUTTON, dd.NSE_URL)

    st.subheader(f"Risk for the next trading day after {doc['as_of']}")
    left, right = st.columns(2)
    position = left.number_input(
        "Long position (₹)", min_value=0.0, value=1_000_000.0, step=100_000.0, format="%.0f"
    )
    budget = right.number_input(
        "Risk budget for the 99% ES (₹)",
        min_value=0.0,
        value=50_000.0,
        step=10_000.0,
        format="%.0f",
    )
    st.caption(f"Position {dd.inr(position)}; risk budget {dd.inr(budget)}. {dd.PCT_CAPTION}")
    cards = st.columns(3)
    for col, key in zip(cards, dd.RISK_CARDS, strict=False):
        c = dd.risk_card(doc, key, position)
        col.metric(f"{c['model']}: 99% VaR", c["VaR ₹"])
        col.caption(f"{c['VaR %']} of position")
        col.metric(f"{c['model']}: 99% ES", c["ES ₹"])
        col.caption(f"{c['ES %']} of position")
        col.caption(dd.PASSED_NOTE)
    cards[2].metric(
        "Volatility vs its 4-year average",
        dd.volatility_ratio(doc),
        help="sigma_next of the GARCH-EVT filter / the sample std of the last W = 1000 losses",
    )
    cards[2].caption(dd.VOL_NOTE)
    with st.expander(dd.T3_EXPANDER, expanded=False):
        st.markdown(headline()["T3"])
        st.caption("D49 T3, verbatim from docs/results_review.md. No model is called best.")

    left, right = st.columns([3, 2])
    with left:
        chart(dd.recent_spec(doc, ser, m, title("Recent losses and 99% VaR")))
    with right:
        st.table(dd.health_rows(doc))
        st.caption(
            "Breaches over the last 250 trading days; zone by the Basel traffic-light rules."
        )

    st.subheader("Crash calculator")
    fall = st.number_input(
        "One-day fall (% of position)",
        min_value=0.1,
        max_value=50.0,
        value=5.0,
        step=0.5,
        key="fall",
    )
    st.table(dd.crash_rows(doc, fall / 100))
    count = dd.history_count(doc, ser, fall / 100)
    w = doc["config"]["window"]
    st.caption(
        f"{count} of the last {w:,} trading days had a fall at least this large."
        if count is not None
        else dd.BELOW_TAIL
    )
    st.caption(f"Return periods at {dd.TRADING_DAYS} trading days a year.")
    st.table(dd.return_level_rows(doc))

    st.subheader("Stress")
    k = st.slider("Volatility shock k (times today's sigma)", 1.0, 3.0, 1.0, 0.25)
    st.table(dd.stress_rows(doc, k, position))
    st.caption(dd.STRESS_NOTE)
    st.markdown("**Replay: the worst days of the series, if they happened tomorrow**")
    st.table(dd.replay_rows(doc, ser, position))
    st.caption(dd.REPLAY_NOTE)

    st.subheader("All levels")
    st.table(dd.forecast_rows(doc, position, dd.LEVELS))
    st.caption(
        "VaR ₹ = V(1 - e^-VaR), exact. ES ₹ = V(1 - e^-ES), slightly conservative (1 - e^-x is "
        "concave). The same models, config and code path as the backtest (D58)."
    )
    st.markdown("**Largest position within the risk budget**")
    st.table(dd.budget_rows(doc, budget))
    st.caption(dd.FAR_NOTE)
    st.table(dd.forecast_rows(doc, position, dd.FAR_TAIL))


# --- Track record --------------------------------------------------------------------------------
def overview() -> None:
    for col, c in zip(st.columns(4), metrics()[:4], strict=False):
        col.metric(c["label"], c["value"], help=c["help"])
    h = headline()
    st.subheader("What holds in every test pass")
    st.markdown(h["T1"])
    st.caption("D49 T1, verbatim from docs/results_review.md.")
    st.markdown(dd.T1_BORDERLINE)
    st.subheader(MAIN_FINDING)
    st.markdown(h["H"])
    st.markdown(h["T3"])
    st.caption("D46 H and D49 T3, verbatim from docs/results_review.md.")


def coverage() -> None:
    figure(svg("F2"), "F2: coverage-test verdicts at 5% per model, level and pass")
    pid, level = pass_and_level("coverage")
    st.table(coverage_rows(pid, level))
    st.caption(LABEL_NOTE)


def comparison() -> None:
    figure(svg("F3"), "F3: Diebold-Mariano statistics, families (i) and (ii)")
    labels = {label: pid for pid, label, _ in dd.PASSES}
    pid = labels[st.selectbox("Pass", list(labels), key="dm-pass")]
    st.table(dm_rows(pid))
    st.caption(LABEL_NOTE)


def track_record(figs: list[dict[str, str]] | None) -> None:
    names = ["Overview", "Coverage", "Model comparison", "Traffic light"]
    tabs = st.tabs([*names, *(["Day by day (local only)"] if figs is not None else [])])
    with tabs[0]:
        overview()
    with tabs[1]:
        coverage()
    with tabs[2]:
        comparison()
    with tabs[3]:
        figure(svg("F4"), "F4: Basel traffic light by 250-day block (plan A4)")
    if figs is not None:
        with tabs[4]:
            banner()
            for f in figs:
                figure(f["uri"], f["title"])


# --- How it works ---------------------------------------------------------------------------------
PIPELINE = (
    ("Window", r"\text{the last } W \text{ daily closes } P_t", "W"),
    ("Losses", r"L_t = -\ln(P_t / P_{t-1})", None),
    (
        "GARCH filter (F4: AR(1) mean, GJR variance)",
        r"L_t = \mu + \phi_1 L_{t-1} + \varepsilon_t,\quad \sigma_t^2 = \omega + (\alpha + "
        r"\gamma\,1\{\varepsilon_{t-1} > 0\})\,\varepsilon_{t-1}^2 + \beta\,\sigma_{t-1}^2,"
        r"\quad z_t = \varepsilon_t / \sigma_t",
        "filter",
    ),
    ("Threshold", r"u = L_{(\lceil W q \rceil)}", "threshold"),
    (
        "GPD tail",
        r"P(L > x) \approx \frac{n_u}{W}\Bigl(1 + \xi\,\frac{x - u}{\beta}\Bigr)^{-1/\xi},"
        r"\quad x \ge u",
        "gpd",
    ),
    (
        "VaR and ES",
        r"\mathrm{VaR}_c = u + \frac{\beta}{\xi}\Bigl[\Bigl(\tfrac{W}{n_u}(1 - c)\Bigr)^{-\xi}"
        r" - 1\Bigr],\quad \mathrm{ES}_c = \frac{\mathrm{VaR}_c + \beta - \xi u}{1 - \xi};"
        r"\quad \text{GARCH-EVT: } \mu_{t+1} + \sigma_{t+1}\,(\cdot)",
        "var",
    ),
    (
        "Backtest",
        r"I_{t+1} = 1\{L_{t+1} > \mathrm{VaR}_t\}\ \text{(Kupiec, Christoffersen, ES residual, "
        r"traffic light, Diebold-Mariano)}",
        None,
    ),
)


def plugged(n: dict[str, str]) -> dict[str, str]:
    return {
        "W": rf"W = {n['W']}",
        "filter": rf"\mu_{{t+1}} = {n['mu_next']},\quad \sigma_{{t+1}} = {n['sigma_next']}",
        "threshold": rf"q = {n['q']}:\quad u = {n['u']},\ n_u = {n['n_u']};\quad "
        rf"u_z = {n['u_z']},\ n_u = {n['n_u_z']}",
        "gpd": rf"\text{{POT-GPD: }} \xi = {n['xi']},\ \beta = {n['beta']};\quad "
        rf"\text{{GARCH-EVT: }} \xi = {n['xi_z']},\ \beta = {n['beta_z']}",
    }


def how_it_works(bundle: dict[str, object] | None) -> None:
    tabs = st.tabs(["Pipeline", "Threshold robustness", "Tail index", "Sensitivities"])
    with tabs[0]:
        nums = None if bundle is None else plugged(dd.pipeline_numbers(bundle["doc"]))
        for i, (name, formula, slot) in enumerate(PIPELINE, start=1):
            st.markdown(f"**{i}. {name}**")
            st.latex(formula)
            if nums is not None and slot in nums:
                st.latex(nums[slot])
        if bundle is not None:
            doc, ser = bundle["doc"], bundle["series"]
            chk = dd.self_check(doc)
            st.markdown(
                f"**Self-check** {chk['symbol']}: VaR and ES at the four levels recomputed from "
                "today's stored u, n_u, xi, beta (and mu_next, sigma_next for GARCH-EVT); "
                "the screen shows the "
                "stored values."
            )
            fall = st.session_state.get("fall", 5.0)
            chart(dd.tail_spec(doc, ser, fall / 100, mode()))
            st.caption(dd.C3_FLOOR_NOTE)
            st.caption(
                "Tail plot: observed exceedance frequency (rank / W) of the window's losses above "
                "u, the POT-GPD curve and the GARCH-EVT curve given today's volatility."
            )
    with tabs[1]:
        figure(svg("F5"), "F5: threshold sensitivity")
        figure(svg("F6"), "F6: D45 OOS-block threshold check")
    with tabs[2]:
        figure(svg("F7"), "F7: rolling xi path by block (descriptive)")
    with tabs[3]:
        level = st.selectbox("VaR level", list(dd.LEVELS), index=1, key="sens-level")
        st.table(sensitivity_rows(level))
        st.caption(LABEL_NOTE)


# --- Proof ---------------------------------------------------------------------------------
def proof(bundle: dict[str, object] | None) -> None:
    p = provenance()
    st.subheader("Commits")
    st.table([{"artefact": k, "commit": v} for k, v in p["commits"].items()])
    st.subheader("Inputs (sha256 checked against the protected-hash list)")
    st.code("\n".join(f"{v}  {k}" for k, v in p["inputs"].items()), language=None)
    st.caption("sha256sum format: verify with sha256sum -c from the repository root.")
    st.subheader("Decisions")
    st.markdown(", ".join(p["decisions"]) + " (docs/decisions.md)")
    if bundle is not None:
        st.subheader("Today's forecast")
        st.table([{"field": k, "value": str(v)} for k, v in bundle["doc"]["provenance"].items()])
        chk = bundle["check"]
        if chk is not None:
            line = CHECK_LINE.format(k=chk["identical"], n=chk["n"])
            st.markdown(f"**{line}** (commit `{chk['commit'][:12]}`).")
    st.subheader("Reproduce it yourself")
    st.markdown(dd.VERIFIED)
    st.code("\n".join(p["reproduce"]), language="bash")


def main() -> None:
    st.set_page_config(page_title="NIFTYTAIL", layout="wide")
    st.markdown(CSS.format(**SURFACE[mode()]), unsafe_allow_html=True)
    st.title("NIFTYTAIL: Nifty 50 tail risk")
    local = dd.local_mode() and not secret_public()
    figs = local_figures(not local, dd.LOCAL_RUN_DIR)
    bundle = live_bundle(not local, dd.LIVE_REPORTS) if local else None
    first = "Today" if local else dd.DEMO_SECTION  # D62 (b): the public demo, never in local mode
    sections = [first, "Track record", "How it works", "Proof"]
    tabs = dict(zip(sections, st.tabs(sections), strict=True))
    with tabs[first]:
        if local:
            today(bundle)
        else:
            today(demo_bundle(), demo=True)
    with tabs["Track record"]:
        track_record(figs)
    with tabs["How it works"]:
        how_it_works(bundle)
    with tabs["Proof"]:
        proof(bundle)


main()
