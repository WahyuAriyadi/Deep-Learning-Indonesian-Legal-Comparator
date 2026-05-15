"""
🏛️ Indonesian Legal Comparator — Hugging Face Space
=====================================================
Full dashboard app for comparing Indonesian anti-corruption law
against international UNCAC standards.

Deploy: Upload this file + requirements.txt to a Hugging Face Space
        (SDK: Gradio)
"""

import gradio as gr
import numpy as np
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download

# ─────────────────────────────────────────
# BOOTSTRAP: Download model from HF Hub
# ─────────────────────────────────────────

REPO_ID    = os.environ.get("MODEL_REPO_ID", "YOUR_USERNAME/indonesian-legal-comparator")
MODEL_PATH = Path("./model_cache")

def bootstrap_model():
    """Download model files from Hugging Face Hub on first run."""
    global model
    print(f"📥 Downloading model from {REPO_ID}...")
    try:
        path = snapshot_download(
            repo_id   = REPO_ID,
            local_dir = str(MODEL_PATH),
            ignore_patterns=["*.md", ".gitattributes"]
        )
        print(f"✅ Model downloaded to {path}")
        return path
    except Exception as e:
        print(f"⚠️  Download failed: {e}")
        print("   Falling back to local files...")
        return "."


# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────

import sys
sys.path.insert(0, str(MODEL_PATH / "src"))
sys.path.insert(0, "./src")

try:
    from legal_comparator import LegalComparator
    model_path = bootstrap_model()
    model      = LegalComparator()
    model.load(
        corpus_path    = str(Path(model_path) / "corpus_structured.json"),
        embedding_path = str(Path(model_path) / "embeddings_labse.npy")
    )
    MODEL_LOADED   = True
    DOKUMEN_LIST   = model.daftar_dokumen()
    PASAL_ID_HINTS = {
        dok: [p.id for p in model.corpus[dok][:5]]
        for dok in DOKUMEN_LIST
    }
except Exception as e:
    print(f"❌ Model load error: {e}")
    MODEL_LOADED = False
    DOKUMEN_LIST = ["UNCAC", "UU_31_1999", "UU_20_2001",
                    "UU_28_1999", "UU_30_2002", "UU_19_2019"]


# ─────────────────────────────────────────
# HELPER: Format outputs
# ─────────────────────────────────────────

def score_bar(score: float) -> str:
    filled = int(score * 20)
    empty  = 20 - filled
    return "█" * filled + "░" * empty

def status_badge(score: float) -> str:
    if score >= 0.80:   return "✅  FULLY ADOPTED"
    elif score >= 0.65: return "🟡  PARTIALLY ADOPTED"
    elif score >= 0.45: return "🟠  LOOSELY RELATED"
    else:               return "❌  GAP — NOT ADOPTED"


# ─────────────────────────────────────────
# FEATURE FUNCTIONS
# ─────────────────────────────────────────

def fn_similarity(pasal_a, pasal_b):
    if not MODEL_LOADED:
        return "❌ Model not loaded. Please check setup.", ""
    try:
        r = model.similarity_score(pasal_a.strip(), pasal_b.strip())
        score = r["score"]

        # Score display
        score_display = f"""
╔══════════════════════════════════════════╗
║         SEMANTIC SIMILARITY SCORE        ║
╠══════════════════════════════════════════╣
║  Score  : {score:.4f}  {score_bar(score)}  ║
║  Level  : {r['level'].upper():<35}║
║  Status : {status_badge(score):<35}║
╚══════════════════════════════════════════╝

Interpretation:
  {r['interpretasi']}
"""
        # Passage display
        passage_display = f"""
┌─── ARTICLE / PASAL A ─── {pasal_a} ───────────────
│ Language : {r['bahasa_a'].upper()}
│
│ {r['preview_a']}

┌─── ARTICLE / PASAL B ─── {pasal_b} ───────────────
│ Language : {r['bahasa_b'].upper()}
│
│ {r['preview_b']}
"""
        return score_display, passage_display
    except Exception as e:
        return f"❌ Error: {str(e)}\n\nTip: Check pasal ID format, e.g. UNCAC_article_15", ""


def fn_gap_analysis(doc_a, doc_b, top_n, gaya):
    if not MODEL_LOADED:
        return "❌ Model not loaded.", ""
    try:
        hasil  = model.compare(doc_a, doc_b, top_n_gap=int(top_n))
        r      = hasil.ringkasan

        # Stats panel
        total = r["total_pasal_a"]
        stats = f"""
╔══════════════════════════════════════════════════════╗
║           GAP ANALYSIS RESULTS                       ║
║  {doc_a}  ←→  {doc_b}
╠══════════════════════════════════════════════════════╣
║  Total Articles Analyzed : {total:<5}                   ║
║                                                      ║
║  ✅ Fully Adopted        : {r['diadopsi_penuh']:<5} ({r['persen_penuh']:>5}%)          ║
║  🟡 Partially Adopted    : {r['diadopsi_sebagian']:<5} ({r['persen_sebagian']:>5}%)          ║
║  ❌ GAP — Not Adopted    : {r['gap_belum_diadopsi']:<5} ({r['persen_gap']:>5}%)          ║
║                                                      ║
║  Avg Semantic Similarity : {r['rata_rata_similarity']:.4f}                    ║
╚══════════════════════════════════════════════════════╝
"""
        # Summary text
        summary = model.summarize_gap(hasil, top_n_gap=int(top_n), gaya=gaya)
        return stats, summary
    except Exception as e:
        return f"❌ Error: {str(e)}", ""


def fn_search(query, doc_filter, top_n):
    if not MODEL_LOADED:
        return "❌ Model not loaded."
    if not query.strip():
        return "⚠️  Please enter a search query."
    try:
        dok = doc_filter if doc_filter != "All Documents" else None
        hasil = model.search(query.strip(), dokumen=dok, top_n=int(top_n))

        lines = [
            f'🔎 Search Results for: "{query}"',
            f'   Filter: {doc_filter}  |  Showing top {top_n} results',
            "═" * 60,
            ""
        ]
        for h in hasil.hasil:
            bar = score_bar(h["score"])
            lines += [
                f"  #{h['rank']}  {h['dokumen']}  —  Article/Pasal {h['nomor_pasal']}",
                f"       Score : {h['score']:.4f}  {bar}",
                f"       Lang  : {h['bahasa'].upper()}",
                f"       ↳ {h['preview'][:200]}...",
                ""
            ]
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error: {str(e)}"


def fn_compare_all(doc_a, top_n):
    """Compare doc_a against ALL other documents at once."""
    if not MODEL_LOADED:
        return "❌ Model not loaded."
    try:
        others = [d for d in DOKUMEN_LIST if d != doc_a]
        lines  = [
            f"📊 MULTI-DOCUMENT COMPARISON",
            f"   Source: {doc_a}  vs  All Indonesian Laws",
            "═" * 65, ""
        ]
        for doc_b in others:
            hasil = model.compare(doc_a, doc_b, top_n_gap=3)
            r     = hasil.ringkasan
            lines += [
                f"  📋 {doc_a}  ←→  {doc_b}",
                f"     ✅ Fully    : {r['diadopsi_penuh']:>3} ({r['persen_penuh']:>5}%)",
                f"     🟡 Partial  : {r['diadopsi_sebagian']:>3} ({r['persen_sebagian']:>5}%)",
                f"     ❌ GAP      : {r['gap_belum_diadopsi']:>3} ({r['persen_gap']:>5}%)",
                f"     Avg sim    : {r['rata_rata_similarity']:.4f}",
                ""
            ]
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Error: {str(e)}"


def fn_corpus_info():
    """Show full corpus statistics."""
    if not MODEL_LOADED:
        return "❌ Model not loaded."
    lines = [
        "📚 CORPUS INFORMATION",
        "═" * 55, ""
    ]
    total_pasal = 0
    for dok in DOKUMEN_LIST:
        info = model.info_dokumen(dok)
        total_pasal += info["jumlah_pasal"]
        lines += [
            f"  📄 {dok}",
            f"     Articles/Pasal : {info['jumlah_pasal']}",
            f"     Language       : {info['bahasa'].upper()}",
            f"     Sample IDs     : {', '.join(info['id_pasal'][:3])}...",
            ""
        ]
    lines += [
        "═" * 55,
        f"  TOTAL  :  {len(DOKUMEN_LIST)} documents  |  {total_pasal} articles/pasal",
        f"  Model  :  LaBSE (Language-agnostic BERT Sentence Embeddings)",
        f"  Langs  :  EN ↔ ID (cross-lingual semantic similarity)"
    ]
    return "\n".join(lines)


# ─────────────────────────────────────────
# GRADIO UI — FULL DASHBOARD
# ─────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Fraunces:ital,wght@0,300;0,700;1,300&display=swap');

:root {
    --cream    : #F5F0E8;
    --ink      : #1A1A2E;
    --gold     : #C9A84C;
    --crimson  : #8B1A1A;
    --slate    : #2D3561;
    --muted    : #6B7280;
    --border   : #D4C5A9;
    --success  : #2D6A4F;
    --warning  : #9C6B00;
    --gap      : #8B1A1A;
}

body, .gradio-container {
    background : var(--cream) !important;
    font-family: 'DM Mono', monospace !important;
}

.main-header {
    background    : var(--ink);
    color         : var(--cream);
    padding       : 2rem 2.5rem;
    border-bottom : 4px solid var(--gold);
    margin-bottom : 1.5rem;
}

.main-header h1 {
    font-family : 'Fraunces', serif;
    font-size   : 2rem;
    font-weight : 700;
    letter-spacing: -0.02em;
    margin      : 0 0 0.25rem 0;
    color       : var(--cream);
}

.main-header p {
    font-size  : 0.8rem;
    color      : var(--gold);
    margin     : 0;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.stat-bar {
    display        : grid;
    grid-template-columns: repeat(3, 1fr);
    gap            : 1rem;
    margin-bottom  : 1.5rem;
    padding        : 0 0.5rem;
}

.stat-card {
    background    : var(--ink);
    color         : var(--cream);
    border-radius : 2px;
    padding       : 1rem 1.25rem;
    border-left   : 4px solid var(--gold);
}

.stat-card .num {
    font-family : 'Fraunces', serif;
    font-size   : 2rem;
    font-weight : 700;
    color       : var(--gold);
    line-height : 1;
}

.stat-card .label {
    font-size   : 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color       : #9CA3AF;
    margin-top  : 0.2rem;
}

.section-label {
    font-size   : 0.65rem;
    font-weight : 500;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color       : var(--muted);
    margin-bottom: 0.5rem;
}

textarea, input[type=text] {
    font-family : 'DM Mono', monospace !important;
    font-size   : 0.82rem !important;
    background  : white !important;
    border      : 1.5px solid var(--border) !important;
    border-radius: 2px !important;
    color       : var(--ink) !important;
}

textarea:focus, input:focus {
    border-color: var(--gold) !important;
    outline     : none !important;
    box-shadow  : 0 0 0 3px rgba(201,168,76,0.15) !important;
}

button.primary {
    background    : var(--ink) !important;
    color         : var(--gold) !important;
    border        : 2px solid var(--gold) !important;
    border-radius : 2px !important;
    font-family   : 'DM Mono', monospace !important;
    font-size     : 0.8rem !important;
    letter-spacing: 0.08em !important;
    font-weight   : 500 !important;
    transition    : all 0.15s ease !important;
}

button.primary:hover {
    background : var(--gold) !important;
    color      : var(--ink) !important;
}

.tab-nav button {
    font-family   : 'DM Mono', monospace !important;
    font-size     : 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color         : var(--muted) !important;
    border-bottom : 2px solid transparent !important;
}

.tab-nav button.selected {
    color        : var(--ink) !important;
    border-bottom: 2px solid var(--gold) !important;
}

.footer-note {
    text-align  : center;
    font-size   : 0.7rem;
    color       : var(--muted);
    padding     : 1.5rem;
    border-top  : 1px solid var(--border);
    margin-top  : 2rem;
    letter-spacing: 0.05em;
}

.hint-box {
    background   : white;
    border       : 1px solid var(--border);
    border-left  : 3px solid var(--gold);
    padding      : 0.75rem 1rem;
    font-size    : 0.75rem;
    color        : var(--muted);
    margin-bottom: 0.75rem;
    border-radius: 0 2px 2px 0;
}
"""

with gr.Blocks(css=CSS, title="🏛️ Indonesian Legal Comparator") as demo:

    # ── Header ──────────────────────────────────────────
    gr.HTML("""
    <div class="main-header">
        <h1>🏛️ Indonesian Legal Comparator</h1>
        <p>NLP model · Anti-corruption law · UNCAC vs Indonesian statutes · Cross-lingual EN ↔ ID</p>
    </div>
    """)

    # ── Stat bar ─────────────────────────────────────────
    n_docs  = len(DOKUMEN_LIST)
    n_pasal = len(model.semua_pasal) if MODEL_LOADED else "—"
    gr.HTML(f"""
    <div class="stat-bar">
        <div class="stat-card">
            <div class="num">{n_docs}</div>
            <div class="label">Legal Documents</div>
        </div>
        <div class="stat-card">
            <div class="num">{n_pasal}</div>
            <div class="label">Articles / Pasal</div>
        </div>
        <div class="stat-card">
            <div class="num">2</div>
            <div class="label">Languages (EN · ID)</div>
        </div>
    </div>
    """)

    # ── Tabs ─────────────────────────────────────────────
    with gr.Tabs():

        # ════════════════════════════════════════
        # TAB 1 — SIMILARITY SCORE
        # ════════════════════════════════════════
        with gr.Tab("📐  Similarity Score"):
            gr.HTML('<div class="hint-box">Compare any two specific articles/pasal by their ID. Supports cross-lingual comparison (EN ↔ ID). Example: <code>UNCAC_article_15</code> vs <code>UU_31_1999_pasal_5</code></div>')

            with gr.Row():
                with gr.Column(scale=1):
                    inp_a   = gr.Textbox(
                        label       = "Article / Pasal ID  —  Document A",
                        placeholder = "UNCAC_article_15",
                        info        = "Format: {DOCUMENT}_{article/pasal}_{NUMBER}"
                    )
                    inp_b   = gr.Textbox(
                        label       = "Article / Pasal ID  —  Document B",
                        placeholder = "UU_31_1999_pasal_5",
                    )
                    btn_sim = gr.Button("⚖️  Calculate Similarity", variant="primary")

                with gr.Column(scale=2):
                    out_score   = gr.Textbox(
                        label = "Similarity Score",
                        lines = 10,
                    )
                    out_preview = gr.Textbox(
                        label = "Article Preview",
                        lines = 10,
                    )

            btn_sim.click(
                fn_similarity,
                inputs  = [inp_a, inp_b],
                outputs = [out_score, out_preview]
            )

            # Quick examples
            gr.HTML('<div class="section-label" style="margin-top:1rem">Quick Examples</div>')
            with gr.Row():
                gr.Examples(
                    examples = [
                        ["UNCAC_article_15", "UU_31_1999_pasal_5"],
                        ["UNCAC_article_17", "UU_20_2001_pasal_8"],
                        ["UNCAC_article_20", "UU_31_1999_pasal_2"],
                        ["UNCAC_article_6",  "UU_30_2002_pasal_3"],
                        ["UNCAC_article_51", "UU_20_2001_pasal_18"],
                    ],
                    inputs   = [inp_a, inp_b],
                    label    = "Click to load example pair"
                )

        # ════════════════════════════════════════
        # TAB 2 — GAP ANALYSIS
        # ════════════════════════════════════════
        with gr.Tab("📊  Gap Analysis"):
            gr.HTML('<div class="hint-box">Select two documents to run a full semantic gap analysis — identifying which provisions of Document A have been adopted, partially adopted, or are missing (GAP) in Document B.</div>')

            with gr.Row():
                with gr.Column(scale=1):
                    sel_a   = gr.Dropdown(
                        DOKUMEN_LIST,
                        label = "Document A  (Reference)",
                        value = "UNCAC"
                    )
                    sel_b   = gr.Dropdown(
                        DOKUMEN_LIST,
                        label = "Document B  (Comparison Target)",
                        value = "UU_31_1999"
                    )
                    n_gap   = gr.Slider(
                        3, 20, value=5, step=1,
                        label = "Top N gaps to display"
                    )
                    gaya    = gr.Radio(
                        ["naratif", "poin", "tabel"],
                        value = "naratif",
                        label = "Summary format"
                    )
                    btn_gap = gr.Button("📊  Run Gap Analysis", variant="primary")

                with gr.Column(scale=2):
                    out_stats   = gr.Textbox(
                        label = "Statistics",
                        lines = 12,
                    )
                    out_summary = gr.Textbox(
                        label = "Detailed Summary",
                        lines = 22,
                    )

            btn_gap.click(
                fn_gap_analysis,
                inputs  = [sel_a, sel_b, n_gap, gaya],
                outputs = [out_stats, out_summary]
            )

        # ════════════════════════════════════════
        # TAB 3 — SEMANTIC SEARCH
        # ════════════════════════════════════════
        with gr.Tab("🔎  Search Pasal"):
            gr.HTML('<div class="hint-box">Search for relevant articles using natural language — in Indonesian or English. The model retrieves semantically similar provisions across all documents.</div>')

            with gr.Row():
                with gr.Column(scale=2):
                    inp_q    = gr.Textbox(
                        label       = "Search Query  (Indonesian or English)",
                        placeholder = "e.g.  suap pejabat negara  /  bribery of public official  /  asset recovery",
                        lines       = 2
                    )
                with gr.Column(scale=1):
                    sel_dok  = gr.Dropdown(
                        ["All Documents"] + DOKUMEN_LIST,
                        label = "Filter by Document",
                        value = "All Documents"
                    )
                    n_res    = gr.Slider(1, 10, value=5, step=1, label="Results to return")
                    btn_src  = gr.Button("🔎  Search", variant="primary")

            out_src = gr.Textbox(
                label = "Search Results",
                lines = 25,
            )
            btn_src.click(
                fn_search,
                inputs  = [inp_q, sel_dok, n_res],
                outputs = out_src
            )

            gr.HTML('<div class="section-label" style="margin-top:1rem">Quick Search Examples</div>')
            gr.Examples(
                examples = [
                    ["suap kepada pejabat negara",              "All Documents"],
                    ["bribery of foreign public officials",     "UNCAC"],
                    ["pemulihan aset hasil tindak pidana",      "All Documents"],
                    ["money laundering financial institutions", "UNCAC"],
                    ["whistleblower protection reporting",      "All Documents"],
                    ["gratifikasi penyelenggara negara",        "UU_20_2001"],
                    ["illicit enrichment unexplained wealth",   "UNCAC"],
                    ["korupsi sektor swasta perusahaan",        "All Documents"],
                ],
                inputs = [inp_q, sel_dok],
                label  = "Click to load example query"
            )

        # ════════════════════════════════════════
        # TAB 4 — MULTI-DOCUMENT COMPARE
        # ════════════════════════════════════════
        with gr.Tab("🗺️  Multi-Document"):
            gr.HTML('<div class="hint-box">Compare one reference document against ALL other documents simultaneously. Useful for a bird\'s-eye view of how comprehensively a law has been adopted across the entire corpus.</div>')

            with gr.Row():
                with gr.Column(scale=1):
                    sel_ref  = gr.Dropdown(
                        DOKUMEN_LIST,
                        label = "Reference Document",
                        value = "UNCAC"
                    )
                    btn_all  = gr.Button("🗺️  Compare Against All", variant="primary")
                    gr.HTML("""
                    <div style="margin-top:1.5rem; padding:1rem;
                         background:white; border:1px solid #D4C5A9;
                         font-size:0.75rem; color:#6B7280; line-height:1.8">
                        <strong>How to read results:</strong><br>
                        ✅ Fully Adopted  → score ≥ 0.80<br>
                        🟡 Partial        → score ≥ 0.65<br>
                        ❌ GAP            → score &lt; 0.65<br><br>
                        Higher average similarity = more comprehensive adoption
                    </div>
                    """)
                with gr.Column(scale=2):
                    out_all = gr.Textbox(
                        label = "Multi-Document Comparison",
                        lines = 30,
                    )

            btn_all.click(fn_compare_all, inputs=[sel_ref], outputs=out_all)

        # ════════════════════════════════════════
        # TAB 5 — CORPUS INFO
        # ════════════════════════════════════════
        with gr.Tab("📚  Corpus Info"):
            gr.HTML('<div class="hint-box">Full details about all documents loaded in the corpus, including article counts, languages, and sample IDs you can use in other tabs.</div>')

            btn_info = gr.Button("📚  Load Corpus Info", variant="primary")
            out_info = gr.Textbox(
                label = "Corpus Details",
                lines = 30,
            )
            btn_info.click(fn_corpus_info, inputs=[], outputs=out_info)

            gr.HTML("""
            <div style="margin-top:1.5rem">
            <table style="width:100%; font-size:0.75rem;
                          border-collapse:collapse; font-family:'DM Mono',monospace">
              <thead>
                <tr style="background:#1A1A2E; color:#F5F0E8">
                  <th style="padding:0.6rem 1rem; text-align:left">Label</th>
                  <th style="padding:0.6rem 1rem; text-align:left">Document</th>
                  <th style="padding:0.6rem 1rem; text-align:left">Year</th>
                  <th style="padding:0.6rem 1rem; text-align:left">Lang</th>
                </tr>
              </thead>
              <tbody>
                <tr style="border-bottom:1px solid #D4C5A9">
                  <td style="padding:0.5rem 1rem"><code>UNCAC</code></td>
                  <td style="padding:0.5rem 1rem">UN Convention Against Corruption</td>
                  <td style="padding:0.5rem 1rem">2003</td>
                  <td style="padding:0.5rem 1rem">EN</td>
                </tr>
                <tr style="border-bottom:1px solid #D4C5A9; background:#faf8f3">
                  <td style="padding:0.5rem 1rem"><code>UU_7_2006</code></td>
                  <td style="padding:0.5rem 1rem">UU Ratifikasi UNCAC</td>
                  <td style="padding:0.5rem 1rem">2006</td>
                  <td style="padding:0.5rem 1rem">ID</td>
                </tr>
                <tr style="border-bottom:1px solid #D4C5A9">
                  <td style="padding:0.5rem 1rem"><code>UU_31_1999</code></td>
                  <td style="padding:0.5rem 1rem">UU Tipikor</td>
                  <td style="padding:0.5rem 1rem">1999</td>
                  <td style="padding:0.5rem 1rem">ID</td>
                </tr>
                <tr style="border-bottom:1px solid #D4C5A9; background:#faf8f3">
                  <td style="padding:0.5rem 1rem"><code>UU_20_2001</code></td>
                  <td style="padding:0.5rem 1rem">UU Tipikor Amendment</td>
                  <td style="padding:0.5rem 1rem">2001</td>
                  <td style="padding:0.5rem 1rem">ID</td>
                </tr>
                <tr style="border-bottom:1px solid #D4C5A9">
                  <td style="padding:0.5rem 1rem"><code>UU_28_1999</code></td>
                  <td style="padding:0.5rem 1rem">UU Penyelenggaraan Negara Bersih KKN</td>
                  <td style="padding:0.5rem 1rem">1999</td>
                  <td style="padding:0.5rem 1rem">ID</td>
                </tr>
                <tr style="border-bottom:1px solid #D4C5A9; background:#faf8f3">
                  <td style="padding:0.5rem 1rem"><code>UU_30_2002</code></td>
                  <td style="padding:0.5rem 1rem">UU KPK (Komisi Pemberantasan Korupsi)</td>
                  <td style="padding:0.5rem 1rem">2002</td>
                  <td style="padding:0.5rem 1em">ID</td>
                </tr>
                <tr>
                  <td style="padding:0.5rem 1rem"><code>UU_19_2019</code></td>
                  <td style="padding:0.5rem 1rem">UU KPK Revision</td>
                  <td style="padding:0.5rem 1rem">2019</td>
                  <td style="padding:0.5rem 1rem">ID</td>
                </tr>
              </tbody>
            </table>
            </div>
            """)

    # ── Footer ───────────────────────────────────────────
    gr.HTML("""
    <div class="footer-note">
        🏛️ Indonesian Legal Comparator  ·
        Model: LaBSE + KeyBERT  ·
        Data: UNCAC (UNODC) + JDIH Indonesia  ·
        License: MIT  ·
        Built for AI Trainer — Indonesian Speaker portfolio
    </div>
    """)


if __name__ == "__main__":
    demo.launch()
