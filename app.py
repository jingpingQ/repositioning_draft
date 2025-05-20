# app.py  –  prettier UI with a side-panel for verbose details
# ------------------------------------------------------------
import asyncio, pandas as pd, gradio as gr
from fetchers import evaluate_candidate


# ------------------------------------------------------------------
# 1 · async core
# ------------------------------------------------------------------
async def _run(drugs: str, disease: str) -> pd.DataFrame:
    names = [d.strip() for d in drugs.split(",") if d.strip()]
    if not names:
        return pd.DataFrame()

    rows = await asyncio.gather(*(evaluate_candidate(n, disease) for n in names))
    return pd.DataFrame(rows).sort_values("overall_score", ascending=False)


# ------------------------------------------------------------------
# 2 · formatting helpers
# ------------------------------------------------------------------
DISPLAY_COLS = {
    "drug":                       "Drug",
    "overall_score":              "Score",
    "target_expression_overlap":  "Target Expr",
    "moa_disease_alignment":      "MoA",
    "trial_failure_reason":       "Trial Fail",
    "safety_tolerability":        "Safety",
    "literature_support":         "Literature",
    "docking_potential":          "Docking",
    "pathway_involvement":        "Pathway",
}

def make_pretty(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    num = df.select_dtypes("number").columns
    df[num] = df[num].round(3)
    return df.rename(columns=DISPLAY_COLS)[DISPLAY_COLS.values()]

# keep verbose text in parallel dict for easy lookup
def make_details_map(df: pd.DataFrame) -> dict[str, str]:
    return dict(zip(df["drug"], df["details"])) if not df.empty else {}

# ------------------------------------------------------------------
# 3 · Gradio UI
# ------------------------------------------------------------------
with gr.Blocks(title="Phase-3 Failure → Repositioning") as demo:
    gr.Markdown(
        "### Drug Repositioning Scorer  \n"
        "Comma-separate ***Phase-3-failed*** drugs; choose a cancer subtype."
    )

    with gr.Row():
        in_drugs   = gr.Textbox(label="Drugs", lines=1,
                                placeholder="rilonacept, teplizumab")
        in_disease = gr.Textbox(label="Target cancer subtype", lines=1,
                                placeholder="triple-negative breast cancer")
        run        = gr.Button("Run pipeline")

    with gr.Row():
        grid   = gr.Dataframe(type="pandas",
                              wrap=True,
                              interactive=False,
                              column_widths=[110,80,96,70,80,70,96,70,80])
        detail = gr.Markdown("← click a row to see full component scores")

    # state for mapping Drug → verbose text
    store = gr.State(value={})

    # main pipeline
    async def launch(drugs, disease):
        df = await _run(drugs, disease)
        return make_pretty(df), make_details_map(df)

    run.click(launch, inputs=[in_drugs, in_disease], outputs=[grid, store])

    # row-click handler
    def show_detail(evt: gr.SelectData, df: pd.DataFrame, mapping: dict):
        drug = df.iloc[evt.index[0]]["Drug"]
        txt  = mapping.get(drug, "")
        # convert the newline-separated list into Markdown bullets
        md   = "#### " + drug + "  \n" + "\n".join(f"* {l}" for l in txt.splitlines())
        return md

    grid.select(show_detail, inputs=[grid, store], outputs=detail)

# ------------------------------------------------------------------
if __name__ == "__main__":
    demo.queue().launch()
