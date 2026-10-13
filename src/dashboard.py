"""
Module 5+6: Live Dashboard with Review-Flag Logic

FastAPI backend serving:
  - GAT predictions per region
  - Explainability (attention) overlay
  - Automatic review flag when a region's predicted score falls in the
    lowest percentile of this run's own distribution

Run with: uvicorn dashboard:app --reload
Then open the printed local URL, or in Colab use ngrok/pyngrok to expose it.

Updated to use the time-aggregated feature columns from graph_construction.py
(temp_mean, ... ) instead of the old single-snapshot columns (temp, ...).
"""

import torch
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from graph_construction import build_graph, FEATURE_COLUMNS
from gat_model import MarinaGAT
from explainability import extract_attention, explain_top_edges

app = FastAPI(
    title="Marina Biodiversity Monitor - India East Coast"
)

# ---- Config ----
REVIEW_PERCENTILE = 20

# ---- Load graph + model once at startup ----
graph = build_graph("ocean_physics.nc")

model = MarinaGAT(
    in_channels=graph.x.shape[1] - 1
)

model.load_state_dict(
    torch.load("marina_gat.pt")
)

model.eval()

node_df = pd.DataFrame(
    graph.x.numpy(),
    columns=FEATURE_COLUMNS
)


def run_prediction_and_flags():
    with torch.no_grad():
        preds, _ = model(
            graph.x,
            graph.edge_index,
            return_attention=True
        )

    preds = preds.squeeze().numpy()

    results = graph.coords.copy()
    results["predicted_score"] = preds
    results["temp_mean"] = node_df["temp_mean"].values

    # Drop land / no-data cells
    results = results[
        results["temp_mean"] != 0.0
    ].reset_index(drop=True)

    # Flag the lowest-scoring regions
    cutoff = np.percentile(
        results["predicted_score"],
        REVIEW_PERCENTILE
    )

    results["flagged_for_review"] = (
        results["predicted_score"] <= cutoff
    )

    return results


@app.get("/api/predictions")
def get_predictions():
    """Returns per-region prediction and review flag."""

    results = run_prediction_and_flags()

    return {
        "records": results.to_dict(
            orient="records"
        )
    }


@app.get("/api/flagged-for-review")
def get_flagged_for_review():
    """Returns regions in the bottom REVIEW_PERCENTILE
    of predicted scores.
    """

    results = run_prediction_and_flags()

    flagged = results[
        results["flagged_for_review"]
    ]

    return {
        "num_zones_flagged": len(flagged),
        "zones": flagged.to_dict(
            orient="records"
        ),
    }


@app.get("/api/explainability")
def get_explainability(top_k: int = 10):
    """Returns top attention-weighted region connections."""

    attn_df, _ = extract_attention(
        model,
        graph
    )

    explained = explain_top_edges(
        attn_df,
        graph,
        node_df,
        top_k=top_k
    )

    return explained.to_dict(
        orient="records"
    )


@app.get("/", response_class=HTMLResponse)
def dashboard():
    """Dashboard map."""

    results = run_prediction_and_flags()

    markers_js = ""

    for _, row in results.iterrows():

        color = (
            "red"
            if row["flagged_for_review"]
            else "green"
        )

        markers_js += (
            f'L.circleMarker('
            f'[{row.lat}, {row.lon}], '
            f'{{radius:8, color:"{color}", fillOpacity:0.7}}'
            f')'
            f'.addTo(map)'
            f'.bindPopup('
            f'"Score: {row.predicted_score:.3f}<br>'
            f'Flagged for review: '
            f'{row.flagged_for_review}"'
            f');\n'
        )

    center_lat = results["lat"].mean()
    center_lon = results["lon"].mean()

    html = f"""
    <html>

    <head>

        <title>
            Marina Biodiversity Monitor -
            India East Coast
        </title>

        <link
            rel="stylesheet"
            href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
        />

        <script
            src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js">
        </script>

        <style>

            #map {{
                height: 85vh;
            }}

            body {{
                font-family: sans-serif;
                margin: 0;
            }}

        </style>

    </head>

    <body>

        <h2 style="margin:10px 10px 0 10px;">
            Marina Biodiversity Monitor -
            India East Coast
        </h2>

        <p style="margin:0 10px;">
            Red = flagged for expert review
            (bottom {REVIEW_PERCENTILE}% of predicted score)
            | Green = not flagged
        </p>

        <div id="map"></div>

        <script>

            var map = L.map('map').setView(
                [{center_lat}, {center_lon}],
                6
            );

            L.tileLayer(
                'https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png'
            ).addTo(map);

            {markers_js}

        </script>

    </body>

    </html>
    """

    return html


if __name__ == "__main__":

    import asyncio
    import uvicorn

    def _running_in_notebook_loop() -> bool:

        try:
            asyncio.get_running_loop()
            return True

        except RuntimeError:
            return False

    if _running_in_notebook_loop():

        import nest_asyncio

        nest_asyncio.apply()

        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=8000
        )

        server = uvicorn.Server(config)

        asyncio.get_event_loop().run_until_complete(
            server.serve()
        )

    else:

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000
        )
