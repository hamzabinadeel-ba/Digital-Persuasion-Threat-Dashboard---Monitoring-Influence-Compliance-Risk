
import base64
import io
import re

import numpy as np
import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

from dash import Dash, dcc, html, Input, Output, callback_context
import dash_bootstrap_components as dbc
import plotly.express as px

# =========================================================
# 1. LOAD DATA + RECREATE EDA FEATURES
# =========================================================

df = pd.read_excel("phishing_cleaned.xlsx")

if "message_text" not in df.columns or "message_label" not in df.columns:
    raise ValueError("Expected columns 'message_text' and 'message_label'.")

df["message_text"] = df["message_text"].astype(str)
df["message_label"] = df["message_label"].astype(str)

# Rename NOT-malicious general class / NOT-Malicious General Class -> Non-malicious
df["message_label"] = df["message_label"].replace(
    {
        "NOT-malicious general class": "Non-malicious",
        "NOT-Malicious General Class": "Non-malicious",
    }
)

# 1.1 Structural feature: word length
df["length_words"] = df["message_text"].str.split().str.len()

# 1.2 Urgency features – same logic as your EDA
urgency_terms = [
    "urgent",
    "immediately",
    "immediate",
    "now",
    "asap",
    "action required",
    "attention",
    "important",
    "respond now",
    "last chance",
    "final notice",
    "warning",
    "verify",
    "verification",
    "alert",
]
pattern = r"(" + "|".join([re.escape(term) for term in urgency_terms]) + r")"
df["urgency_count"] = df["message_text"].str.lower().str.count(pattern)
df["is_urgent"] = np.where(df["urgency_count"] > 0, 1, 0)

# 1.3 Clean text for lexical analysis
def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["message_text"].apply(preprocess_text)

LABELS = sorted(df["message_label"].unique())
DEFAULT_LABEL = "Phishing" if "Phishing" in LABELS else LABELS[0]

# Pre-compute aggregates for top visuals
length_stats = (
    df.groupby("message_label")["length_words"]
    .mean()
    .reset_index(name="mean_words")
)
length_stats = length_stats.sort_values("mean_words", ascending=True)

urgency_stats = (
    df.groupby("message_label")
    .agg(total=("message_text", "count"), urgent=("is_urgent", "sum"))
    .reset_index()
)
urgency_stats["prop_urgent"] = urgency_stats["urgent"] / urgency_stats["total"]
urgency_stats = urgency_stats.sort_values("prop_urgent", ascending=False)

# =========================================================
# 2. COLOUR SCHEME & APP LAYOUT – DARK EXECUTIVE THEME
# =========================================================

DARK_BG = "#171F31"
CARD_BG = "#2c3243"
TEXT_MAIN = "#f5f5f5"
TEXT_MUTED = "#9da5b4"

# Category colour map (consistent across charts & metric cards)
LABEL_COLOR_MAP = {
    "Phishing": "#ff5b5b",       # red
    "Scareware": "#ffb74d",      # orange
    "Malware": "#00bcd4",        # cyan/teal
    "Pretexting": "#ba68c8",     # purple
    "Baiting": "#81c784",        # green
    "Non-malicious": "#90a4ae",  # grey-blue
}

DEFAULT_OTHER_COLOR = "#00bcd4"

LABEL_CMAP = {
    "Phishing": "Reds",
    "Scareware": "Oranges",
    "Malware": "YlGnBu",
    "Pretexting": "Purples",
    "Baiting": "Greens",
    "Non-malicious": "Greys",
}

external_stylesheets = [dbc.themes.SUPERHERO]

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Phishing Threat Signature"

app.layout = html.Div(
    style={"backgroundColor": DARK_BG, "minHeight": "100vh", "padding": "20px"},
    children=[
        # Dashboard title (centred)
        html.Div(
            [
                html.H3(
                    "PHISHING THREAT SIGNATURE DASHBOARD",
                    style={
                        "color": TEXT_MAIN,
                        "fontWeight": "700",
                        "textAlign": "center",
                        "fontSize": "28px",
                    },
                ),
                html.Hr(style={"borderColor": "#ff6b6b"}),
            ]
        ),

        # FILTER ROW
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label(
                            "Threat Category Filter",
                            style={
                                "color": TEXT_MAIN,
                                "fontWeight": "600",
                                "marginBottom": "4px",
                                "fontSize": "16px",
                            },
                        ),
                        dcc.Dropdown(
                            id="label-filter",
                            options=[
                                {"label": lab, "value": lab} for lab in LABELS
                            ],
                            value=LABELS,  # default: all selected
                            multi=True,
                            style={
                                "color": "#000000",
                            },
                        ),
                    ],
                    md=6,
                ),
            ],
            style={"marginBottom": "20px"},
        ),

        # TOP ROW: V1 + V2
        dbc.Row(
            [
                # V1 – Structural risk & trust impact
                dbc.Col(
                    dbc.Card(
                        style={
                            "backgroundColor": CARD_BG,
                            "borderRadius": "12px",
                            "border": "1px solid #222a3b",
                        },
                        children=[
                            dbc.CardHeader(
                                "Structural Risk and Trust Impact: Message Length Comparison",
                                style={
                                    "backgroundColor": CARD_BG,
                                    "color": TEXT_MAIN,
                                    "fontWeight": "800",
                                    "borderBottom": "none",
                                    "fontSize": "20px",
                                },
                            ),
                            dbc.CardBody(
                                dcc.Graph(id="length-bar"),
                                style={"padding": "10px"},
                            ),
                        ],
                    ),
                    md=6,
                ),

                # V2 – Primary susceptibility vector
                dbc.Col(
                    dbc.Card(
                        style={
                            "backgroundColor": CARD_BG,
                            "borderRadius": "12px",
                            "border": "1px solid #222a3b",
                        },
                        children=[
                            dbc.CardHeader(
                                "Primary Susceptibility Vector: Urgency Cue Frequency",
                                style={
                                    "backgroundColor": CARD_BG,
                                    "color": TEXT_MAIN,
                                    "fontWeight": "800",
                                    "borderBottom": "none",
                                    "fontSize": "20px",
                                },
                            ),
                            dbc.CardBody(
                                dcc.Graph(id="urgency-bar"),
                                style={"padding": "10px"},
                            ),
                        ],
                    ),
                    md=6,
                ),
            ],
            style={"marginBottom": "20px"},
        ),

        # BOTTOM ROW: V3 – lexical
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        style={
                            "backgroundColor": CARD_BG,
                            "borderRadius": "12px",
                            "border": "1px solid #222a3b",
                        },
                        children=[
                            dbc.CardHeader(
                                [
                                    html.Div(
                                        "Ethical Risk Assessment: Attack Narrative Lexicon",
                                        style={
                                            "color": TEXT_MAIN,
                                            "fontWeight": "800",
                                            "fontSize": "20px",
                                        },
                                    ),
                                    html.Div(
                                        id="lexical-focus",
                                        style={
                                            "color": TEXT_MUTED,
                                            "fontSize": "12px",
                                            "marginTop": "4px",
                                        },
                                    ),
                                ],
                                style={
                                    "backgroundColor": CARD_BG,
                                    "borderBottom": "none",
                                },
                            ),
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            # Wordcloud on the left with heading + border
                                            dbc.Col(
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            "Unigram Word Cloud",
                                                            style={
                                                                "color": TEXT_MAIN,
                                                                "fontWeight": "600",
                                                                "marginBottom": "6px",
                                                                "fontSize": "20px",
                                                            },
                                                        ),
                                                        html.Div(
                                                            html.Img(
                                                                id="wordcloud-img",
                                                                style={
                                                                    "maxWidth": "100%",
                                                                    "height": "350px",
                                                                    "objectFit": "contain",
                                                                },
                                                            ),
                                                            style={
                                                                "display": "flex",
                                                                "justifyContent": "center",
                                                                "alignItems": "center",
                                                                "border": "1px solid #222a3b",
                                                                "borderRadius": "10px",
                                                                "padding": "10px",
                                                            },
                                                        ),
                                                    ]
                                                ),
                                                md=8,
                                            ),
                                            # Bigram metric cards on the right with heading + border
                                            dbc.Col(
                                                html.Div(
                                                    [
                                                        html.Div(
                                                            "Bigram Metric Card",
                                                            style={
                                                                "color": TEXT_MAIN,
                                                                "fontWeight": "600",
                                                                "marginBottom": "6px",
                                                                "fontSize": "20px",
                                                            },
                                                        ),
                                                        html.Div(
                                                            id="bigram-cards",
                                                            style={
                                                                "border": "1px solid #222a3b",
                                                                "borderRadius": "10px",
                                                                "padding": "10px",
                                                            },
                                                        ),
                                                    ]
                                                ),
                                                md=4,
                                            ),
                                        ]
                                    )
                                ]
                            ),
                        ],
                    ),
                    md=12,
                )
            ]
        ),
    ],
)

# =========================================================
# 3. STATIC FIGURES (V1 & V2)
# =========================================================

def make_length_figure(selected_labels):
    data = length_stats.copy()
    if selected_labels:
        data = data[data["message_label"].isin(selected_labels)]

    if data.empty:
        fig = px.scatter()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor= CARD_BG,
            plot_bgcolor= CARD_BG,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                dict(
                    text="No data for selected filters",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(color=TEXT_MAIN),
                )
            ],
        )
        return fig

    fig = px.bar(
        data,
        x="mean_words",
        y="message_label",
        orientation="h",
        color="message_label",
        color_discrete_map=LABEL_COLOR_MAP,
        text=data["mean_words"].round(2),
        labels={"message_label": "Legend"},   # << rename legend title
    )

    fig.update_traces(
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Mean words: %{x:.2f}<extra></extra>",
    )

    # Prominent callout for phishing vs Non-malicious – only if Phishing visible
    if "Phishing" in data["message_label"].values:
        phishing_mean = float(
            data.loc[data["message_label"] == "Phishing", "mean_words"].iloc[0]
        )
        callout_text = "A Phishing message is 3.4× longer than a Non-malicious message"

        fig.add_annotation(
            x=phishing_mean * 0.7,   # a bit to the right of centre of bar
            y="Phishing",
            yshift=200,               # above the bar
            text=callout_text,
            showarrow=False,
            font=dict(color=TEXT_MAIN, size=15, family="Segoe UI"),
            align="center",
            bgcolor="#0d4b9b",
            bordercolor="#fefefd",
            borderpad=6,
            borderwidth=1,
            opacity=0.95,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_MAIN, family="Segoe UI", size=12),
        margin=dict(l=90, r=40, t=30, b=30),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=TEXT_MAIN),
        ),
    )
    fig.update_xaxes(title_text="Mean Word Count", gridcolor="#252b3a")
    fig.update_yaxes(title_text="Threat Category", gridcolor="#252b3a")

    return fig


def make_urgency_figure(selected_labels):
    data = urgency_stats.copy()
    if selected_labels:
        data = data[data["message_label"].isin(selected_labels)]

    if data.empty:
        fig = px.scatter()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor= CARD_BG,
            plot_bgcolor= CARD_BG,
            xaxis={"visible": False},
            yaxis={"visible": False},
            annotations=[
                dict(
                    text="No data for selected filters",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(color=TEXT_MAIN),
                )
            ],
        )
        return fig

    fig = px.bar(
        data,
        x="message_label",
        y="prop_urgent",
        color="message_label",
        color_discrete_map=LABEL_COLOR_MAP,
        text=(data["prop_urgent"] * 100).round(2).astype(str) + "%",
        labels={"message_label": "Legend"},   # << rename legend title
    )
    fig.update_traces(
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Urgent messages: %{y:.2%}<extra></extra>",
    )

    # Prominent annotation for phishing – only if Phishing visible
    if "Phishing" in data["message_label"].values:
        phishing_prop = float(
            data.loc[data["message_label"] == "Phishing", "prop_urgent"].iloc[0]
        )
        note = (
            f"Phishing is the most frequent ({phishing_prop * 100:.2f}%) "
            "and uses the most diverse urgency terms"
        )
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.35,   # inside plot area to the right
            y=0.80,
            xanchor="left",
            text=note,
            showarrow=False,
            font=dict(color=TEXT_MAIN, size=15, family="Segoe UI"),
            align="left",
            bgcolor="#0d4b9b",
            bordercolor="#fefefd",
            borderpad=6,
            borderwidth=1,
            opacity=0.95,
        )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=CARD_BG,
        plot_bgcolor=CARD_BG,
        font=dict(color=TEXT_MAIN, family="Segoe UI", size=12),
        margin=dict(l=60, r=80, t=40, b=40),
        yaxis_tickformat=".0%",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color=TEXT_MAIN),
        ),
    )
    fig.update_xaxes(title_text="Threat Category", gridcolor="#252b3a")
    fig.update_yaxes(title_text="% Messages with Urgency", gridcolor="#252b3a")

    return fig


# Top charts now depend on the filter
@app.callback(
    Output("length-bar", "figure"),
    Input("label-filter", "value"),
)
def render_length(selected_labels):
    return make_length_figure(selected_labels)


@app.callback(
    Output("urgency-bar", "figure"),
    Input("label-filter", "value"),
)
def render_urgency(selected_labels):
    return make_urgency_figure(selected_labels)


# =========================================================
# 4. WORD CLOUD + BIGRAM METRIC CARDS (V3)
# =========================================================


@app.callback(
    [
        Output("wordcloud-img", "src"),
        Output("bigram-cards", "children"),
        Output("lexical-focus", "children"),
    ],
    [
        Input("length-bar", "clickData"),
        Input("urgency-bar", "clickData"),
        Input("label-filter", "value"),
    ],
)
def update_lexical_view(length_click, urgency_click, selected_labels):
    ctx = callback_context
    available = selected_labels or LABELS

    # default label respects filter
    label = DEFAULT_LABEL
    if label not in available:
        label = available[0]

    # If user clicked on bar, override with clicked label (if still visible)
    if ctx.triggered:
        trigger = ctx.triggered[0]["prop_id"].split(".")[0]
        proposed = None
        if trigger == "length-bar" and length_click:
            proposed = length_click["points"][0]["y"]
        elif trigger == "urgency-bar" and urgency_click:
            proposed = urgency_click["points"][0]["x"]
        if proposed in available:
            label = proposed

    subset = df[df["message_label"].isin(available)]
    subset = subset[subset["message_label"] == label]

    if subset.empty:
        return "", [], f"Currently focused on: {label} (no messages available for current filter)"

    all_text = " ".join(subset["clean_text"].tolist())

    cmap = LABEL_CMAP.get(label, "Blues")
    wc = WordCloud(
        width=800,
        height=400,
        background_color=None,
        mode="RGBA",
        colormap=cmap,
        max_words=100,
    ).generate(all_text)

    img = wc.to_image()
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode()
    src = f"data:image/png;base64,{img_b64}"

    # Top bigrams
    vectorizer_bi = CountVectorizer(stop_words="english", ngram_range=(2, 2))
    X_bi = vectorizer_bi.fit_transform(subset["clean_text"])
    counts = X_bi.sum(axis=0).A1
    vocab = vectorizer_bi.get_feature_names_out()

    bigram_df = (
        pd.DataFrame({"bigram": vocab, "count": counts})
        .sort_values("count", ascending=False)
        .head(3)
    )

    cards = []
    border_color = LABEL_COLOR_MAP.get(label, DEFAULT_OTHER_COLOR)

    if bigram_df.empty:
        cards.append(
            html.Div(
                "No frequent bigram phrases identified.",
                style={"color": TEXT_MUTED, "fontSize": "12px"},
            )
        )
    else:
        for _, row in bigram_df.iterrows():
            bigram = row["bigram"]
            count = int(row["count"])
            cards.append(
                dbc.Card(
                    style={
                        "border": f"2px solid {border_color}",
                        "borderRadius": "10px",
                        "backgroundColor": CARD_BG,
                        "marginBottom": "10px",
                    },
                    children=dbc.CardBody(
                        html.Div(
                            [
                                # Bigram on the left
                                html.Span(
                                    bigram,
                                    style={
                                        "color": TEXT_MAIN,
                                        "fontWeight": "600",
                                        "fontSize": "18px",
                                    },
                                ),
                                # Count on the right
                                html.Span(
                                    f"{count} times",
                                    style={
                                        "color": TEXT_MUTED,
                                        "fontSize": "18px",
                                    },
                                ),
                            ],
                            style={
                                "display": "flex",
                                "justifyContent": "space-between",
                                "alignItems": "center",
                                "width": "100%",
                            },
                        )
                    ),
                )
            )

    focus_text = (
        f"Currently focused on: {label}. Click any bar above or adjust filters to change the focus category."
    )

    return src, cards, focus_text


# =========================================================
# 5. RUN APP
# =========================================================

if __name__ == "__main__":
    app.run(debug=True)
