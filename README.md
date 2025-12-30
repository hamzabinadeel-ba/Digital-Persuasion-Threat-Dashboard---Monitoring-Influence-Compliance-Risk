# Digital-Persuasion-Threat-Dashboard: Monitoring-Influence-Compliance-Risk
An exploratory NLP and visual analytics project analysing phishing and social engineering messages. Using Python and Dash, the project examines message length, urgency cues and lexical patterns across threat types, critiques AI-generated insights and delivers an interactive dashboard to support cyber-risk monitoring and executive decision-making.

# Project Overview
This project applies exploratory data analysis (EDA), natural language processing (NLP) and interactive visualisation to analyse 621 real-world text messages across 6 categories (phishing, malware, scareware, baiting, pretexting, non-malicious). Using Python and Dash, the project quantifies message length, urgency cues and lexical patterns to surface behavioural risk signals and deliver an executive-facing dashboard for ethical cyber-risk monitoring.

# Business and Analytical Context

Modern cyber threats rely on psychological persuasion, not just technical exploits. This project focuses on how malicious messages influence users through:

- Narrative complexity

- Time pressure and urgency

- Repeated lexical manipulation patterns

- Senior decision-makers rarely see these signals quantified.

This dashboard makes them measurable, comparable and actionable.

# Dataset

- **Source:** Zenodo – Multiclass NLP Dataset for Phishing & Social Engineering

- **Total messages analysed:** 621 (from 624 raw records after cleaning)

- **Threat categories (6):**

  - Phishing (114)

  - Scareware (100)

  - Baiting (80)

  - Malware (78)

  - Pretexting (78)

  - Non-malicious (171)

# Methodology
## Data Preparation

- Resolved structural data issues (labels embedded in text)

- Parsed and cleaned messages using Python (pandas, regex)

- Engineered interpretable NLP features:

  - Word count per message
  - Binary urgency flags
  - Unigram and bigram frequencies

## Exploratory Data Analysis (EDA)

- Message Length (Structural Persuasion)
- Phishing: 80.5 words, max 519, SD 140.7
- Scareware: 14.2 words, very low variance
- Baiting & Malware: 16-24 words
- Non-malicious: 23.7 words

**Phishing messages are 3.4× longer than non-malicious messages, indicating legitimacy-building narratives.**

## Urgency Cue Analysis (Psychological Pressure)
Percentage of messages containing urgency cues:

- Phishing: 42.98%
- Pretexting: 37.18%
- Non-malicious: 18.13%
- Scareware: 13%
- Malware: 8.97%
- Baiting: 2.5%

**Phishing shows the highest frequency and diversity of urgency cues.**

## Lexical & Narrative Patterns

- **Phishing:** “account”, “verify”, “click”, “reset”
- **Scareware:** “download”, “infected”, “attachment”
- **Baiting:** “free”, “offer”, “exclusive”
- **Pretexting:** “customer”, “password”, “service”
- **Non-malicious:** conversational phrases (“best regards”, “let know”)

**Distinct unigram and bigram signatures confirm differentiated persuasion strategies.**

# Interactive Dashboard
## Tools

- Python Dash used for application framework
- Plotly used for interactive charts
- Bootstrap used for responsive layout

## Visual Components (3 Interconnected Views)

- **Visual 1: Message Length Comparison:** Quantifies narrative complexity across threat types.
- **Visual 2: Urgency Cue Frequency:** Displays proportion of messages with time-pressure language.
- **Visual 3: Lexical Risk Signatures:** Unigram word cloud + top bigram metric cards.

## Interactivity

- Click-driven cross-filtering
- Global threat-category dropdown
- Pre-aggregated metrics for real-time responsiveness

# AI Governance and Critical Evaluation

Generative AI was used responsibly and transparently:

- **Gemini:** Ideation and exploratory hypotheses
- **ChatGPT:** Coding scaffolding for Dash

All AI outputs were:
- Treated as hypotheses, not facts
- Quantitatively validated against EDA results
- Corrected where numerical hallucinations occurred

This project demonstrates human-in-the-loop analytics, not AI dependency.

# Key Quantitative Insights

- Phishing messages are longer, more variable and urgency-heavy
- Urgency appears in 43% of phishing vs 18% of non-malicious messages
- Each threat category has distinct lexical persuasion fingerprints
- Structural and linguistic cues can be measured before user compromise

# Skills Demonstrated

- Exploratory data analysis (EDA)
- NLP feature engineering (interpretable methods)
- Interactive dashboard development
- Quantitative persuasion analysis
- Ethical and critical AI evaluation
- Executive-level data storytelling

# Intended Audience

- Analytics and data science roles
- Cybersecurity and risk teams
- Consulting and advisory positions
- Organisations monitoring digital persuasion threats

# Key Takeaway
Cybersecurity risk is behavioural as much as technical. This project shows how quantitative NLP and visual analytics can expose hidden persuasion mechanisms and support ethical, data-driven cyber-risk decisions.
