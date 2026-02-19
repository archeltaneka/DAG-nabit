# 🎯 DAG-nabit: Business Strategy Simulation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://dag-nabit.streamlit.app)
[![Tests](https://img.shields.io/badge/tests-in--development-yellow.svg)](https://github.com/archeltaneka/DAG-nabit/actions)

**DAG-nabit** is an interactive business strategy simulation dashboard designed to demonstrate the power of **causal inference** in marketing and customer targeting. It bridges the gap between simple A/B testing and advanced uplift modeling to help businesses optimize their strategy and maximize ROI.

---

## 🚀 Key Features

### 1. Ground Truth Simulation
*   **Customer Segmentation**: God-mode view into 4 distinct customer types: **Loyalists**, **Persuadables**, **Sleeping Dogs**, and **Lost Causes**.
*   **Feature Distributions**: Interactive histograms showing how observable customer features (age, activity, account value) correlate with hidden personas.

### 2. Experimentation Framework
*   **Randomized A/B Test**: Power analysis and conversion lift tracking for clean experiments.
*   **Selection Bias Demonstration**: Run "Naive" tests that target only highly active users to see how selection bias inflates marketing metrics.
*   **The A/B Test Paradox**: Visual contrast between true causal lift and biased observations.

### 3. Causal Inference Methods
*   **Propensity Score Matching (PSM)**: Rebalancing biased groups to mimic randomized trials.
*   **Double Machine Learning (DML)**: Leveraging ML to isolate treatment effects in complex, non-linear environments.
*   **Uplift Modeling**: Predicting individual-level treatment effects to identify "Persuadables" and avoid "Sleeping Dogs."

---

## 🛠️ Tech Stack

*   **Frontend**: [Streamlit](https://streamlit.io/)
*   **Analytics**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
*   **Visualization**: [Plotly](https://plotly.com/python/)
*   **Causal ML**: Scikit-Learn (custom wrappers), logic for PSM and DML.

---

## 📁 Project Structure

```text
├── src/
│   ├── generators/       # Simulation engine & data generation
│   ├── experiments/      # A/B testing & Multi-arm bandit logic
│   └── causal_analysis/  # PSM, Double ML, and Uplift models
├── notebooks/            # Deep-dive walkthroughs & theory
├── tests/                # Unit tests (In Development)
├── app.py                # Main Streamlit dashboard
└── brand_guideline.md    # Design system and UI specs
```

---

## 🏗️ Getting Started

### Prerequisites

*   Python 3.10 or higher
*   pip

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/archeltaneka/DAG-nabit.git
    cd DAG-nabit
    ```

2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the dashboard**:
    ```bash
    streamlit run app.py
    ```

---

## 🧪 Testing

The unit test suite is currently **under development**. You can run existing checks using:

```bash
python -m pytest tests/
```

---

## 📜 License

MIT License © 2025 **Archel Taneka**

## ⚙️ Want to contribute?

PRs, suggestions, and issues are welcome.