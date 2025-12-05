# Method of Critical Fluctuations (MCF) Toolbox

This repository contains a **Streamlit** application for analyzing time series data using the **Method of Critical Fluctuations (MCF)**.

## Features

-   **Interactive Data Import**: Support for `.txt`, `.csv`, and `.dat` files.
-   **Interactive Selection**: Select critical windows using mouse interactions (Box/Lasso select) on a Plotly chart or synchronized manual input boxes.
-   **Vectorized Calculations**: Efficient computation of Autocorrelation and Laminar Lengths using NumPy.
-   **Statistical Analysis**: Rolling statistics (Mean, Std Dev) and Autocorrelation function.
-   **Critical Point Analysis**: Interactive histogram for determining the critical point ($\phi_0$).
-   **Laminar Length Distribution**: Power-law fitting with visualization of $p_2, p_3$ exponents and $R^2$ directly on the plots.

## Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/AndreasBakogiannis/MCF-Toolbox.git
    cd MCF-Toolbox
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

## Dependencies

-   `streamlit`
-   `pandas`
-   `numpy`
-   `scipy`
-   `plotly`
-   `streamlit-plotly-events`
