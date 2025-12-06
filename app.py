import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
from streamlit_plotly_events import plotly_events
import io

# --- Helper Functions ---

@st.cache_data
def fortran_autocorr(y, max_lag=500):
    """
    Calculates autocorrelation exactly as specified in the Fortran code provided.
    
    Fortran Logic:
      barx = sum(y) / n
      var = sum((y - barx)**2)
      for m = 0 to 500:
          sumc = sum( (y(i)-barx) * (y(i+m)-barx) )
          corr(m) = sumc / var
    """
    n = len(y)
    if n == 0: return np.zeros(max_lag+1)
    
    # 1. Calculate Mean (barx)
    barx = np.mean(y)
    
    # 2. Calculate Variance term (sum2 in Fortran)
    # Note: Fortran code calculates sum of squared differences, NOT divided by N.
    var = np.sum((y - barx)**2)
    
    if var == 0: return np.zeros(max_lag+1)
    
    corr = []
    
    # 3. Loop for lags
    # We use numpy vectorization for the inner loop (i) to keep Python fast,
    # but the math is identical to the Fortran DO loops.
    effective_lag = min(max_lag, n - 1)
    
    for m in range(effective_lag + 1):
        # Construct the shifted arrays
        # Fortran: i goes from 1 to n-m
        # Python: slice 0 to n-m
        
        # term1 = y(i) - barx
        term1 = y[:n-m] - barx
        
        # term2 = y(i+m) - barx
        term2 = y[m:] - barx
        
        # sumc = sum(term1 * term2)
        sumc = np.sum(term1 * term2)
        
        # corr(m) = sumc / var
        corr.append(sumc / var)
        
    return np.array(corr)

@st.cache_data
def get_laminar_lengths(data, phi_0, phi_l):
    """
    Calculates laminar lengths based on the condition: phi_0 < x <= phi_l
    Matches the logic in MCF.docx and coupled_maps.m
    """
    # Vectorized detection
    condition = (data > phi_0) & (data <= phi_l)
    
    # Pad to detect edges at start/end
    padded = np.concatenate(([0], condition.astype(int), [0]))
    diffs = np.diff(padded)
    
    # 1 indicates 0->1 (start of region), -1 indicates 1->0 (end of region)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    
    lengths = ends - starts
    
    # Return only valid lengths > 0
    return lengths[lengths > 0]

def truncated_power_law(x, p1, p2, p3):
    return p1 * (x**-p2) * np.exp(-p3 * x)

@st.cache_data
def parse_uploaded_file(file_content, filename, col_idx):
    try:
        file_buffer = io.BytesIO(file_content)
        preview = None
        
        if filename.endswith('.txt'):
            raw_data = np.loadtxt(file_buffer)
        else:
            try:
                # Attempt comma-separated
                df = pd.read_csv(file_buffer, header=None, sep=',')
                # Check if header exists (heuristic)
                if df.iloc[0].apply(lambda x: isinstance(x, str)).any():
                     file_buffer.seek(0)
                     df = pd.read_csv(file_buffer, sep=',')
            except:
                # Fallback to whitespace
                file_buffer.seek(0)
                df = pd.read_csv(file_buffer, delim_whitespace=True, header=None)
            
            if isinstance(df, pd.DataFrame):
                preview = df.head()
                max_col = df.shape[1] - 1
                if col_idx > max_col:
                    return None, f"Column index {col_idx} out of bounds (max {max_col}).", None
                raw_data = df.iloc[:, col_idx].values
            else:
                 return None, "Failed to parse dataframe.", None

        return raw_data.flatten(), None, preview
    except Exception as e:
        return None, str(e), None

def render_interactive_selection(raw_data, step, key_prefix="main"):
    """Renders the interactive plot with Box Select enabled."""
    
    fig_main = go.Figure()
    fig_main.add_trace(go.Scattergl(
        x=np.arange(0, len(raw_data), step), 
        y=raw_data[::step],
        mode='lines', 
        line=dict(color='#007bff', width=1), 
        name='Raw Data'
    ))
    
    # Visualize current selection
    if st.session_state.sel_start < st.session_state.sel_end:
        fig_main.add_vrect(
            x0=st.session_state.sel_start, 
            x1=st.session_state.sel_end,
            fillcolor="red", opacity=0.2, 
            layer="below", line_width=0,
        )

    fig_main.update_layout(
        title="Full Time Series (Box Select Enabled)",
        xaxis_title="Index", 
        yaxis_title="Amplitude",
        height=500 if key_prefix != "main" else 400,
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_white",
        dragmode='select', # Box select by default
        modebar_remove=['lasso2d'] # Remove Lasso as requested
    )
    
    # Unique key ensures plot refreshes correctly
    plot_key = f"{key_prefix}_plot_select_{st.session_state.plot_refresh_id}"
    selected_points = plotly_events(fig_main, select_event=True, key=plot_key)
    
    if selected_points:
        xs = [p['x'] for p in selected_points]
        if xs:
            new_start = int(min(xs))
            new_end = int(max(xs))
            if new_start != st.session_state.sel_start or new_end != st.session_state.sel_end:
                st.session_state.sel_start = new_start
                st.session_state.sel_end = new_end
                st.rerun()

    st.info("Use the **Box Select** tool on the plot OR Input Boxes below to select range.")
    
    col_sel1, col_sel2 = st.columns(2)
    with col_sel1:
        st.number_input("Start Index", min_value=0, max_value=len(raw_data)-1, 
                        key=f'{key_prefix}_sel_start_input', value=st.session_state.sel_start,
                        on_change=update_range_from_inputs)
        
    with col_sel2:
        st.number_input("End Index", min_value=0, max_value=len(raw_data), 
                        key=f'{key_prefix}_sel_end_input', value=st.session_state.sel_end,
                        on_change=update_range_from_inputs)

# --- App Layout ---

st.set_page_config(page_title="MCF Toolbox (v2.6)", layout="wide", page_icon="📈")

st.title("📈 Method of Critical Fluctuations (MCF) Toolbox")
st.markdown("""
This toolbox allows you to analyze time series data using the Method of Critical Fluctuations. 
Import your data, select a critical window, and analyze statistics and laminar length distributions.
""")

# --- Sidebar: Data Import ---
with st.sidebar:
    st.header("1. Data Import")
    uploaded_file = st.file_uploader("Upload Time Series", type=['txt', 'csv', 'dat'])
    
    col_idx = st.number_input("Select Column Index", min_value=0, value=0, help="For multi-column files")
    
    st.divider()
    st.markdown("### Settings")
    st.info("Additional settings can be found in the main view tabs.")

# --- Main Logic ---

if 'sel_start' not in st.session_state: st.session_state.sel_start = 0
if 'sel_end' not in st.session_state: st.session_state.sel_end = 0
if 'phi_0' not in st.session_state: st.session_state.phi_0 = 0.0
if 'plot_refresh_id' not in st.session_state: st.session_state.plot_refresh_id = 0

def update_range_from_inputs():
    st.session_state.plot_refresh_id += 1
    for k in st.session_state:
        if k.endswith('_sel_start_input'): st.session_state.sel_start = st.session_state[k]
        elif k.endswith('_sel_end_input'): st.session_state.sel_end = st.session_state[k]

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    raw_data, error, preview = parse_uploaded_file(file_bytes, uploaded_file.name, col_idx)
    
    if error:
        st.error(f"Error loading file: {error}")
    else:
        st.sidebar.success(f"Loaded {len(raw_data):,} points.")
        if preview is not None:
            with st.expander("Data Preview", expanded=False):
                st.dataframe(preview)
        
        # --- Section 2: Selection ---
        st.subheader("2. Select Critical Window")
        
        step = 1 if len(raw_data) < 50000 else int(len(raw_data) / 10000)
        render_interactive_selection(raw_data, step, key_prefix="main")

        start_val = st.session_state.sel_start
        end_val = st.session_state.sel_end
        
        if st.session_state.sel_start >= st.session_state.sel_end and st.session_state.sel_end != 0:
             st.warning("Start Index must be less than End Index.")
        
        if start_val < end_val:
            section = raw_data[start_val:end_val]
            
            # --- Tabs for Analysis ---
            tab1, tab2, tab3 = st.tabs(["📊 Statistics", "📐 Histogram & φ₀", "📉 Laminar Analysis"])
            
            # --- Tab 1: Statistics ---
            with tab1:
                st.subheader("Critical Window Statistics")
                
                # Static Highlight
                fig_high = go.Figure()
                fig_high.add_trace(go.Scattergl(x=np.arange(0, len(raw_data), step), y=raw_data[::step], mode='lines', line=dict(color='lightgray'), name='Full Data'))
                fig_high.add_trace(go.Scattergl(x=np.arange(start_val, end_val), y=section, mode='lines', line=dict(color='red'), name='CW'))
                fig_high.update_layout(height=300, margin=dict(l=0,r=0,t=30,b=0), title="Critical Window",xaxis_title="Index", yaxis_title="Amplitude", template="plotly_white")
                st.plotly_chart(fig_high, use_container_width=True)
                
                col_s1, col_s2 = st.columns(2)
                
                with col_s1:
                    st.markdown("#### Rolling Statistics (Window=500)")
                    if len(section) > 500:
                        windows = np.arange(500, len(section), 500)
                        means = [np.mean(section[:w]) for w in windows]
                        stds = [np.std(section[:w]) for w in windows]
                        
                        fig_roll = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1)
                        fig_roll.add_trace(go.Scatter(x=windows, y=means, mode='lines+markers', name='Mean'), row=1, col=1)
                        fig_roll.add_trace(go.Scatter(x=windows, y=stds, mode='lines+markers', name='Std Dev', line=dict(color='orange')), row=2, col=1)
                        fig_roll.update_layout(height=400, showlegend=False, template="plotly_white")
                        fig_roll.update_yaxes(title_text="Mean", row=1, col=1)
                        fig_roll.update_yaxes(title_text="Std Dev", row=2, col=1)
                        st.plotly_chart(fig_roll, use_container_width=True)
                    else:
                        st.warning("Selection too short for rolling stats.")

                with col_s2:
                    st.markdown("#### Autocorrelation")
                    with st.spinner("Computing autocorrelation..."):
                        # Use Fortran-style function
                        corr = fortran_autocorr(section)
                    
                    fig_ac = go.Figure()
                    fig_ac.add_trace(go.Scatter(y=corr, mode='lines', name='Autocorr', line=dict(color='black')))
                    fig_ac.update_layout(
                        title="Autocorrelation Function (Fortran)",
                        xaxis_title="Lag",
                        yaxis_title="C(m)/C(0)",
                        height=400,
                        template="plotly_white"
                    )
                    st.plotly_chart(fig_ac, use_container_width=True)

            # --- Tab 2: Histogram ---
            with tab2:
                col_h1, col_h2 = st.columns([1, 2])
                
                with col_h1:
                    st.subheader("Settings")
                    add_noise = st.checkbox("Add Uniform Noise", help="Useful to break discretization artifacts")
                    eps = 0.0
                    if add_noise:
                        eps = st.number_input("Noise Epsilon", value=0.001, format="%.4f")
                    
                    bin_mode = st.radio("Binning", ["Auto", "Manual Number", "Custom Edges"])
                    bins_arg = 'auto'
                    if bin_mode == "Manual Number":
                        bins_arg = st.number_input("Number of Bins", 5, 10000, 50)
                    elif bin_mode == "Custom Edges":
                        edges_txt = st.text_input("Edges (space separated)", "0 10 20")
                        try: bins_arg = sorted([float(x) for x in edges_txt.split()])
                        except: bins_arg = 50
                    
                    st.markdown("---")
                    st.markdown(r"**Critical Point ($\phi_0$)**")
                    st.session_state.phi_0 = st.number_input(r"Set $\phi_0$", value=st.session_state.phi_0, format="%.3f")
                    
                    enable_click = st.checkbox(r"Click on graph to set $\phi_0$", value=False)

                with col_h2:
                    data_hist = section.copy()
                    if add_noise and eps > 0:
                        data_hist += np.random.uniform(-eps, eps, len(data_hist))
                    
                    counts, edges = np.histogram(data_hist, bins=bins_arg)
                    centers = (edges[:-1] + edges[1:]) / 2
                    
                    fig_hist = go.Figure()
                    fig_hist.add_trace(go.Bar(
                        x=centers, y=counts,
                        marker_color='#17a2b8', opacity=0.7,
                        name='Distribution'
                    ))
                    fig_hist.add_vline(x=st.session_state.phi_0, line_width=2, line_dash="dash", line_color="red")
                    fig_hist.add_annotation(x=st.session_state.phi_0, y=max(counts), text="φ₀", showarrow=False, yshift=10)
                    
                    fig_hist.update_layout(
                        title="Distribution of E-fluctuations of CW ",
                        xaxis_title="Value", yaxis_title="Count",
                        bargap=0, height=450, template="plotly_white",
                        hovermode="x"
                    )
                    
                    if enable_click:
                        selected = plotly_events(fig_hist, click_event=True, hover_event=False)
                        if selected:
                            st.session_state.phi_0 = selected[0]['x']
                            st.rerun()
                    else:
                        st.plotly_chart(fig_hist, use_container_width=True)

            # --- Tab 3: Laminar Analysis ---
            with tab3:
                st.subheader("Laminar Length Distribution Analysis")
                
                with st.form("fit_form"):
                    c1, c2 = st.columns(2)
                    with c1:
                        phi_l_txt = st.text_input(r"$\phi_l$ Values (space separated)", value="", placeholder="e.g. 0.5 1.0 1.5")
                    with c2:
                        lmax = st.number_input(r"Max Length ($L_{\max}$) for fitting", value=0, help="0 to include all points")
                    
                    submitted = st.form_submit_button("Run Analysis")
                
                if submitted:
                    try:
                        phi_l_vals = [float(x) for x in phi_l_txt.split()]
                        if not phi_l_vals:
                            st.warning("Please enter at least one phi_l value.")
                        else:
                            results = []
                            res_tabs = st.tabs([rf"$\phi_l$={p}" for p in phi_l_vals])
                            
                            for i, phi_l in enumerate(phi_l_vals):
                                with res_tabs[i]:
                                    lengths = get_laminar_lengths(data_hist, st.session_state.phi_0, phi_l)
                                    
                                    if len(lengths) == 0:
                                        st.warning("No laminar regions found.")
                                        continue
                                        
                                    bins_fit = np.arange(1, np.max(lengths) + 2) - 0.5
                                    counts_l, edges_l = np.histogram(lengths, bins=bins_fit)
                                    centers_l = (edges_l[:-1] + edges_l[1:]) / 2
                                    
                                    if lmax > 0: mask = (counts_l > 0) & (centers_l <= lmax)
                                    else: mask = (counts_l > 0)
                                    
                                    x_fit = centers_l[mask]
                                    y_fit = counts_l[mask]
                                    
                                    if len(x_fit) < 3:
                                        st.warning("Not enough points to fit.")
                                        continue
                                    
                                    try:
                                        p0_guess = [np.max(y_fit), 1.33, 0.01]
                                        popt, pcov = curve_fit(truncated_power_law, x_fit, y_fit,
                                                             p0=p0_guess, bounds=([0, 0, -1], [np.inf, 5, 1]))
                                        p1, p2, p3 = popt
                                        perr = np.sqrt(np.diag(pcov))
                                        
                                        residuals = y_fit - truncated_power_law(x_fit, *popt)
                                        ss_res = np.sum(residuals**2)
                                        ss_tot = np.sum((y_fit - np.mean(y_fit))**2)
                                        r2 = 1 - (ss_res / ss_tot)
                                        
                                        results.append({'phi_l': phi_l, 'p2': p2, 'p2_err': perr[1], 'p3': p3, 'p3_err': perr[2], 'R2': r2})
                                        
                                        fig_fit = go.Figure()
                                        fig_fit.add_trace(go.Scatter(x=x_fit, y=y_fit, mode='markers', name='Data', marker=dict(color='black')))
                                        
                                        x_smooth = np.logspace(np.log10(min(x_fit)), np.log10(max(x_fit)), 100)
                                        y_smooth = truncated_power_law(x_smooth, *popt)
                                        fig_fit.add_trace(go.Scatter(x=x_smooth, y=y_smooth, mode='lines', name='Fit', line=dict(color='red')))
                                        
                                        result_text = (f"p<sub>2</sub> = {p2:.2f} ± {perr[1]:.2f}<br>"
                                                       f"p<sub>3</sub> = {p3:.2f} ± {perr[2]:.2f}<br>"
                                                       f"R² = {r2:.4f}")
                                        
                                        fig_fit.add_annotation(text=result_text, align='left', showarrow=False,
                                                               xref='paper', yref='paper', x=0.95, y=0.95,
                                                               xanchor='right', yanchor='top', bordercolor='black', borderwidth=1,
                                                               bgcolor='rgba(255, 255, 255, 0.8)')

                                        fig_fit.update_layout(
                                            title=f"Distribution of Laminar lengths (φ<sub>l</sub> = {phi_l})",
                                            xaxis_title="Laminar Length L", yaxis_title="P(L)",
                                            xaxis_type="log", yaxis_type="log",
                                            template="plotly_white"
                                        )
                                        
                                        st.plotly_chart(fig_fit, use_container_width=True)
                                            
                                    except Exception as e:
                                        st.error(f"Fit failed: {e}")

                            if len(results) > 0:
                                res_df = pd.DataFrame(results)
                                st.divider()
                                st.subheader("Parameter Stability Summary")
                                
                                col_sum1, col_sum2 = st.columns([2, 1])
                                with col_sum1:
                                    fig_sum = go.Figure()
                                    fig_sum.add_trace(go.Scatter(
                                        x=res_df['phi_l'], y=res_df['p2'],
                                        error_y=dict(type='data', array=res_df['p2_err'], visible=True),
                                        mode='lines+markers', name='p2 (Exponent)'
                                    ))
                                    fig_sum.update_layout(title="Exponent p<sub>2</sub> Stability", xaxis_title="φ<sub>l</sub>", yaxis_title="p<sub>2</sub>", template="plotly_white")
                                    st.plotly_chart(fig_sum, use_container_width=True)
                                with col_sum2:
                                    st.dataframe(res_df[['phi_l', 'p2', 'p3', 'R2']].style.format("{:.4f}"))
                                    
                    except Exception as e:
                        st.error(f"Input Error: {e}")