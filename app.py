import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from src.data import generate_synthetic_ocean_data, load_custom_dataset
from src.models import train_and_evaluate, get_depth_wise_metrics, FEATURE_COLS, TARGET_COL
from src.plots import (
    plot_depth_wise_errors,
    plot_prediction_map,
    plot_vertical_profile,
    plot_metrics_comparison
)

# 1. Page Config
st.set_page_config(
    layout="wide",
    page_title="OceanRecon - Subsurface Ocean Temperature Benchmark",
    page_icon="🌊"
)

# Initialize Session State
if 'df' not in st.session_state:
    # Auto-generate synthetic data on first run for immediate usability
    st.session_state.df = generate_synthetic_ocean_data(n_profiles=150, random_state=42)
    st.session_state.data_source_type = "Synthetic Ocean Data"
    st.session_state.n_profiles = 150

if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}

# Title Block
st.title("🌊 OceanRecon: Subsurface Temperature Benchmark")
st.markdown(
    """
    This research dashboard facilitates benchmarking machine learning models for predicting **Subsurface Ocean Temperature** 
    from surface parameter observations (SST, SSH, SSS, Wind) and depth.
    """
)

# 2. Sidebar Navigation & Data Controls
st.sidebar.header("📂 Data Source Configuration")

data_source = st.sidebar.radio(
    "Select Dataset Source:",
    ["Synthetic Ocean Data Generator", "Upload Custom CSV Dataset"],
    index=0 if st.session_state.data_source_type == "Synthetic Ocean Data" else 1
)

if data_source == "Synthetic Ocean Data Generator":
    n_profiles = st.sidebar.slider("Number of Spatial Profiles:", min_value=50, max_value=500, value=st.session_state.n_profiles, step=50)
    if st.sidebar.button("Generate & Load Dataset"):
        st.session_state.df = generate_synthetic_ocean_data(n_profiles=n_profiles, random_state=42)
        st.session_state.data_source_type = "Synthetic Ocean Data"
        st.session_state.n_profiles = n_profiles
        st.session_state.trained_models = {} # reset models on new data
        st.success(f"Generated synthetic dataset with {n_profiles} profiles!")
else:
    uploaded_file = st.sidebar.file_uploader("Upload Ocean CSV File:", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = load_custom_dataset(uploaded_file)
            st.session_state.df = uploaded_df
            st.session_state.data_source_type = "Custom Upload"
            st.session_state.trained_models = {} # reset models on new data
            st.success("Custom CSV dataset loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Error loading CSV: {e}")

# Global Training Config in Sidebar
st.sidebar.header("⚙️ Global Training Config")
train_split = st.sidebar.slider("Train Split Ratio:", min_value=0.5, max_value=0.9, value=0.8, step=0.05)
rand_seed = st.sidebar.number_input("Random Seed:", min_value=1, max_value=9999, value=42)

# Show dataset summary in sidebar
df = st.session_state.df
n_rows = len(df)
n_unique_profiles = len(df[['latitude', 'longitude']].drop_duplicates())
available_depths = sorted(df['depth'].unique())

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Info")
st.sidebar.markdown(f"**Source:** {st.session_state.data_source_type}")
st.sidebar.markdown(f"**Total Records:** {n_rows}")
st.sidebar.markdown(f"**Unique Profiles:** {n_unique_profiles}")
st.sidebar.markdown(f"**Depth Levels:** {len(available_depths)}")

# 3. Main Dashboard Tabs
tabs = st.tabs(["📊 Data Explorer", "🏋️ Model Training & Evaluation", "⚔️ Model Comparison"])

# --- TAB 1: DATA EXPLORER ---
with tabs[0]:
    st.header("Exploratory Data Analysis")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Spatial Grid Distribution")
        st.markdown("Select a surface variable or specific depth level to visualize its spatial distribution.")
        
        map_param = st.selectbox(
            "Select Map Parameter:",
            ["sst", "ssh", "sss", "ussw", "vssw", "Subsurface Temp at Depth"]
        )
        
        if map_param == "Subsurface Temp at Depth":
            sel_depth = st.selectbox("Select Depth (m) for Map:", available_depths, index=min(len(available_depths)//2, len(available_depths)-1))
            fig_map = px.scatter(
                df[df['depth'] == sel_depth],
                x='longitude',
                y='latitude',
                color='temp',
                color_continuous_scale='Thermal',
                title=f"Subsurface Temperature at Depth = {sel_depth}m",
                template="plotly_white",
                height=400
            )
        else:
            # For surface parameters, deduplicate to one row per spatial profile
            df_surf = df.drop_duplicates(subset=['latitude', 'longitude'])
            fig_map = px.scatter(
                df_surf,
                x='longitude',
                y='latitude',
                color=map_param,
                color_continuous_scale='Thermal' if map_param in ['sst', 'ssh', 'sss'] else 'Plasma',
                title=f"Sea Surface {map_param.upper()} Distribution",
                template="plotly_white",
                height=400
            )
            
        fig_map.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))
        st.plotly_chart(fig_map, use_container_width=True)
        
    with col2:
        st.subheader("Observed Vertical Temperature Profile")
        st.markdown("Select an existing coordinate profile to view temperature variation with depth.")
        
        unique_coords = df[['latitude', 'longitude']].drop_duplicates().values
        coord_options = [f"Lat: {c[0]:.3f}, Lon: {c[1]:.3f}" for c in unique_coords]
        selected_coord_str = st.selectbox("Select Profile Coordinates:", coord_options, index=0)
        
        # Parse coordinates
        selected_idx = coord_options.index(selected_coord_str)
        sel_lat, sel_lon = unique_coords[selected_idx]
        
        df_profile = df[(df['latitude'] == sel_lat) & (df['longitude'] == sel_lon)].sort_values('depth')
        
        fig_prof = go.Figure()
        fig_prof.add_trace(go.Scatter(
            x=df_profile['temp'],
            y=df_profile['depth'],
            mode='lines+markers',
            name='Observed Temp',
            line=dict(color='navy', width=3),
            marker=dict(size=8)
        ))
        fig_prof.update_layout(
            title=f"Vertical Temperature Profile at {selected_coord_str}",
            xaxis_title="Temperature (°C)",
            yaxis_title="Depth (meters)",
            yaxis=dict(autorange="reverse"),
            template="plotly_white",
            height=400
        )
        st.plotly_chart(fig_prof, use_container_width=True)
        
    st.markdown("---")
    st.subheader("Feature Statistics & Correlations")
    
    col_stat1, col_stat2 = st.columns([1, 1])
    with col_stat1:
        st.write("Summary Statistics (Surface + Depth Target):")
        st.dataframe(df.describe().T[['mean', 'std', 'min', '50%', 'max']])
        
    with col_stat2:
        st.write("Feature Pearson Correlation Matrix:")
        corr = df.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1.0,
            zmax=1.0,
            template="plotly_white",
            height=320
        )
        fig_corr.update_layout(margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_corr, use_container_width=True)


# --- TAB 2: MODEL TRAINING & EVALUATION ---
with tabs[1]:
    st.header("Train and Evaluate Model")
    
    # Model Selection UI
    model_category = st.radio(
        "Select Model Phase:",
        ["Phase 1: Classical Machine Learning Models", "Phase 2: Deep Learning Models (Future Phases)"],
        index=0
    )
    
    if model_category == "Phase 1: Classical Machine Learning Models":
        model_name = st.selectbox(
            "Select ML Model:",
            ["Linear Regression", "Random Forest", "XGBoost", "LightGBM"]
        )
        
        # Hyperparameters selectors
        st.subheader("Model Hyperparameters")
        hyperparams = {}
        
        if model_name == "Linear Regression":
            st.info("Linear Regression has no major hyperparameters to tune.")
            
        elif model_name == "Random Forest":
            col_hp1, col_hp2 = st.columns(2)
            with col_hp1:
                hyperparams['n_estimators'] = st.slider("Number of Trees:", min_value=10, max_value=250, value=100, step=10)
            with col_hp2:
                hyperparams['max_depth'] = st.slider("Max Depth of Trees:", min_value=3, max_value=25, value=12, step=1)
                
        elif model_name in ["XGBoost", "LightGBM"]:
            col_hp1, col_hp2, col_hp3 = st.columns(3)
            with col_hp1:
                hyperparams['n_estimators'] = st.slider("Number of Boosting Iterations:", min_value=20, max_value=300, value=100, step=10)
            with col_hp2:
                hyperparams['max_depth'] = st.slider("Max Depth:", min_value=2, max_value=15, value=6 if model_name == "XGBoost" else 10, step=1)
            with col_hp3:
                hyperparams['learning_rate'] = st.slider("Learning Rate (eta):", min_value=0.01, max_value=0.3, value=0.1, step=0.01)
                
        # Run Training Button
        st.markdown("---")
        if st.button(f"Train {model_name}"):
            with st.spinner(f"Training {model_name} on {n_rows} records..."):
                results = train_and_evaluate(
                    df=df,
                    model_name=model_name,
                    hyperparams=hyperparams,
                    train_size=train_split,
                    random_state=rand_seed
                )
                st.session_state.trained_models[model_name] = results
                st.success(f"Successfully trained {model_name} in {results['train_time']:.3f} seconds!")
                
    else:
        # Display Phase 2 placeholders as disabled/unavailable
        st.selectbox(
            "Deep Learning Architecture:",
            ["CNN", "U-Net", "Attention U-Net", "GAN / AIGAN"],
            disabled=True,
            help="Deep learning models are planned for Phase 2 and require PyTorch/TensorFlow setup."
        )
        st.warning("Deep Learning model architectures are locked for future implementation phases (Phase 2). Please use classical ML models for benchmarking now.")
        model_name = None
        
    # If currently selected model has training results, show them!
    if model_name and model_name in st.session_state.trained_models:
        results = st.session_state.trained_models[model_name]
        df_eval = results['df_test_eval']
        
        st.markdown("---")
        st.subheader(f"📊 {model_name} Evaluation Summary")
        
        # Display overall metrics cards
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        
        with col_m1:
            st.metric(
                label="Overall Test RMSE",
                value=f"{results['metrics_test']['rmse']:.4f} °C",
                delta=f"Train: {results['metrics_train']['rmse']:.4f} °C",
                delta_color="inverse"
            )
        with col_m2:
            st.metric(
                label="Overall Test MAE",
                value=f"{results['metrics_test']['mae']:.4f} °C",
                delta=f"Train: {results['metrics_train']['mae']:.4f} °C",
                delta_color="inverse"
            )
        with col_m3:
            st.metric(
                label="Overall Test R² Score",
                value=f"{results['metrics_test']['r2']:.4f}",
                delta=f"Train: {results['metrics_train']['r2']:.4f}"
            )
        with col_m4:
            st.metric(
                label="Training Time",
                value=f"{results['train_time']:.4f} sec"
            )
            
        # Detailed Plotting tabs for single model
        st.markdown("### Model Diagnostics Plots")
        p_tab1, p_tab2, p_tab3 = st.tabs(["📉 Depth-wise Errors", "🗺️ Prediction & Error Maps", "📍 Profile Comparison"])
        
        # Sub-tab 1: Depth-wise errors
        with p_tab1:
            col_dw1, col_dw2 = st.columns([2, 1])
            with col_dw1:
                depth_metrics = get_depth_wise_metrics(df_eval)
                selected_err_metric = st.selectbox("Select Plotting Metric:", ["rmse", "mae", "r2"], key="single_err_metric")
                fig_dw = plot_depth_wise_errors({model_name: depth_metrics}, metric=selected_err_metric)
                st.plotly_chart(fig_dw, use_container_width=True)
            with col_dw2:
                st.write("Depth-wise Error Table:")
                st.dataframe(
                    depth_metrics.rename(columns={
                        'depth': 'Depth (m)',
                        'rmse': 'RMSE (°C)',
                        'mae': 'MAE (°C)',
                        'r2': 'R²',
                        'count': 'Samples Count'
                    }).style.format({
                        'Depth (m)': '{:.0f}',
                        'RMSE (°C)': '{:.4f}',
                        'MAE (°C)': '{:.4f}',
                        'R²': '{:.4f}',
                        'Samples Count': '{:,.0f}'
                    }),
                    use_container_width=True,
                    height=450
                )
                
        # Sub-tab 2: Prediction maps
        with p_tab2:
            col_map1, col_map2 = st.columns([1, 4])
            with col_map1:
                st.write("Configure Map:")
                sel_map_depth = st.selectbox("Depth (m):", available_depths, key="single_map_depth")
                sel_map_param = st.selectbox(
                    "Parameter to Plot:",
                    ["temp_true", "temp_pred", "error", "abs_error"],
                    key="single_map_param"
                )
            with col_map2:
                fig_map = plot_prediction_map(df_eval, sel_map_depth, parameter=sel_map_param, model_name=model_name)
                st.plotly_chart(fig_map, use_container_width=True)
                
        # Sub-tab 3: Profile comparison
        with p_tab3:
            col_prof1, col_prof2 = st.columns([1, 3])
            with col_prof1:
                st.write("Select coordinates to check predictions:")
                unique_test_coords = df_eval[['latitude', 'longitude']].drop_duplicates().values
                test_coord_options = [f"Lat: {c[0]:.3f}, Lon: {c[1]:.3f}" for c in unique_test_coords]
                selected_test_coord_str = st.selectbox("Profile Coordinates:", test_coord_options, key="single_profile_coords")
                
                selected_test_idx = test_coord_options.index(selected_test_coord_str)
                test_lat, test_lon = unique_test_coords[selected_test_idx]
            with col_prof2:
                fig_prof = plot_vertical_profile(
                    df_eval, 
                    test_lat, 
                    test_lon, 
                    model_eval_dict={model_name: df_eval}
                )
                st.plotly_chart(fig_prof, use_container_width=True)
    elif model_name:
        st.info(f"Model '{model_name}' has not been trained yet. Click 'Train {model_name}' above to train and visualize results.")


# --- TAB 3: MULTI-MODEL COMPARISON ---
with tabs[2]:
    st.header("Compare Models Side-by-Side")
    
    trained_models_list = list(st.session_state.trained_models.keys())
    
    if len(trained_models_list) == 0:
        st.warning("No models have been trained yet. Please train models in the 'Model Training & Evaluation' tab first before accessing comparisons.")
    else:
        st.write("Select trained models to compare:")
        selected_compare_models = st.multiselect(
            "Compare Models:",
            trained_models_list,
            default=trained_models_list
        )
        
        if len(selected_compare_models) < 1:
            st.info("Please select at least one trained model to view comparison diagnostics.")
        else:
            # 1. Comparative Metrics Table
            st.subheader("Overall Comparison Metrics")
            comp_data = []
            for m_name in selected_compare_models:
                m_res = st.session_state.trained_models[m_name]
                comp_data.append({
                    'Model': m_name,
                    'Train RMSE (°C)': m_res['metrics_train']['rmse'],
                    'Test RMSE (°C)': m_res['metrics_test']['rmse'],
                    'Train MAE (°C)': m_res['metrics_train']['mae'],
                    'Test MAE (°C)': m_res['metrics_test']['mae'],
                    'Train R²': m_res['metrics_train']['r2'],
                    'Test R²': m_res['metrics_test']['r2'],
                    'Training Time (s)': m_res['train_time']
                })
            df_comp = pd.DataFrame(comp_data)
            st.dataframe(
                df_comp.style.format({
                    'Train RMSE (°C)': '{:.4f}',
                    'Test RMSE (°C)': '{:.4f}',
                    'Train MAE (°C)': '{:.4f}',
                    'Test MAE (°C)': '{:.4f}',
                    'Train R²': '{:.4f}',
                    'Test R²': '{:.4f}',
                    'Training Time (s)': '{:.4f}'
                }),
                use_container_width=True
            )
            
            # 2. Metric comparison bar charts
            st.subheader("Performance Metric Breakdown")
            m_comp_tabs = st.tabs(["RMSE Comparison", "MAE Comparison", "R² Comparison"])
            
            # Build metrics_dict for plot functions
            metrics_dict = {name: st.session_state.trained_models[name]['metrics_test'] for name in selected_compare_models}
            
            with m_comp_tabs[0]:
                fig_comp_rmse = plot_metrics_comparison(metrics_dict, metric_type='rmse')
                st.plotly_chart(fig_comp_rmse, use_container_width=True)
            with m_comp_tabs[1]:
                fig_comp_mae = plot_metrics_comparison(metrics_dict, metric_type='mae')
                st.plotly_chart(fig_comp_mae, use_container_width=True)
            with m_comp_tabs[2]:
                fig_comp_r2 = plot_metrics_comparison(metrics_dict, metric_type='r2')
                st.plotly_chart(fig_comp_r2, use_container_width=True)
                
            # 3. Overlaid Depth-wise Errors
            st.subheader("Overlaid Depth-wise Error Profile")
            st.markdown("Compare model prediction errors at different ocean depths simultaneously.")
            
            col_c1, col_c2 = st.columns([1, 4])
            with col_c1:
                comp_err_metric = st.selectbox("Metric to Compare:", ["rmse", "mae", "r2"], key="comp_err_metric")
            with col_c2:
                # Gather depth wise metrics for chosen models
                comp_dw_dict = {}
                for m_name in selected_compare_models:
                    m_eval = st.session_state.trained_models[m_name]['df_test_eval']
                    comp_dw_dict[m_name] = get_depth_wise_metrics(m_eval)
                    
                fig_comp_dw = plot_depth_wise_errors(comp_dw_dict, metric=comp_err_metric)
                st.plotly_chart(fig_comp_dw, use_container_width=True)
                
            # 4. Overlaid vertical profile comparison at specific points
            st.subheader("Vertical Temperature Profile Comparison")
            st.markdown("Select a coordinate point and see how well each model reconstructs the full temperature-depth profile.")
            
            col_prof_c1, col_prof_c2 = st.columns([1, 4])
            
            # Get test coordinates from the first selected model
            first_model_name = selected_compare_models[0]
            first_model_eval = st.session_state.trained_models[first_model_name]['df_test_eval']
            unique_test_coords_comp = first_model_eval[['latitude', 'longitude']].drop_duplicates().values
            test_coord_options_comp = [f"Lat: {c[0]:.3f}, Lon: {c[1]:.3f}" for c in unique_test_coords_comp]
            
            with col_prof_c1:
                selected_comp_coord_str = st.selectbox("Profile Coordinates:", test_coord_options_comp, key="comp_profile_coords")
                selected_comp_idx = test_coord_options_comp.index(selected_comp_coord_str)
                comp_lat, comp_lon = unique_test_coords_comp[selected_comp_idx]
            with col_prof_c2:
                # Gather eval dataframes for all selected models
                model_eval_dict = {name: st.session_state.trained_models[name]['df_test_eval'] for name in selected_compare_models}
                
                fig_comp_prof = plot_vertical_profile(
                    first_model_eval, 
                    comp_lat, 
                    comp_lon, 
                    model_eval_dict=model_eval_dict
                )
                st.plotly_chart(fig_comp_prof, use_container_width=True)
