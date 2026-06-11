import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_depth_wise_errors(metrics_dfs_dict, metric='rmse'):
    """
    Plots the error metrics (RMSE/MAE) vs depth for one or more models.
    Depth is on the Y-axis (inverted), and the error is on the X-axis.
    
    Parameters:
    -----------
    metrics_dfs_dict : dict
        Dict mapping model_name -> depth-wise metrics DataFrame (from get_depth_wise_metrics)
    metric : str
        'rmse' or 'mae' or 'r2'
    """
    fig = go.Figure()
    
    metric_labels = {
        'rmse': 'RMSE (°C)',
        'mae': 'MAE (°C)',
        'r2': 'R²'
    }
    
    colors = px.colors.qualitative.Plotly
    
    for idx, (model_name, df_metrics) in enumerate(metrics_dfs_dict.items()):
        color = colors[idx % len(colors)]
        
        fig.add_trace(go.Scatter(
            x=df_metrics[metric],
            y=df_metrics['depth'],
            mode='lines+markers',
            name=model_name,
            line=dict(color=color, width=2),
            marker=dict(size=6),
            hovertemplate=f"Model: {model_name}<br>Depth: %{{y}}m<br>{metric_labels[metric]}: %{{x:.3f}}<extra></extra>"
        ))
        
    fig.update_layout(
        title=f"Depth-wise Model Performance ({metric_labels[metric]})",
        xaxis_title=metric_labels[metric],
        yaxis_title="Depth (meters)",
        yaxis=dict(autorange="reversed"),  # Invert axis so 0m (surface) is at the top
        hovermode="closest",
        legend_title="Models",
        template="plotly_white",
        height=600,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_prediction_map(df_eval, selected_depth, parameter='error', model_name=None):
    """
    Plots a 2D spatial map of a parameter at a specific depth level.
    
    Parameters:
    -----------
    df_eval : pd.DataFrame
        Evaluation DataFrame with columns: ['latitude', 'longitude', 'depth', 'temp_true', 'temp_pred', 'error', 'abs_error']
    selected_depth : float
        The depth level to visualize.
    parameter : str
        'temp_true', 'temp_pred', 'error', or 'abs_error'
    model_name : str (optional)
        Name of the model (for title)
    """
    # Find rows with the exact depth
    df_depth = df_eval[df_eval['depth'] == selected_depth]
    
    if df_depth.empty:
        # Fallback to the closest depth level
        available_depths = df_eval['depth'].unique()
        closest_depth = available_depths[np.argmin(np.abs(available_depths - selected_depth))]
        df_depth = df_eval[df_eval['depth'] == closest_depth]
        selected_depth = closest_depth
        
    param_labels = {
        'temp_true': 'Observed Temperature (°C)',
        'temp_pred': 'Predicted Temperature (°C)',
        'error': 'Prediction Error (°C)',
        'abs_error': 'Absolute Error (°C)'
    }
    
    # Configure colorscale based on the parameter
    if parameter == 'error':
        colorscale = 'RdBu' # diverging (blue for underprediction, red for overprediction)
        # Center the color range around 0
        max_val = max(abs(df_depth['error'].min()), abs(df_depth['error'].max()), 0.1)
        color_range = [-max_val, max_val]
    elif parameter == 'abs_error':
        colorscale = 'Reds'
        color_range = [0, df_depth['abs_error'].max()]
    else:
        colorscale = 'Thermal'
        color_range = [df_depth[parameter].min(), df_depth[parameter].max()]
        
    fig = px.scatter(
        df_depth,
        x='longitude',
        y='latitude',
        color=parameter,
        color_continuous_scale=colorscale,
        range_color=color_range,
        title=f"{param_labels[parameter]} at {selected_depth}m" + (f" ({model_name})" if model_name else ""),
        labels={parameter: 'Temp (°C)' if 'temp' in parameter else 'Err (°C)'},
        hover_data={
            'longitude': ':.2f',
            'latitude': ':.2f',
            'sst': ':.2f',
            'temp_true': ':.2f',
            'temp_pred': ':.2f',
            'error': ':.2f'
        },
        template="plotly_white",
        height=500
    )
    
    fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=0.5, color='DarkSlateGrey')))
    fig.update_layout(
        xaxis_title="Longitude (°E)",
        yaxis_title="Latitude (°N)",
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_vertical_profile(df_eval, selected_lat, selected_lon, model_eval_dict=None):
    """
    Finds the nearest coordinate point in the evaluation dataset and plots the vertical temperature profile.
    Overlays predictions of multiple models if provided.
    
    Parameters:
    -----------
    df_eval : pd.DataFrame
        Evaluation DataFrame containing truth and coordinate columns.
    selected_lat : float
    selected_lon : float
    model_eval_dict : dict
        Dict mapping model_name -> df_eval DataFrame (each containing 'temp_pred')
    """
    # 1. Find nearest profile point in df_eval
    # Deduplicate coordinates to find unique profiles
    unique_coords = df_eval[['latitude', 'longitude']].drop_duplicates()
    
    dists = np.sqrt((unique_coords['latitude'] - selected_lat)**2 + (unique_coords['longitude'] - selected_lon)**2)
    nearest_idx = dists.idxmin()
    nearest_lat = unique_coords.loc[nearest_idx, 'latitude']
    nearest_lon = unique_coords.loc[nearest_idx, 'longitude']
    
    # Get ground truth profile at this location
    df_truth_profile = df_eval[
        (df_eval['latitude'] == nearest_lat) & 
        (df_eval['longitude'] == nearest_lon)
    ].sort_values('depth')
    
    fig = go.Figure()
    
    # Plot Ground Truth
    fig.add_trace(go.Scatter(
        x=df_truth_profile['temp_true'],
        y=df_truth_profile['depth'],
        mode='lines+markers',
        name='Observed (Argo/EN4)',
        line=dict(color='black', width=3),
        marker=dict(symbol='circle', size=8),
        hovertemplate="Truth<br>Depth: %{y}m<br>Temp: %{x:.2f}°C<extra></extra>"
    ))
    
    # Overlay model predictions
    colors = px.colors.qualitative.Plotly
    if model_eval_dict:
        for idx, (model_name, df_m_eval) in enumerate(model_eval_dict.items()):
            color = colors[idx % len(colors)]
            df_m_profile = df_m_eval[
                (df_m_eval['latitude'] == nearest_lat) & 
                (df_m_eval['longitude'] == nearest_lon)
            ].sort_values('depth')
            
            fig.add_trace(go.Scatter(
                x=df_m_profile['temp_pred'],
                y=df_m_profile['depth'],
                mode='lines+markers',
                name=model_name,
                line=dict(color=color, width=2, dash='dash'),
                marker=dict(symbol='x', size=6),
                hovertemplate=f"{model_name}<br>Depth: %{{y}}m<br>Temp: %{{x:.2f}}°C<extra></extra>"
            ))
            
    fig.update_layout(
        title=f"Vertical Temperature Profile comparison at Lat: {nearest_lat:.3f}°N, Lon: {nearest_lon:.3f}°E",
        xaxis_title="Temperature (°C)",
        yaxis_title="Depth (meters)",
        yaxis=dict(autorange="reversed"), # Invert depth axis
        hovermode="closest",
        legend_title="Legend",
        template="plotly_white",
        height=600,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig

def plot_metrics_comparison(metrics_dict, metric_type='rmse'):
    """
    Plots a bar chart comparing a metric (RMSE, MAE, R2) across multiple models.
    
    Parameters:
    -----------
    metrics_dict : dict
        Dict mapping model_name -> dict of metrics, e.g. {'XGBoost': {'rmse': 0.35, 'mae': 0.2, 'r2': 0.95}}
    metric_type : str
        'rmse', 'mae', or 'r2'
    """
    model_names = list(metrics_dict.keys())
    values = [metrics_dict[name][metric_type] for name in model_names]
    
    metric_labels = {
        'rmse': 'RMSE (Lower is Better)',
        'mae': 'MAE (Lower is Better)',
        'r2': 'R² Score (Higher is Better)'
    }
    
    # Select color scale based on the metric
    if metric_type == 'r2':
        colors = ['#2ca02c' if v == max(values) else '#bcbd22' for v in values] # highlight best
    else:
        colors = ['#1f77b4' if v == min(values) else '#7f7f7f' for v in values] # highlight best (lowest)
        
    fig = go.Figure(data=[
        go.Bar(
            x=model_names,
            y=values,
            marker_color=colors,
            text=[f"{v:.4f}" for v in values],
            textposition='auto',
            hovertemplate="Model: %{x}<br>Value: %{y:.4f}<extra></extra>"
        )
    ])
    
    fig.update_layout(
        title=f"Model Comparison: {metric_labels[metric_type]}",
        xaxis_title="Models",
        yaxis_title=metric_type.upper() + (" (°C)" if metric_type != 'r2' else ""),
        template="plotly_white",
        height=400,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig
