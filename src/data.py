import numpy as np
import pandas as pd

def generate_synthetic_ocean_data(n_profiles=150, random_state=42):
    """
    Generates a physically realistic synthetic ocean dataset.
    
    Each profile contains observations at various depth levels.
    Surface features: sst, ssh, sss, ussw, vssw, latitude, longitude
    Subsurface target: temp (temperature at depth z)
    
    Parameters:
    -----------
    n_profiles : int
        Number of unique spatial profiles to generate.
    random_state : int
        Seed for reproducibility.
        
    Returns:
    --------
    pd.DataFrame
        Flattend DataFrame with columns: 
        ['latitude', 'longitude', 'sst', 'ssh', 'sss', 'ussw', 'vssw', 'depth', 'temp']
    """
    np.random.seed(random_state)
    
    # 1. Define standard depths (in meters)
    depths = np.array([
        0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 700, 800, 900, 1000
    ])
    n_depths = len(depths)
    
    # 2. Generate spatial coordinates (e.g. North Atlantic Gulf Stream region)
    # Latitudes: 25°N to 45°N
    # Longitudes: -75°E to -45°E
    latitudes = np.random.uniform(25.0, 45.0, n_profiles)
    longitudes = np.random.uniform(-75.0, -45.0, n_profiles)
    
    data_list = []
    
    for i in range(n_profiles):
        lat = latitudes[i]
        lon = longitudes[i]
        
        # 3. Simulate surface variables based on ocean physics
        # SST: Warmer in the south, colder in the north, with longitudinal gradient (e.g., Gulf Stream path)
        # We can model a temperature front
        dist_from_front = (lat - 35.0) - 0.2 * (lon + 60.0)
        sst_base = 24.0 - 0.6 * (lat - 25.0) + 0.1 * (lon + 75.0)
        # Apply smooth tanh transition for a front
        sst = sst_base - 3.0 * np.tanh(dist_from_front / 3.0) + np.random.normal(0, 0.4)
        sst = np.clip(sst, 8.0, 30.0) # bound check
        
        # SSH: Aligns with SST due to thermal expansion and geostrophy
        ssh = 0.04 * (sst - 18.0) + np.random.normal(0, 0.08)
        ssh = np.clip(ssh, -0.8, 1.2)
        
        # SSS: Typically higher in warm evaporation regions, lower in cold/precipitation regions
        sss = 35.0 + 0.05 * (sst - 20.0) - 0.02 * (lat - 35.0) + np.random.normal(0, 0.15)
        sss = np.clip(sss, 32.0, 37.5)
        
        # Winds: USSW (Zonal, U) and VSSW (Meridional, V)
        # Westerlies in the north, trade-like or variable in south
        ussw = 4.0 + 0.3 * (lat - 35.0) + np.random.normal(0, 1.5)
        vssw = -2.0 - 0.1 * (lon + 60.0) + np.random.normal(0, 1.5)
        
        # 4. Generate Subsurface Temperature profile T(z)
        # We use a logistic thermocline model:
        # T(z) = T_deep + (SST - T_deep) / (1 + exp((z - z_c) / w))
        # z_c: Thermocline depth (center of transition). Warmer SST / higher SSH implies deeper thermocline
        zc = 120.0 + 6.0 * (sst - 18.0) + 80.0 * ssh + np.random.normal(0, 15.0)
        zc = np.clip(zc, 40.0, 350.0) # keep thermocline in reasonable depth bounds
        
        # w: Thermocline width (thickness of transition layer)
        w = 35.0 + 1.5 * (sst - 18.0)
        w = np.clip(w, 15.0, 80.0)
        
        t_deep = 3.5 # deep ocean asymptotic temperature
        
        for z in depths:
            # Base thermocline formula
            temp_val = t_deep + (sst - t_deep) / (1.0 + np.exp((z - zc) / w))
            
            # Add small fluctuations, higher variance in the thermocline (unstable density gradient region)
            # Thermocline is where dT/dz is highest
            dt_dz = -(sst - t_deep) * np.exp((z - zc) / w) / (w * (1.0 + np.exp((z - zc) / w))**2)
            noise_amplitude = 0.05 + 2.5 * abs(dt_dz) # noise increases where temperature gradient is steep
            temp_noise = np.random.normal(0, noise_amplitude)
            
            final_temp = temp_val + temp_noise
            # Subsurface temperature shouldn't exceed surface temperature (in this simple model)
            # nor drop below freezing point of seawater (~ -1.9C)
            final_temp = np.clip(final_temp, -1.5, sst)
            
            data_list.append({
                'latitude': lat,
                'longitude': lon,
                'sst': sst,
                'ssh': ssh,
                'sss': sss,
                'ussw': ussw,
                'vssw': vssw,
                'depth': float(z),
                'temp': final_temp
            })
            
    return pd.DataFrame(data_list)

def load_custom_dataset(file_or_path):
    """
    Loads and validates an uploaded custom CSV ocean dataset.
    
    Expected Columns:
    -----------------
    ['latitude', 'longitude', 'sst', 'ssh', 'sss', 'ussw', 'vssw', 'depth', 'temp']
    
    Returns:
    --------
    pd.DataFrame
    """
    df = pd.read_csv(file_or_path)
    
    required_cols = ['latitude', 'longitude', 'sst', 'ssh', 'sss', 'ussw', 'vssw', 'depth', 'temp']
    
    # Check case-insensitive and rename if matches
    df.columns = [col.lower().strip() for col in df.columns]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Uploaded dataset is missing required columns: {missing_cols}. "
            f"Please ensure it contains: {required_cols}"
        )
        
    # Drop rows with NaN in critical features or targets
    df = df.dropna(subset=required_cols)
    
    return df[required_cols]
