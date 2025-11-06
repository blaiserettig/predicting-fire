import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from rasterstats import zonal_stats
import os
import helper
import rasterio
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import xarray as xr

occ = gpd.read_file("data/mtbs_fod_pts_data/mtbs_FODpoints_DD.shp")
bnd = gpd.read_file("data/mtbs_perimeter_data/mtbs_perims_DD.shp")

# These files do not have a state or geographic field explicitly, but the beginning two characters from EVENT_ID are the state they belong to ex: "WA123..."
occ["STATE"] = occ["Event_ID"].str[:2]
bnd["STATE"] = bnd["Event_ID"].str[:2]

pnw = ["WA", "OR", "ID"]
occ_pnw = occ[occ["STATE"].isin(pnw)]
bnd_pnw = bnd[bnd["STATE"].isin(pnw)]

occ_pnw.to_file("data/mtbs_fod_pts_data/mtbs_FODpoints_DD_pnw.shp")
bnd_pnw.to_file("data/mtbs_perimeter_data/mtbs_perims_DD_pnw.shp")

print(occ_pnw["STATE"].value_counts())
print(bnd_pnw["STATE"].value_counts())

print(occ_pnw.columns)
print(occ_pnw.head())

print(bnd_pnw.columns)
print(bnd_pnw.head())

bnd_pnw['Ig_Date'] = pd.to_datetime(bnd_pnw['Ig_Date'], errors='coerce')
bnd_pnw['YEAR'] = pd.to_datetime(bnd_pnw['Ig_Date'], errors='coerce').dt.year

bnd_pnw = bnd_pnw.to_crs(epsg=5070)
bnd_pnw['area_ha'] = bnd_pnw.geometry.area / 10000.0

## FIGURE 1

annual = bnd_pnw.groupby('YEAR').agg(
    fire_count = ('Event_ID','count'),
    total_area_ha = ('area_ha','sum'),
    median_area_ha = ('area_ha','median'),
    mean_area_ha = ('area_ha','mean')
).reset_index()

sns.set_theme(style="darkgrid")
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.lineplot(data=annual, x="YEAR", y="fire_count", ax=ax[0])
sns.lineplot(data=annual, x="YEAR", y="total_area_ha", ax=ax[1])
ax[0].set_title("Fire Count per Year")
ax[1].set_title("Total Burned Area (ha)")
plt.tight_layout()
plt.show()

###

### FIGURE 2

fires = bnd_pnw[(bnd_pnw['YEAR'] >= 2000) & (bnd_pnw['YEAR'] <= 2024)]
climate_stats = []

# to geographic CRS
fires_geo = fires.to_crs(epsg=4269)
fires_geo['centroid_lon'] = fires_geo.geometry.centroid.x
fires_geo['centroid_lat'] = fires_geo.geometry.centroid.y

for year in range(2000, 2025):
    fires_year = fires_geo[fires_geo['YEAR'] == year]
    if fires_year.empty:
        continue

    if year not in helper.prism_paths['tmean'] or year not in helper.prism_paths['ppt']:
        continue
    
    tmean_tif = helper.prism_paths['tmean'][year]
    ppt_tif = helper.prism_paths['ppt'][year]

    print(f"Processing {year}...")

    with rasterio.open(tmean_tif) as src:
        tmean_vals = [x[0] for x in src.sample(zip(fires_year['centroid_lon'], fires_year['centroid_lat']))]
    
    with rasterio.open(ppt_tif) as src:
        ppt_vals = [x[0] for x in src.sample(zip(fires_year['centroid_lon'], fires_year['centroid_lat']))]
    
    fires_year = fires_year.copy()
    fires_year['tmean'] = tmean_vals
    fires_year['ppt'] = ppt_vals
    
    fires_year.loc[fires_year['tmean'] < -9000, 'tmean'] = np.nan
    fires_year.loc[fires_year['ppt'] < -9000, 'ppt'] = np.nan
    
    climate_stats.append(fires_year)

fires_climate = gpd.GeoDataFrame(pd.concat(climate_stats, ignore_index=True), crs=fires_geo.crs)
fires_climate = fires_climate.dropna(subset=['tmean', 'ppt'])
fires_climate = fires_climate.to_crs(epsg=5070)

sns.lmplot(data=fires_climate, x='tmean', y='area_ha', hue='STATE', scatter_kws={'s':10})
plt.title("Fire Size vs Mean Annual Temperature (PRISM 2000–2024)")
plt.show()

sns.lmplot(data=fires_climate, x='ppt', y='area_ha', hue='STATE', scatter_kws={'s':10})
plt.title("Fire Size vs Total Annual Precipitation (PRISM 2000–2024)")
plt.show()

###

### FIGURE 3

fires_climate['temp_x_drought'] = fires_climate['tmean'] * (1 / fires_climate['ppt'])
fires_climate['vpd_proxy'] = fires_climate['tmean'] / fires_climate['ppt']

# Random Forest to identify important predictors
features = ['tmean', 'ppt', 'temp_x_drought', 'vpd_proxy']
X = fires_climate[features].dropna()
y = fires_climate.loc[X.index, 'area_ha']

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, np.log1p(y))  # log transform for skewed area data

# Feature importance
importance_df = pd.DataFrame({
    'feature': features,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

sns.barplot(data=importance_df, x='importance', y='feature')
plt.title('Climate Variable Importance for Fire Size')
plt.show()


###

### FIGURE 4

# MEGA FIRES
threshold = fires_climate['area_ha'].quantile(0.95)
fires_climate['is_megafire'] = fires_climate['area_ha'] > threshold

# Compare climate conditions
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.boxplot(data=fires_climate, x='is_megafire', y='tmean', ax=axes[0])
axes[0].set_title('Temperature: Normal vs Mega Fires')

sns.boxplot(data=fires_climate, x='is_megafire', y='ppt', ax=axes[1])
axes[1].set_title('Precipitation: Normal vs Mega Fires')

plt.tight_layout()
plt.show()

# Statistical test
from scipy.stats import mannwhitneyu
stat, p = mannwhitneyu(
    fires_climate[fires_climate['is_megafire']]['tmean'].dropna(),
    fires_climate[~fires_climate['is_megafire']]['tmean'].dropna()
)
print(f"Temperature difference p-value: {p}")

###

###

fires_with_year = fires_climate[['Event_ID', 'YEAR', 'geometry', 'area_ha']].copy()

reburns = []
for idx, fire in fires_with_year.iterrows():
    # Find fires in previous years that overlap
    previous_fires = fires_with_year[fires_with_year['YEAR'] < fire['YEAR']]
    overlaps = previous_fires[previous_fires.intersects(fire.geometry)]
    
    if len(overlaps) > 0:
        reburns.append({
            'Event_ID': fire['Event_ID'],
            'YEAR': fire['YEAR'],
            'previous_fires': len(overlaps),
            'years_since_last': fire['YEAR'] - overlaps['YEAR'].max()
        })

reburn_df = pd.DataFrame(reburns)
print(f"Reburned areas: {len(reburn_df)}")
sns.histplot(data=reburn_df, x='years_since_last', bins=20)
plt.title('Fire Return Interval Distribution')
plt.show()

###

###

fires_climate['MONTH'] = fires_climate['Ig_Date'].dt.month

season_metrics = fires_climate.groupby('YEAR').agg({
    'MONTH': ['min', 'max', lambda x: x.max() - x.min()],
    'Event_ID': 'count'
}).reset_index()

season_metrics.columns = ['YEAR', 'first_fire_month', 'last_fire_month', 
                          'season_length_months', 'fire_count']

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(season_metrics['YEAR'], season_metrics['season_length_months'], 
         color='tab:red', marker='o')
ax1.set_ylabel('Fire Season Length (months)', color='tab:red')
ax1.set_xlabel('Year')
ax2 = ax1.twinx()
ax2.plot(season_metrics['YEAR'], season_metrics['fire_count'], 
         color='tab:blue', alpha=0.3)
ax2.set_ylabel('Fire Count', color='tab:blue')
plt.title('Fire Season Duration Trends')
plt.show()

### ICS 209 DATA

### 1

ics = pd.read_csv("data/ics209/ics209-plus-wf_sitreps_1999to2020.csv", low_memory=False)
ics['REPORT_TO_DATE'] = pd.to_datetime(ics['REPORT_TO_DATE'], format='%m/%d/%Y', errors='coerce')
ics['DISCOVERY_DATE'] = pd.to_datetime(ics['DISCOVERY_DATE'], format='%m/%d/%Y', errors='coerce')
ics['days_since_discovery'] = (ics['REPORT_TO_DATE'] - ics['DISCOVERY_DATE']).dt.days
ics_pnw = ics[ics['POO_STATE'].isin(['WA', 'OR', 'ID'])].copy()


ics_pnw['CY'] = pd.to_numeric(ics_pnw['CY'], errors='coerce')
ics_pnw['DISCOVERY_DATE'] = ics_pnw['DISCOVERY_DATE'].fillna(pd.to_datetime(ics_pnw['CY'].astype(str) + '-07-01'))
ics_pnw['REPORT_TO_DATE'] = ics_pnw['REPORT_TO_DATE'].fillna(pd.to_datetime(ics_pnw['CY'].astype(str) + '-08-01'))
ics_pnw['days_since_discovery'] = (ics_pnw['REPORT_TO_DATE'] - ics_pnw['DISCOVERY_DATE']).dt.days

def analyze_fire_trajectory(fire_id):
    fire_reports = ics_pnw[ics_pnw['FIRE_EVENT_ID'] == fire_id].sort_values('REPORT_TO_DATE')
    
    if len(fire_reports) < 3:
        return None
    
    days = fire_reports['days_since_discovery'].dropna()
    duration_days = days.max() if not days.empty else np.nan
    
    return {
        'fire_id': fire_id,
        'duration_days': duration_days,
        'final_acres': fire_reports['ACRES'].max(),
        'max_personnel': fire_reports['TOTAL_PERSONNEL'].max(),
        'peak_personnel_day': fire_reports.loc[fire_reports['TOTAL_PERSONNEL'].idxmax(), 'days_since_discovery']
                                if fire_reports['TOTAL_PERSONNEL'].notna().any() else np.nan,
        'total_cost': fire_reports['EST_IM_COST_TO_DATE'].max(),
        'growth_rate': (
            fire_reports['ACRES'].max() / duration_days
            if duration_days and duration_days > 0
            else np.nan
        ),
        'structures_destroyed': fire_reports['STR_DESTROYED'].max(),
        'total_evacuated': fire_reports['NUM_EVACUATED'].max()
    }

fire_trajectories = []
for fire_id in ics_pnw['FIRE_EVENT_ID'].unique():
    traj = analyze_fire_trajectory(fire_id)
    if traj:
        fire_trajectories.append(traj)

traj_df = pd.DataFrame(fire_trajectories)
traj_df = traj_df.dropna(subset=['growth_rate', 'duration_days', 'total_cost', 'max_personnel'])

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].scatter(traj_df['final_acres'], traj_df['max_personnel'], alpha=0.5)
axes[0].set_xlabel('Final Fire Size (acres)')
axes[0].set_ylabel('Peak Personnel')
axes[0].set_xscale('symlog')
axes[0].set_yscale('symlog')
axes[0].set_title('Personnel vs Fire Size')

axes[1].scatter(traj_df['growth_rate'], traj_df['max_personnel'], alpha=0.5)
axes[1].set_xlabel('Growth Rate (acres/day)')
axes[1].set_ylabel('Peak Personnel')
axes[1].set_xscale('symlog')
axes[1].set_yscale('symlog')
axes[1].set_title('Personnel vs Growth Rate')

# axes[1,0].scatter(traj_df['duration_days'], traj_df['total_cost'], alpha=0.5)
# axes[1,0].set_xlabel('Fire Duration (days)')
# axes[1,0].set_ylabel('Total Cost ($)')
# axes[1,0].set_yscale('symlog')
# axes[1,0].set_title('Cost vs Duration')

axes[2].scatter(traj_df['final_acres'], traj_df['total_cost'], 
                  c=traj_df['structures_destroyed'], cmap='Reds', alpha=0.6)
axes[2].set_xlabel('Final Fire Size (acres)')
axes[2].set_ylabel('Total Cost ($)')
axes[2].set_xscale('symlog')
axes[2].set_yscale('symlog')
axes[2].set_title('Cost vs Size (color = structures destroyed)')
plt.colorbar(axes[2].collections[0], ax=axes[2])

plt.tight_layout()
plt.show()

### 3 Calculate personnel efficiency metrics

ics_pnw['personnel_per_1000acres'] = (ics_pnw['TOTAL_PERSONNEL'] / 
                                       (ics_pnw['ACRES'] / 1000)).replace([np.inf, -np.inf], np.nan)
ics_pnw['cost_per_acre'] = (ics_pnw['EST_IM_COST_TO_DATE'] / 
                             ics_pnw['ACRES']).replace([np.inf, -np.inf], np.nan)

efficiency_metrics = ics_pnw.groupby('FIRE_EVENT_ID').agg({
    'ACRES': 'max',
    'TOTAL_PERSONNEL': 'max',
    'EST_IM_COST_TO_DATE': 'max',
    'STR_DESTROYED': 'max',
    'STR_THREATENED': 'max',
    'days_since_discovery': 'max'
}).reset_index()

efficiency_metrics['personnel_per_1000acres'] = (efficiency_metrics['TOTAL_PERSONNEL'] / 
                                                   (efficiency_metrics['ACRES'] / 1000))
efficiency_metrics['cost_per_acre'] = (efficiency_metrics['EST_IM_COST_TO_DATE'] / 
                                        efficiency_metrics['ACRES'])

efficiency_metrics['structures_at_risk'] = efficiency_metrics['STR_THREATENED'] > 0

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.boxplot(data=efficiency_metrics[efficiency_metrics['personnel_per_1000acres'] < 100], 
            x='structures_at_risk', y='personnel_per_1000acres', ax=axes[0])
axes[0].set_title('Personnel Density: Structures at Risk vs Not')
axes[0].set_ylabel('Personnel per 1000 acres')

sns.boxplot(data=efficiency_metrics[efficiency_metrics['cost_per_acre'] < 10000], 
            x='structures_at_risk', y='cost_per_acre', ax=axes[1])
axes[1].set_title('Cost per Acre: Structures at Risk vs Not')
axes[1].set_ylabel('Cost per Acre ($)')

plt.tight_layout()
plt.show()

print(f"Median personnel (structures at risk): {efficiency_metrics[efficiency_metrics['structures_at_risk']]['personnel_per_1000acres'].median():.1f}")
print(f"Median personnel (no structures): {efficiency_metrics[~efficiency_metrics['structures_at_risk']]['personnel_per_1000acres'].median():.1f}")

### 4

impact_data = ics_pnw.groupby('FIRE_EVENT_ID').agg({
    'ACRES': 'max',
    'NUM_EVACUATED': 'max',
    'STR_DESTROYED': 'max',
    'STR_DAMAGED': 'max',
    'STR_THREATENED': 'max',
    'TOTAL_PERSONNEL': 'max',
    'FATALITIES': 'max',
    'INJURIES': 'max',
    'POO_STATE': 'first'
}).reset_index()

impact_data['had_evacuation'] = impact_data['NUM_EVACUATED'] > 0
impact_data['had_structure_loss'] = impact_data['STR_DESTROYED'] > 0

threat_vs_loss = impact_data[impact_data['STR_THREATENED'] > 0].copy()
threat_vs_loss['loss_rate'] = (threat_vs_loss['STR_DESTROYED'] / 
                                threat_vs_loss['STR_THREATENED'])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].scatter(impact_data['ACRES'], impact_data['NUM_EVACUATED'], alpha=0.5)
axes[0,0].set_xlabel('Fire Size (acres)')
axes[0,0].set_ylabel('People Evacuated')
axes[0,0].set_xscale('log')
axes[0,0].set_yscale('log')
axes[0,0].set_title('Evacuations vs Fire Size')

axes[0,1].scatter(threat_vs_loss['STR_THREATENED'], threat_vs_loss['loss_rate'], alpha=0.5)
axes[0,1].set_xlabel('Structures Threatened')
axes[0,1].set_ylabel('Loss Rate (destroyed/threatened)')
axes[0,1].set_xscale('log')
axes[0,1].set_title('Structure Loss Rate')

evac_fires = impact_data[impact_data['had_evacuation']].copy()
axes[1,0].scatter(evac_fires['NUM_EVACUATED'], evac_fires['TOTAL_PERSONNEL'], alpha=0.5)
axes[1,0].set_xlabel('People Evacuated')
axes[1,0].set_ylabel('Peak Personnel')
axes[1,0].set_xscale('log')
axes[1,0].set_yscale('log')
axes[1,0].set_title('Personnel Response to Evacuations')

state_impacts = impact_data.groupby('POO_STATE').agg({
    'had_evacuation': 'mean',
    'had_structure_loss': 'mean',
    'FIRE_EVENT_ID': 'count'
}).reset_index()
state_impacts.columns = ['STATE', 'pct_with_evacuation', 'pct_with_structure_loss', 'fire_count']

x = np.arange(len(state_impacts))
width = 0.35
axes[1,1].bar(x - width/2, state_impacts['pct_with_evacuation'], width, label='Had Evacuation')
axes[1,1].bar(x + width/2, state_impacts['pct_with_structure_loss'], width, label='Had Structure Loss')
axes[1,1].set_xticks(x)
axes[1,1].set_xticklabels(state_impacts['STATE'])
axes[1,1].set_ylabel('Proportion of Fires')
axes[1,1].set_title('Impact Frequency by State')
axes[1,1].legend()

plt.tight_layout()
plt.show()

print(f"\nTotal fires with evacuations: {impact_data['had_evacuation'].sum()}")
print(f"Total fires with structure loss: {impact_data['had_structure_loss'].sum()}")
print(f"Average loss rate when structures threatened: {threat_vs_loss['loss_rate'].mean():.2%}")

### 5

ics_pnw_valid = ics_pnw[ics_pnw['CY'].notna()].copy()
ics_pnw_valid['year'] = ics_pnw_valid['CY'].astype(int)
ics_pnw_valid['date'] = pd.to_datetime(ics_pnw_valid['year'].astype(str) + '-07-01')

print(f"Valid date records: {len(ics_pnw_valid)} out of {len(ics_pnw)}")
print(f"Date range: {ics_pnw_valid['date'].min()} to {ics_pnw_valid['date'].max()}")

daily_demand = ics_pnw_valid.groupby('year').agg({
    'TOTAL_PERSONNEL': 'sum',
    'FIRE_EVENT_ID': 'nunique',
    'ACRES': 'sum',
    'EST_IM_COST_TO_DATE': 'sum'
}).reset_index()

daily_demand.columns = ['year', 'total_personnel', 'active_fires', 'total_acres', 'cumulative_cost']

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
daily_demand.plot(x='year', y='total_personnel', ax=ax[0], title='Total Personnel per Year')
daily_demand.plot(x='year', y='active_fires', ax=ax[1], title='Number of Fires per Year')
plt.tight_layout()
plt.show()

### 6

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# use early reports to predict final cost
cost_prediction_data = []

for fire_id in ics_pnw['FIRE_EVENT_ID'].unique():
    fire_reports = ics_pnw[ics_pnw['FIRE_EVENT_ID'] == fire_id].sort_values('REPORT_TO_DATE')
    
    if len(fire_reports) < 2:
        continue
    
    early_report = fire_reports.iloc[0] 
    final_report = fire_reports.iloc[-1]  
    
    cost_prediction_data.append({
        'initial_acres': early_report['ACRES'],
        'initial_personnel': early_report['TOTAL_PERSONNEL'],
        'structures_threatened': early_report['STR_THREATENED'],
        'initial_cost': early_report['EST_IM_COST_TO_DATE'],
        'state': early_report['POO_STATE'],
        'month': pd.to_datetime(early_report['DISCOVERY_DATE']).month,
        'final_cost': final_report['PROJECTED_FINAL_IM_COST']
    })

cost_df = pd.DataFrame(cost_prediction_data)
cost_df = pd.get_dummies(cost_df, columns=['state'], drop_first=True)
cost_df = cost_df.dropna(subset=['final_cost'])
cost_df = cost_df[cost_df['final_cost'] > 0]

features = [col for col in cost_df.columns if col != 'final_cost']
X = cost_df[features].fillna(0)
y = np.log1p(cost_df['final_cost']) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_cost = RandomForestRegressor(n_estimators=100, random_state=42)
rf_cost.fit(X_train, y_train)

y_pred = rf_cost.predict(X_test)

print(f"Cost Prediction Model R²: {r2_score(y_test, y_pred):.3f}")
print(f"MAE (log scale): {mean_absolute_error(y_test, y_pred):.3f}")

importance_df = pd.DataFrame({
    'feature': features,
    'importance': rf_cost.feature_importances_
}).sort_values('importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x='importance', y='feature')
plt.title('Top 10 Predictors of Final Fire Cost')
plt.tight_layout()
plt.show()










###

### TERRA WS DATA

### Extract wind speed data for fire locations

def extract_wind_speed_for_fire(fire_row, year):
    """Extract mean wind speed for a fire location and time"""
    try:
        nc_file = f"data/terra/ws/TerraClimate_ws_{year}.nc"
        ds = xr.open_dataset(nc_file)
        
        if fire_row.geometry.geom_type == 'Point':
            lon, lat = fire_row.geometry.x, fire_row.geometry.y
        else:
            centroid = fire_row.geometry.centroid
            lon, lat = centroid.x, centroid.y
        
        month = fire_row['Ig_Date'].month if pd.notna(fire_row['Ig_Date']) else 7
        
        ws_data = ds.sel(lon=lon, lat=lat, method='nearest')

        ws_value = ws_data['ws'].isel(time=month-1).values
        
        ds.close()
        return float(ws_value)
    except Exception as e:
        return np.nan

fires_climate_geo = fires_climate.to_crs(epsg=4326)

print("Extracting wind speed data for fires...")
fires_climate_geo['wind_speed'] = fires_climate_geo.apply(
    lambda row: extract_wind_speed_for_fire(row, int(row['YEAR'])), 
    axis=1
)

print(f"Extracted wind speed for {fires_climate_geo['wind_speed'].notna().sum()} fires")
print(f"Wind speed range: {fires_climate_geo['wind_speed'].min():.2f} to {fires_climate_geo['wind_speed'].max():.2f} m/s")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# fire size vs wind speed
axes[0].scatter(fires_climate_geo['wind_speed'], fires_climate_geo['area_ha'], 
                alpha=0.5, s=20, c=fires_climate_geo['STATE'].map({'WA': 0, 'OR': 1, 'ID': 2}),
                cmap='viridis')
axes[0].set_xlabel('Wind Speed (m/s)')
axes[0].set_ylabel('Fire Size (ha)')
axes[0].set_yscale('log')
axes[0].set_title('Fire Size vs Wind Speed')
axes[0].grid(True, alpha=0.3)

# ws by state
sns.boxplot(data=fires_climate_geo[fires_climate_geo['wind_speed'].notna()], 
            x='STATE', y='wind_speed', ax=axes[1])
axes[1].set_title('Wind Speed Distribution by State')
axes[1].set_ylabel('Wind Speed (m/s)')

# combined climate factors
scatter = axes[2].scatter(fires_climate_geo['tmean'], fires_climate_geo['wind_speed'],
                         c=np.log1p(fires_climate_geo['area_ha']), 
                         cmap='YlOrRd', alpha=0.6, s=30)
axes[2].set_xlabel('Mean Temperature (°C)')
axes[2].set_ylabel('Wind Speed (m/s)')
axes[2].set_title('Fire Size by Temperature and Wind')
plt.colorbar(scatter, ax=axes[2], label='log(Fire Size ha)')

plt.tight_layout()
plt.show()

climate_vars = ['tmean', 'ppt', 'wind_speed', 'area_ha']
corr_matrix = fires_climate_geo[climate_vars].corr()
print("\nClimate Variable Correlations:")
print(corr_matrix)

fires_climate['wind_speed'] = fires_climate_geo['wind_speed'].values










### Analyze seasonal wind patterns and fire occurrence

def get_monthly_wind_climatology(year_start=2000, year_end=2024):
    """Calculate monthly wind speed climatology for PNW"""

    pnw_bounds = {
        'lon_min': -125, 'lon_max': -110,
        'lat_min': 42, 'lat_max': 49
    }
    
    monthly_data = []
    
    for year in range(year_start, year_end + 1):
        try:
            nc_file = f"data/terra/ws/TerraClimate_ws_{year}.nc"
            ds = xr.open_dataset(nc_file)

            ds_pnw = ds.sel(
                lon=slice(pnw_bounds['lon_min'], pnw_bounds['lon_max']),
                lat=slice(pnw_bounds['lat_min'], pnw_bounds['lat_max'])
            )

            for month in range(12):
                ws_mean = ds_pnw['ws'].isel(time=month).mean().values
                monthly_data.append({
                    'year': year,
                    'month': month + 1,
                    'wind_speed': float(ws_mean)
                })
            
            ds.close()
            print(f"Processed {year}")
        except Exception as e:
            print(f"Error processing {year}: {e}")
            continue
    
    return pd.DataFrame(monthly_data)

print("Calculating wind speed climatology...")
wind_climatology = get_monthly_wind_climatology(2000, 2024)

fires_climate['month'] = fires_climate['Ig_Date'].dt.month
fire_counts = fires_climate.groupby(['YEAR', 'month']).agg({
    'Event_ID': 'count',
    'area_ha': 'sum'
}).reset_index()
fire_counts.columns = ['year', 'month', 'fire_count', 'total_area']

seasonal_analysis = wind_climatology.merge(
    fire_counts, 
    on=['year', 'month'],
    how='left'
).fillna(0)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# avg wind speed by month
monthly_avg_wind = seasonal_analysis.groupby('month')['wind_speed'].mean()
axes[0, 0].bar(range(1, 13), monthly_avg_wind, color='steelblue')
axes[0, 0].set_xlabel('Month')
axes[0, 0].set_ylabel('Average Wind Speed (m/s)')
axes[0, 0].set_title('Average Wind Speed by Month (2000-2024)')
axes[0, 0].set_xticks(range(1, 13))
axes[0, 0].set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
axes[0, 0].grid(True, alpha=0.3, axis='y')

# fire count by month
monthly_fires = seasonal_analysis.groupby('month')['fire_count'].sum()
axes[0, 1].bar(range(1, 13), monthly_fires, color='orangered')
axes[0, 1].set_xlabel('Month')
axes[0, 1].set_ylabel('Total Fire Count')
axes[0, 1].set_title('Fire Occurrence by Month (2000-2024)')
axes[0, 1].set_xticks(range(1, 13))
axes[0, 1].set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'])
axes[0, 1].grid(True, alpha=0.3, axis='y')

# ws vs fire count by month
axes[1, 0].scatter(seasonal_analysis['wind_speed'], 
                   seasonal_analysis['fire_count'],
                   alpha=0.5, s=30, c=seasonal_analysis['month'], cmap='twilight')
axes[1, 0].set_xlabel('Wind Speed (m/s)')
axes[1, 0].set_ylabel('Fire Count')
axes[1, 0].set_title('Monthly Wind Speed vs Fire Count')
axes[1, 0].grid(True, alpha=0.3)

valid_data = seasonal_analysis[seasonal_analysis['fire_count'] > 0]
if len(valid_data) > 0:
    corr = valid_data[['wind_speed', 'fire_count']].corr().iloc[0, 1]
    axes[1, 0].text(0.05, 0.95, f'r = {corr:.3f}', 
                    transform=axes[1, 0].transAxes, 
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# series of wind speed and fires (annual aggregation for clarity)
annual_data = seasonal_analysis.groupby('year').agg({
    'wind_speed': 'mean',
    'fire_count': 'sum'
}).reset_index()

ax4a = axes[1, 1]
ax4b = ax4a.twinx()

ax4a.plot(annual_data['year'], annual_data['wind_speed'], 
          color='steelblue', marker='o', linewidth=2, label='Wind Speed')
ax4b.plot(annual_data['year'], annual_data['fire_count'], 
          color='orangered', marker='s', linewidth=2, label='Fire Count')

ax4a.set_xlabel('Year')
ax4a.set_ylabel('Mean Wind Speed (m/s)', color='steelblue')
ax4b.set_ylabel('Fire Count', color='orangered')
ax4a.tick_params(axis='y', labelcolor='steelblue')
ax4b.tick_params(axis='y', labelcolor='orangered')
axes[1, 1].set_title('Annual Wind Speed and Fire Activity')
ax4a.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\nSeasonal Wind Statistics:")
print(seasonal_analysis.groupby('month')[['wind_speed', 'fire_count']].mean())












### Identify extreme wind events and their fire impacts

wind_climatology['wind_percentile'] = wind_climatology.groupby('month')['wind_speed'].rank(pct=True)

extreme_wind = wind_climatology[wind_climatology['wind_percentile'] > 0.90].copy()

print(f"Extreme wind months identified: {len(extreme_wind)}")

fires_climate['year_month'] = (fires_climate['YEAR'].astype(str) + '-' + 
                                fires_climate['month'].astype(str).str.zfill(2))
extreme_wind['year_month'] = (extreme_wind['year'].astype(str) + '-' + 
                               extreme_wind['month'].astype(str).str.zfill(2))

fires_climate['extreme_wind_period'] = fires_climate['year_month'].isin(extreme_wind['year_month'])

print(f"Fires during extreme wind: {fires_climate['extreme_wind_period'].sum()}")
print(f"Fires during normal wind: {(~fires_climate['extreme_wind_period']).sum()}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

fire_size_comparison = fires_climate[fires_climate['area_ha'] > 0].copy()
sns.boxplot(data=fire_size_comparison, x='extreme_wind_period', y='area_ha', ax=axes[0, 0])
axes[0, 0].set_yscale('log')
axes[0, 0].set_xticklabels(['Normal Winds', 'Extreme Winds'])
axes[0, 0].set_title('Fire Size: Normal vs Extreme Wind Periods')
axes[0, 0].set_ylabel('Fire Size (ha)')

normal_mean = fire_size_comparison[~fire_size_comparison['extreme_wind_period']]['area_ha'].mean()
extreme_mean = fire_size_comparison[fire_size_comparison['extreme_wind_period']]['area_ha'].mean()
axes[0, 0].text(0.5, 0.95, f'Normal: {normal_mean:.0f} ha\nExtreme: {extreme_mean:.0f} ha',
                transform=axes[0, 0].transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

wind_counts = fires_climate.groupby('extreme_wind_period').size()
axes[0, 1].bar(['Normal Winds', 'Extreme Winds'], wind_counts.values, 
               color=['steelblue', 'orangered'])
axes[0, 1].set_ylabel('Fire Count')
axes[0, 1].set_title('Number of Fires by Wind Condition')
for i, v in enumerate(wind_counts.values):
    axes[0, 1].text(i, v, str(v), ha='center', va='bottom')

monthly_extreme = fires_climate.groupby(['month', 'extreme_wind_period']).size().unstack(fill_value=0)
monthly_extreme.plot(kind='bar', ax=axes[1, 0], color=['steelblue', 'orangered'], width=0.8)
axes[1, 0].set_xlabel('Month')
axes[1, 0].set_ylabel('Fire Count')
axes[1, 0].set_title('Fire Occurrence by Month and Wind Condition')
axes[1, 0].legend(['Normal Winds', 'Extreme Winds'])
axes[1, 0].set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], rotation=0)
axes[1, 0].grid(True, alpha=0.3, axis='y')

state_wind = fires_climate.groupby(['STATE', 'extreme_wind_period']).size().unstack(fill_value=0)
state_wind.plot(kind='bar', ax=axes[1, 1], color=['steelblue', 'orangered'], width=0.7)
axes[1, 1].set_ylabel('Fire Count')
axes[1, 1].set_title('Fire Occurrence by State and Wind Condition')
axes[1, 1].legend(['Normal Winds', 'Extreme Winds'])
axes[1, 1].set_xticklabels(axes[1, 1].get_xticklabels(), rotation=0)
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

from scipy.stats import mannwhitneyu

normal_wind_fires = fires_climate[~fires_climate['extreme_wind_period']]['area_ha'].dropna()
extreme_wind_fires = fires_climate[fires_climate['extreme_wind_period']]['area_ha'].dropna()

normal_wind_fires = normal_wind_fires[normal_wind_fires > 0]
extreme_wind_fires = extreme_wind_fires[extreme_wind_fires > 0]

if len(normal_wind_fires) > 0 and len(extreme_wind_fires) > 0:
    stat, p_value = mannwhitneyu(extreme_wind_fires, normal_wind_fires, alternative='greater')
    print(f"\n=== Statistical Comparison ===")
    print(f"Mann-Whitney U test (are extreme wind fires larger?):")
    print(f"  p-value: {p_value:.4f}")
    print(f"  Mean size (normal winds): {normal_wind_fires.mean():.1f} ha")
    print(f"  Mean size (extreme winds): {extreme_wind_fires.mean():.1f} ha")
    print(f"  Median size (normal winds): {normal_wind_fires.median():.1f} ha")
    print(f"  Median size (extreme winds): {extreme_wind_fires.median():.1f} ha")
    print(f"  Ratio of means: {extreme_wind_fires.mean() / normal_wind_fires.mean():.2f}x")












### Create a climate risk index combining temperature, precipitation, and wind

from sklearn.preprocessing import StandardScaler

climate_features = fires_climate[['tmean', 'ppt', 'wind_speed', 'area_ha']].dropna()

scaler = StandardScaler()
climate_scaled = scaler.fit_transform(climate_features[['tmean', 'ppt', 'wind_speed']])

# Risk index: high temp + low precip + high wind = high risk
# Invert precipitation (low precip = high risk)
climate_features['climate_risk_index'] = (
    climate_scaled[:, 0] +  # high temperature
    (-climate_scaled[:, 1]) +  # low precipitation
    climate_scaled[:, 2]  # high wind
) / 3

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].hist(climate_features['climate_risk_index'], bins=50, 
                color='coral', edgecolor='black')
axes[0, 0].set_xlabel('Climate Risk Index')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Distribution of Climate Risk Index')
axes[0, 0].axvline(0, color='red', linestyle='--', label='Neutral')
axes[0, 0].legend()

axes[0, 1].scatter(climate_features['climate_risk_index'], 
                   climate_features['area_ha'],
                   alpha=0.5, s=20, c=climate_features['climate_risk_index'],
                   cmap='RdYlBu_r')
axes[0, 1].set_xlabel('Climate Risk Index')
axes[0, 1].set_ylabel('Fire Size (ha)')
axes[0, 1].set_yscale('log')
axes[0, 1].set_title('Fire Size vs Climate Risk Index')
axes[0, 1].grid(True, alpha=0.3)

corr = climate_features[['climate_risk_index', 'area_ha']].corr().iloc[0, 1]
axes[0, 1].text(0.05, 0.95, f'r = {corr:.3f}', 
                transform=axes[0, 1].transAxes, va='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

climate_features['risk_category'] = pd.cut(
    climate_features['climate_risk_index'],
    bins=[-np.inf, -0.5, 0.5, np.inf],
    labels=['Low Risk', 'Moderate Risk', 'High Risk']
)

sns.boxplot(data=climate_features, x='risk_category', y='area_ha', ax=axes[1, 0])
axes[1, 0].set_yscale('log')
axes[1, 0].set_ylabel('Fire Size (ha)')
axes[1, 0].set_title('Fire Size by Climate Risk Category')
axes[1, 0].grid(True, alpha=0.3, axis='y')

from mpl_toolkits.mplot3d import Axes3D
ax3d = fig.add_subplot(224, projection='3d')
scatter = ax3d.scatter(climate_features['tmean'], 
                       climate_features['wind_speed'],
                       climate_features['ppt'],
                       c=np.log1p(climate_features['area_ha']),
                       cmap='YlOrRd', alpha=0.6, s=20)
ax3d.set_xlabel('Temperature (°C)')
ax3d.set_ylabel('Wind Speed (m/s)')
ax3d.set_zlabel('Precipitation (mm)')
ax3d.set_title('Climate Space Colored by Fire Size')
plt.colorbar(scatter, ax=ax3d, label='log(Fire Size)')

plt.tight_layout()
plt.show()

print("\n=== Climate Risk Statistics ===")
print(climate_features.groupby('risk_category')['area_ha'].describe())