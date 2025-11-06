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

### FIGURE 1

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