import geopandas as gpd

occ = gpd.read_file("data/mtbs_fod_pts_data/mtbs_FODpoints_DD.shp")
bnd = gpd.read_file("data/mtbs_perimeter_data/mtbs_perims_DD.shp")

occ["STATE"] = occ["Event_ID"].str[:2]
bnd["STATE"] = bnd["Event_ID"].str[:2]

pnw = ["WA", "OR", "ID"]
occ_pnw = occ[occ["STATE"].isin(pnw)]
bnd_pnw = bnd[bnd["STATE"].isin(pnw)]

occ_pnw.to_file("data/mtbs_fod_pts_data/mtbs_FODpoints_DD_pnw.shp")
bnd_pnw.to_file("data/mtbs_perimeter_data/mtbs_perims_DD_pnw.shp")

print(occ_pnw["STATE"].value_counts())
print(bnd_pnw["STATE"].value_counts())