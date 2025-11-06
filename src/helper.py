def prism_raster_paths(base_dir, variable, years=range(2000, 2025)):
    paths = {}
    for y in years:
        if variable == 'tmean':
            paths[y] = f"{base_dir}/mean_temp/annual/prism_tmean_us_25m_{y}/prism_tmean_us_25m_{y}.tif"
        elif variable == 'ppt':
            paths[y] = f"{base_dir}/precip/annual/prism_ppt_us_25m_{y}/prism_ppt_us_25m_{y}.tif"
    return paths

prism_paths = {
    'tmean': prism_raster_paths("data/prism", 'tmean'),
    'ppt': prism_raster_paths("data/prism", 'ppt')
}