import cdsapi

dataset = "reanalysis-era5-single-levels-monthly-means"
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": [
        "2m_temperature",
        "land_sea_mask"
    ],
    "year": ["2000"],
    "month": ["01"],
    "time": ["00:00"],
    "data_format": "netcdf",
    "download_format": "zip",
    "area": [90, -180, -90, 180]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download('era5_landmask.zip')