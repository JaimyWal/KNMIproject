import cdsapi

dataset = "insitu-gridded-observations-europe"
request = {
    "product_type": "ensemble_mean",
    "variable": [
        "mean_temperature",
        "minimum_temperature",
        "maximum_temperature",
        "precipitation_amount",
        "sea_level_pressure",
        "surface_shortwave_downwelling_radiation",
        "relative_humidity",
        "wind_speed"
    ],
    "grid_resolution": "0_1deg",
    "period": "full_period",
    "version": ["31_0e"]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download('eobs_fine_all_data_full.zip')