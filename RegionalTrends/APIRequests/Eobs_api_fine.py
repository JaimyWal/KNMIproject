import cdsapi

dataset = "insitu-gridded-observations-europe"
request = {
    "product_type": "ensemble_mean",
    "variable": [
        "mean_temperature",
        "precipitation_amount"
    ],
    "grid_resolution": "0_1deg",
    "period": "full_period",
    "version": ["31_0e"]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download('eobs_data_fine.zip')