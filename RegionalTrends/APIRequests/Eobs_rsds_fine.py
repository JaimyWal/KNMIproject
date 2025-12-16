import cdsapi

dataset = "insitu-gridded-observations-europe"
request = {
    "product_type": "ensemble_mean",
    "variable": ["surface_shortwave_downwelling_radiation"],
    "grid_resolution": "0_1deg",
    "period": "full_period",
    "version": ["31_0e"]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download('eobs_rsds_fine.zip')