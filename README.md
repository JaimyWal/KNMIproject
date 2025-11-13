# KNMIproject

## European Climate Data Analysis

This project analyzes European climate observations (E-OBS dataset) to visualize temperature and precipitation patterns and calculate regional climate trends.

### Features

- **Temperature and precipitation analysis** from E-OBS gridded observations
- **Linear trend calculations** for climate variables over decades
- **Publication-quality map visualizations** using Cartopy
- **Efficient data processing** with xarray and dask
- **Customizable regional and temporal selections**

### Project Structure

```
KNMIproject/
├── RegionalTrends/
│   ├── Eobs.py          # Main analysis script for E-OBS data
│   ├── Eobs_api.py      # Data download script using CDS API
│   └── PlotFigure.py    # Mapping and visualization functions
└── README.md
```

### Quick Start

The main analysis workflow:
1. Download E-OBS data using `Eobs_api.py`
2. Process and analyze data with `Eobs.py`
3. Generate maps using functions from `PlotFigure.py`

### Dependencies

- numpy
- xarray
- pandas
- matplotlib
- cartopy
- colormaps
- cmocean
- dask
- cdsapi (for data downloads)

---

## 🤖 Using GitHub Copilot Agent

**Have questions about what you can ask here?** See [COPILOT_HELP.md](COPILOT_HELP.md) for:
- What tasks I can help with
- Example questions you can ask
- Best practices for working together
- Understanding what this repository does

**Quick examples:**
- "Add error handling to the data loading function"
- "Create a function to calculate seasonal averages"
- "Add unit tests for the trend calculations"
- "Explain how the plot_map function works"
- "Optimize memory usage in data processing"