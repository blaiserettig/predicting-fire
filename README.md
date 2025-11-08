<object data="https://github.com/blaiserettig/predicting-fire/blob/6bd11031f1fadc95679b839c76e71544c5c6f518/report/370_Final.pdf" type="application/pdf" width="700px" height="700px">
    <embed src="https://github.com/blaiserettig/predicting-fire/blob/6bd11031f1fadc95679b839c76e71544c5c6f518/report/370_Final.pdf">
        <p>This browser does not support PDFs. Please download the PDF to view it: <a href="https://github.com/blaiserettig/predicting-fire/blob/6bd11031f1fadc95679b839c76e71544c5c6f518/report/370_Final.pdf">Download PDF</a>.</p>
    </embed>
</object>


# Independent Research Project as Final Project for CPSC-370


# Wildfire Impact Prediction Model

This module provides a machine learning model to predict downstream impacts of new wildfire events. The model predicts normalized indices (0-1 scale) for four impact categories:

1. **Firefighting Resource Demand** - Expected personnel and resource requirements
2. **Evacuation Risk** - Expected number of people to be evacuated
3. **Structure Threat** - Expected structures damaged or destroyed
4. **Supression Cost** - Expected cost to effectively supress the fire

## Input Parameters

### Required
- **fire_size_ha**: Fire size in hectares
- **fire_location**: Fire location as:
  - Dictionary: `{'latitude': float, 'longitude': float}`
  - Tuple: `(latitude, longitude)`

### Optional (but recommended for better* predictions)
- **drought_index**: DSCI (Drought Severity and Coverage Index) value
- **atmospheric_conditions**: Dictionary with:
  - `temperature`: Mean temperature in Celsius
  - `precipitation`: Total precipitation in mm
  - `wind_speed`: Wind speed in m/s
- **svi**: Social Vulnerability Index rank (0-1, higher = more vulnerable)
- **proximity_to_infrastructure**: Distance to nearest power plant in km
- **year**: Year of fire
- **month**: Month of fire (1-12)
- **state**: State code ('WA', 'OR', 'ID')

## Output Interpretation

All output indices are normalized to a 0-1 scale:
- **0.0**: Very low predicted impact
- **0.5**: Moderate predicted impact
- **1.0**: Very high predicted impact

## Model Architecture

- **Algorithm**: Gradient Boosting Regressor (scikit-learn)
- **Features**: 
  - Location (latitude, longitude, state)
  - Climate (temperature, precipitation, wind speed)
  - Drought (DSCI annual and monthly)
  - Infrastructure proximity
  - Social vulnerability (SVI)
  - Temporal (year, month with cyclical encoding)
  - Derived features (drought indices, ratios)
- **Preprocessing**: StandardScaler for features, MinMaxScaler for outputs
- **Training**: Separate models for each impact category

## Data Sources

The model uses data from:
- **MTBS**: Fire boundaries and areas
- **ICS209**: Personnel, evacuations, damages
- **DSCI**: Drought severity index
- **PRISM**: Temperature and precipitation
- **TerraClimate**: Wind speed
- **EIA**: Energy infrastructure locations
- **SVI**: Social Vulnerability Index

## Model Performance

After training, the model provides:
- R² scores for each impact category
- RMSE and MAE metrics
- Training vs test performance

~[image info](https://raw.githubusercontent.com/blaiserettig/predicting-fire/refs/heads/main/img/prediction_examples_improved.png)

Is the model any good?

**No!**

It's actually really bad. In fact, it does worse than just picking the average value in almost all cases. 

![image info](https://raw.githubusercontent.com/blaiserettig/predicting-fire/refs/heads/main/img/model_performance_improved.png)

The only thing it predicts marginally better than flipping a coin for is structure threat.

## Notes

- The model is trained on Pacific Northwest (WA, OR, ID) data
- Predictions are most accurate when all input parameters are provided
- Missing parameters will use default values (will reduce accuracy)

