# Model card

## Project context

Short recap of the project context.

## Data

# Model Card
This project involves building a predictive model for real estate prices. The goal is to predict the price of a property based on various features such as property type, location, state of the building, and more. This model can be used by real estate companies to estimate property prices and make informed decisions.

## Data
The input dataset is a CSV file named `properties.csv`. The target variable is `price`. The features used in the model are:

property_type: The type of the property (e.g., house, apartment).
subproperty_type: More specific type of the property (e.g., villa, penthouse).
region: The region where the property is located.
province: The province where the property is located.
locality: The locality or city where the property is located.
equipped_kitchen: Whether the property has an equipped kitchen (yes/no).
state_building: The state of the building (e.g., new, good, to be renovated).
epc: Energy performance certificate rating.
heating_type: The type of heating in the property (e.g., gas, electric).
construction_year: The year the property was constructed.
nbr_frontages: The number of frontages of the property.
nbr_bedrooms: The number of bedrooms in the property.
latitude: The latitude coordinate of the property.
longitude: The longitude coordinate of the property.
total_area_sqm: The total area of the property in square meters.
surface_land_sqm: The surface area of the land of the property in square meters.
terrace_sqm: The area of the terrace of the property in square meters.
garden_sqm: The area of the garden of the property in square meters.
primary_energy_consumption_sqm: The primary energy consumption of the property per square meter.
cadastral_income: The cadastral income of the property.
fl_furnished: Whether the property is furnished (1 for yes, 0 for no).
fl_open_fire: Whether the property has an open fire (1 for yes, 0 for no).
fl_terrace: Whether the property has a terrace (1 for yes, 0 for no).
fl_garden: Whether the property has a garden (1 for yes, 0 for no).
fl_swimming_pool: Whether the property has a swimming pool (1 for yes, 0 for no).
fl_floodzone: Whether the property is in a flood zone (1 for yes, 0 for no).
fl_double_glazing: Whether the property has double glazing windows (1 for yes, 0 for no).

## Model Details
Several models were tested during the development process. The final model chosen was Linear Regression and Random Forests.

## Performance
The performance of the models was evaluated using the R² score achieving 0.80 on the training set and 0.68 on the test set.

## Limitations
The model has several limitations. First, it assumes a linear relationship between the features and the target variable, which may not hold true for all properties. Second, the model may not perform well on properties that are significantly different from those in the training data. Finally, the model does not account for temporal trends or geographical differences in property prices.

## Usage
The model requires Python 3.6+ and the following Python libraries: pandas, numpy, scikit-learn, and matplotlib. The `train.py` script can be used to train the model and the `predict.py` script can be used to generate predictions. To train the model, run `python train.py` in the terminal. To generate predictions, run `python predict.py` in the terminal.

## Maintainers
For any questions or issues, please contact the maintainers:
- Omar Hamdy: omar@becode.com