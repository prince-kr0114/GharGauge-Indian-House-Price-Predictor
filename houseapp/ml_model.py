from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

import pandas as pd
from pathlib import Path



BASE_DIR = Path(__file__).resolve().parent.parent

data = pd.read_csv(BASE_DIR / "house_price_india.csv")

X = data[[
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "street",
    "city",
    "statezip",
    "country"
]]


y = data["price"]


col = [
    "street",
    "city",
    "statezip",
    "country"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            col
        )
    ],
    remainder="passthrough"
)


model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),

    (
        "regressor",
        RandomForestRegressor(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=2,
            random_state=42
        )
    )
])


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)


model.fit(X_train, y_train)

predict = model.predict(X_test)

r2 = r2_score(y_test, predict)

print("R2 Score:", round(r2, 4))


new_house = pd.DataFrame([{
    "bedrooms": 5,
    "bathrooms": 2,
    "sqft_living": 3008,
    "sqft_lot": 5000,
    "floors": 2,
    "waterfront": 0,
    "view": 0,
    "condition": 3,
    "sqft_above": 2000,
    "sqft_basement": 1008,
    "yr_built": 2010,
    "yr_renovated": 0,

    # India example
    "street": "Sector 62",
    "city": "Noida",
    "statezip": "UP 201309",
    "country": "India"
}])


result = model.predict(new_house)


print(
    "Predicted House Price:",
    round(result[0], 2),
    "lakhs."
)



def predict_house(data):
    """
    data = dictionary containing house details
    """

    new_house = pd.DataFrame([data])

    result = model.predict(new_house)

    return result[0]