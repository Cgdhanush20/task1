import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']
target = 'SalePrice'
numeric_cols = train_df.select_dtypes(include=['number']).columns
train_df[numeric_cols] = train_df[numeric_cols].fillna(train_df[numeric_cols].median())
test_df[numeric_cols.intersection(test_df.columns)] = test_df[numeric_cols.intersection(test_df.columns)].fillna(test_df[numeric_cols.intersection(test_df.columns)].median())
train_df = train_df[features + [target]].dropna()
test_df = test_df[features].dropna()
X = train_df[features]
y = train_df[target]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)
mse = mean_squared_error(y_val, y_pred)
rmse = mse ** 0.5
print(f'Root Mean Squared Error: {rmse}')
X_test = test_df[features]
test_preds = model.predict(X_test)
submission_df = pd.DataFrame({
    'Id': test_df.index,
    'SalePrice': test_preds
})
submission_df.to_csv('submission.csv', index=False)
