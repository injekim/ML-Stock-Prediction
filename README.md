# ML Stock Prediction
> Stock return prediction using machine learning methods.

## Data Source

Historical stock data for Apple Inc. from Yahoo Finance up until the 30th of May, 2023.

```python
df = yf.Ticker(ticker).history(period=period, prepost=True)
```

## Target
To predict whether or not the stock price will increase the following day.

```python
df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
```


## Stock History EDA Analysis

### Price
![Price/Volume plot](./images/EDA_0.png)
Based on the Open/Close plot and High/Low plot, the price of Apple's stock has been steadily increasing with some fluctuations. Also, the Volume chart shows a substantial increase in trading volume during the years 1998 and 2007. This can be attributed to significant events like the introduction of the iMac and the iPhone, as well as the overall growth of the technology sector during those periods.

### Daily price change
![Daily price diff. plot](./images/EDA_1.png)
Although the price difference increases as the stock price increases, the relative price change remains mostly consistent over the years.

### Other metrics
![Other plots](./images/EDA_2.png)

### Correlation
![Daily price diff. plot](./images/EDA_3.png)

### Pairplot against Target value
![Daily price diff. plot](./images/EDA_4.png)

## Training process

### Model

The GradientBoostingClassifier is chosen for its strength in capturing non-linear relationships, assessing feature importance, and robustness to outliers and missing data.

```python
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(random_state=0)
```

### Backtesting
Backtesting is used to more accurately evaluate the model's performance across diverse market conditions and minimise the impact of market anomalies in the test set.

```python
def backtest(model, X, y, init=1000, test=365, selector=None):
	preds = pd.DataFrame()
	for i in range(init, len(X), test):
		# Train test split
		X_train, X_test = X[:i], X[i:i+test]
		y_train, y_test = y[:i], y[i:i+test]
		close = X_test['Close'].copy()
		
		# Feature selection
		if selector is not None:
			selector.fit(X_train, y_train)
			selected = selector.get_support()
			X_train, X_test = X_train.iloc[:, selected], X_test.iloc[:, selected]
		
		# Fit model
		model.fit(X_train, y_train)

		temp = pd.concat([
			close,
			y_test,
			pd.DataFrame(model.predict(X_test), columns=['y_pred'], index=y_test.index),
			pd.DataFrame(model.predict_proba(X_test)[:,1], columns=['y_prob'], index=y_test.index)
		], axis=1)
		preds = pd.concat([preds, temp], axis=0)
	return preds
```

### Added features

To enhance the accuracy of the model, the following features have been added:

* SMA (Simple Moving Average): The moving average of close prices, which provides a smoothed representation of the stock's price trend over a specified time period.

* STD (Moving Standard Deviation): The moving standard deviation of close prices, which measures the volatility of the stock's price over a specific time period.

* Ratios of Various Features: Calculation of ratios such as High/Close, Low/Close, SMA_365/SMA_90, and others, which provide additional insights into the relationships between different aspects of the stock's performance.

* Last Dividend Payment Amount: The amount of the most recent dividend payment made by the company. Changes in dividend policy or payment amounts can influence market sentiment and stock prices.

* Days Since Last Dividend Payment: The number of days elapsed since the last dividend payment, which can capture the impact of dividend announcements on stock returns.

* Days Since Last Stock Split: The number of days since the last stock split occurred, which considers the effects of corporate actions on the stock's performance.

* Bollinger Bands: The Bollinger Bands indicator consists of a moving average and upper and lower bands that represent price volatility. It helps identify potential price breakouts or reversals.

* RSI (Relative Strength Index): A momentum oscillator that measures the speed and change of price movements. It indicates whether a stock is overbought or oversold, potentially signalling upcoming price reversals.

* MACD (Moving Average Convergence Divergence): A trend-following momentum indicator that calculates the relationship between two moving averages of a stock's price. It helps identify potential buy or sell signals.

* Trend: The sum of the target value over the last seven days. For example, if the stock price went up on five out of the last seven days, the trend value would be five.

### Feature selection

To further improve the model, scikit-learn's SequentialFeatureSelector is used to select the most relevant features.

```python
from sklearn.feature_selection import SequentialFeatureSelector
selector = SequentialFeatureSelector(clf, n_features_to_select='auto', n_jobs=-1)
results = backtest(clf, X, y, selector=selector)
```


## Results


## How to run
Run the notebooks in the following sequence:

    data_preparation.ipynb -> EDA_analysis.ipynb -> model.ipynb