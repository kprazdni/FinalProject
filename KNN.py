import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#download data for SPY and 9 sector ETFS
tickers = ["SPY", "XLE", "XLU", "XLK", "XLB", "XLP", "XLY", "XLI", "XLV", "XLF"]
data = yf.download(tickers, start="2010-01-01", end="2019-12-31")
prices = DataFrame(data["Adj Close"])
log_daily_returns = np.log(prices) - np.log(prices.shift(1))
returns = log_daily_returns.iloc[1:]
returns_model = returns.copy()

# Export table of all adjusted close prices to csv
#prices.to_csv(r'C:\Users\Class2020\final_data.csv', header=True)

# Plot all prices on one graph
prices.plot()
plt.xlabel("Date")
plt.ylabel("Adjusted Price")
plt.title("All ETF Prices")
plt.show()

# Define predictors and target
predictors = returns_model[["XLE", "XLU", "XLK", "XLB", "XLP", "XLY", "XLI", "XLV", "XLF"]]
target = np.asarray(returns_model['SPY'], dtype="|S6")

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(predictors, target, test_size=0.2, random_state=6)

# Make list of accuracy rates and plot to see which k to choose
k_range = range(1,26)
scores = {}
scores_list = []
for k in k_range :
    # Create KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=k)
    # Train the model using the training set
    knn_model = knn.fit(X_train, y_train)
    # Predict the response for test set
    y_pred = knn_model.predict(X_test)
    # Model Accuracy
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))

plt.plot(k_range, scores_list)
plt.xlabel("Value of k for KNN")
plt.ylabel("Testing Accuracy")
plt.show()

# Retrain model for k=4
knn = KNeighborsClassifier(n_neighbors=4)
knn_model1 = knn.fit(X_train, y_train)
y_pred = knn_model1.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
# Accuracy: 0.16500994035785288

