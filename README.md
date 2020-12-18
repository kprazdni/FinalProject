# FinalProject
## Purpose: 

The purpose of this project is to predict the movement of the SPY using the SPDR ETFs through various machine learning algorithms. The movements are classified by an upward (+1), downward (-1) or neutral (0) movement. A movement of 0.5% or less would indicate a neutral movement. 

## Data Preparation:

Extract stock data for the following tickers using yfinance.download() from start date (2010-01-01) to end date (2019-12-31):

    SPY -  SPDR S&P 500 ETF Trust
   
    XLE - Energy Select Sector

    XLU - Utilities Select Sector

    XLK - Technology Select Sector

    XLB - Materials Select Sector

    XLP - Consumer Staples Select Sector
    
    XLY - Consumer Discretionary Select Sector

    XLI - Industrial Select Sector

    XLV - Health Care Select Sector

    XLF - Financial Select Sector

Select the Adjusted Close Price columns and export to a csv file named “SPY_data”.

## Algorithms and Functions:

### main.py 
main(): This function imports and runs the python files listed below

### KNN.py
performKNN(self) : reads in csv file; creates matrix of -1,0, and 1s based on negative, neutral or positive returns, performs the K-Nearest Neighbors algorithm for a range of k values  (1-26) to decide which k value results in the highest accuracy rate, and then runs a final model with the best k value; returns the Accuracy rate and Confusion matrix for the final model.

### LDAQDA.py
runldaqda(pd1) : Takes in “SPY_data.csv” as pd1 and converts it to neutral, positive, or negative change in price; also creates training and testing sets for the target data and the rest of the data

ldaqda(x_train, x_test, y_train, y_test) : Takes in training and testing data and performs LDA and QDA to find accuracy of test data; returns accuracy of LDA and QDA

ldatesting(x_train, x_test, y_train, y_test) : LDAtesting tries to find the highest accuracy for the LDA model by varying different shrinkage values with type lsqr and eigen; outputs table with all accuracies with shrinkage value and lists max accuracy and where it occurs

### TreeAndForestModels.py
DataLoader.getClassifiedData(filePath, threshold, lag) : read in the data csv, classify the data according to the threshold argument, and lag the class column according to the lag argument

ModelRunner.getData(filePath, threshold, lag) : create an instance of the DataLoader class, retrieve the data from it, and save training/testing sets for use in runModels()

ModelRunner.runModels() :  use the train-test splits from getData(...) to run and print out the scores/confusion matrix of the Decision Tree and Random Forest models

### SVM.py
svm_output(): Runs a SVM on the given csv file and returns an accuracy score and confusion matrix.
