
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def filter_outliers(data, column_name):
    """
    Filters outliers from a dataset based on the Interquartile Range (IQR) method.

    Parameters:
    data (DataFrame): The dataset containing the data.
    column_name (str): The name of the column to filter for outliers.

    Returns:
    DataFrame: The filtered dataset without outliers.
    """
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1

    # Lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filter the data to exclude outliers
    filtered_data = data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]

    return filtered_data

def detect_and_plot_outliers(data, column_name, ax, show):
    """
    Detects outliers in the data and plots the data column along with the outliers marked in red.

    Parameters:
    data (DataFrame): The dataset containing the data.
    column_name (str): The name of the column to detect outliers in.
    ax (matplotlib.axes.Axes): The Axes object to plot on.
    show (bool): Flag to indicate whether to display the plot or not.

    Returns:
    DataFrame: A DataFrame containing the outliers.
    """
    Q1 = data[column_name].quantile(0.25)
    Q3 = data[column_name].quantile(0.75)
    IQR = Q3 - Q1

    # Lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Marking outliers in red
    outliers = data[(data[column_name] < lower_bound) | (data[column_name] > upper_bound)]
    if show:
        # Plot the curve
        ax.plot(outliers.index, outliers[column_name], 'ro', label='Outliers')
        ax.plot(data.index, data[column_name], label=column_name)
        
        ax.set_title(f'{column_name} Curve')
        ax.set_xlabel('Index')
        ax.set_ylabel(column_name)
        ax.legend()
    return outliers,lower_bound,upper_bound

def detect_and_plot_outliers_zscore(data, column_name, ax, window_size, show=True, z_threshold=3):
    """
    Detects outliers based on Z-scores using a rolling window and plots the column data and outliers.

    Parameters:
    data (DataFrame): The dataset containing the data.
    column_name (str): The name of the column to detect outliers in.
    ax (matplotlib.axes.Axes): The Axes object to plot on.
    window_size (int): The size of the rolling window for calculating Z-scores.
    show (bool): Whether to show the plot or not.
    z_threshold (float): The threshold of the Z-score to identify outliers.

    Returns:
    DataFrame: A DataFrame containing the outliers.
    """
    data1=data.copy()
    # Rolling window calculation for mean and standard deviation
    rolling_mean = data1[column_name].rolling(window=window_size).mean()
    rolling_std = data1[column_name].rolling(window=window_size).std()

    # Calculate Z-scores using the rolling mean and standard deviation
    data1['Rolling_Z-Score'] = (data1[column_name] - rolling_mean) / rolling_std

    # Identify outliers
    outliers = data1[(data1['Rolling_Z-Score'].abs() > z_threshold) & (data1['Rolling_Z-Score'].notnull())]

    if show:
        # Plot the data
        ax.plot(data.index, data[column_name], label=column_name)
        # Mark the outliers in red on the plot
        ax.plot(outliers.index, outliers[column_name], 'ro', label='Outliers')
        # Set the title and labels
        ax.set_title(f'{column_name} with Rolling Window Z-score Outliers')
        ax.set_xlabel('Index')
        ax.set_ylabel(column_name)
        ax.legend()
    return outliers

def regression_data(df, col1, col2, p, max_delay):
    """
    Prepares the dataset for regression analysis by creating shifted columns for time-series prediction and splitting into train and test sets.

    Parameters:
    df (DataFrame): The input dataframe containing the time series data.
    col1 (str): The name of the column to use for prediction (independent variable).
    col2 (str): The name of the column to be predicted (dependent variable).
    p (float): The proportion of data to be used as test data.
    max_delay (list): A list of integers representing the lag periods to include as features.

    Returns:
    tuple: A tuple containing the training and testing dataframes for X and y.
    """
    
    dfc = df.copy()
    # Preparing the data for regression by creating shifted columns
    l = []
    for i in max_delay:
        dfc['Shift-' + str(i)] = dfc[col1].shift(i)
        l.append('Shift-' + str(i))
    dfc = dfc.dropna()
    z = round(len(df) * p)
    train = dfc.iloc[:-z]
    test = dfc.iloc[-z:]
    # Add a constant term for regression equation
    X_train = sm.add_constant(train[l])
    y_train = train[col2]
    X_test = sm.add_constant(test[l])
    y_test = test[col2]
    return X_train, X_test, y_train, y_test

def poly_regression(X_train, X_test, y_train, y_test, degree, show=True):
    """
    Performs polynomial regression on the training set and evaluates it using the test set. Optionally plots the actual vs predicted values.

    Parameters:
    X_train (DataFrame): Training data features.
    X_test (DataFrame): Test data features.
    y_train (Series): Actual values for training.
    y_test (Series): Actual values for testing.
    degree (int): The degree of the polynomial features to be used.
    show (bool): If True, show a plot of the actual vs predicted values.

    Returns:
    tuple: A tuple containing the R-squared score and the predictions made by the model.
    """
 
    # Initialize a PolynomialFeatures object with the given degree
    poly = PolynomialFeatures(degree=degree)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    # Fit the OLS regression model using the transformed polynomial features
    model = sm.OLS(y_train, sm.add_constant(X_train_poly)).fit()
    # Make predictions using the model
    predictions = model.predict(sm.add_constant(X_test_poly))

    # Calculate R^2 score for the regression model's fit
    r2 = r2_score(y_test, predictions)
    if show:
        # Plot the actual vs predicted values
        print(f"R^2 Score: {r2}")
        plt.figure(figsize=(10, 5))
        plt.plot(X_test.index, y_test, label='Actual', color='blue')
        plt.plot(X_test.index, predictions, label='Predicted', color='red', linestyle='--')
        plt.title('Polynomial Regression: Actual vs Predicted')
        plt.xlabel('Index')
        plt.ylabel('Cumulative Log Return')
        plt.legend()
        plt.show()
    return r2, predictions

