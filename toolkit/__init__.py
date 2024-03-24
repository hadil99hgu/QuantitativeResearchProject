# __init__.py for toolkit package

# Import necessary modules from the package
from .tools import detect_and_plot_outliers, detect_and_plot_outliers_zscore,filter_outliers, regression_data,poly_regression

# Define what is available to import from the package directly
__all__ = ['detect_and_plot_outliers', 'detect_and_plot_outliers_zscore', 'filter_outliers', 'regression_data','poly_regression']
