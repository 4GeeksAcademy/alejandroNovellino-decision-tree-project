import os
from typing import TypedDict, NotRequired, Tuple

from IPython.core.display_functions import display
from dotenv import load_dotenv
from IPython.display import Markdown
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


# load the .env file variables
load_dotenv()


def db_connect() -> None:
    """
    Connects to the database using sqlalchemy
    """

    engine = create_engine(os.getenv("DATABASE_URL"))
    engine.connect()

    return


class DataLoadingError(Exception):
    """Custom exception raised when data loading fails."""


# noinspection SpellCheckingInspection
class ReadCsvParams(TypedDict):
    """
    Class for setting the parameters for reading a CSV file.

    Attributes:
        delimiter (str): Delimiter in the CSV file.
        nrows (Optional[int]): Number of rows of file to read.
    """

    delimiter: str
    nrows: NotRequired[int]


class SaveCsvParams(TypedDict):
    """
    Class for setting the parameters for saving a CSV file.

    Attributes:
        sep (str): Delimiter in the output CSV file.
    """

    sep: str


def load_data(
    file_path: str,
    url: str,
    read_csv_params: ReadCsvParams,
    save_csv_params: SaveCsvParams,
) -> pd.DataFrame:
    """
    Loads data from a file if it exists, otherwise from a URL.

    Args:
        file_path (str): The path to the file.
        url (str): The URL to load data from if the file doesn't exist.
        read_csv_params (ReadCsvParams): The parameters for reading the CSV file.
        save_csv_params (SaveCsvParams): The parameters for saving the CSV file.

    Returns:
        pandas.DataFrame: The loaded DataFrame.

    Raise:
        Exception: if no data could be loaded

    Examples:
        Data not saved before in local .csv file:

        > from utils import load_data
        > file_path = '../data/raw/AB_NYC_2019.csv'
        > url = 'https://raw.githubusercontent.com/data.csv'
        > df = load_data(file_path=file_path, url=url)

        File not found. Loading data from URL: https://raw.githubusercontent.com/data.csv
        Data saved to file: ../data/raw/AB_NYC_2019.csv

        Data have been saved before in local .csv file:

        > from utils import load_data
        > file_path = '../data/raw/AB_NYC_2019.csv'
        > url = 'https://raw.githubusercontent.com/data.csv'
        > df = load_data(file_path=file_path, url=url)

        Loading data from file: ../data/raw/AB_NYC_2019.csv
    """

    # verify if the file exists
    if os.path.exists(file_path):

        print(f"Loading data from file: {file_path}")

        # load the data form the file
        df = pd.read_csv(file_path, **read_csv_params)  # type: ignore

        # return the loaded dataframe form local file
        return df

    else:

        print(f"File not found. Loading data from URL: {url}")

        try:
            # file not found so try to get the data from the URL
            df = pd.read_csv(url, **read_csv_params)  # type: ignore

            # save the DataFrame to the file for future use
            df.to_csv(file_path, index=False, sep=save_csv_params["sep"])

            print(f"Data saved to file: {file_path}")

        except Exception as e:
            print(f"Error loading data from URL: {e}")
            raise DataLoadingError() from e

    return df


def split_my_data(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Wrapper for the sklearn tran_test function to have strict typing
    Splits data into training and testing sets with strict type hints.

    Args:
        x (DataFrame): The features (input data). Can be a Pandas DataFrame or a NumPy array.
        y (Series): The target (output data). Can be a Pandas Series or a NumPy array.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before splitting.

    Returns:
        A tuple containing X_train, X_test, y_train, and y_test.

    """

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state
    )

    return x_train, x_test, y_train, y_test


def draw_corr_matrix(corr: pd.DataFrame,) -> None:
    """
    Draw a correlation matrix using seaborn.

    Args:
        corr (DataFrame): The correlation matrix to draw.

    Returns:
        None

    """

    # generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # set up the matplotlib figure
    plt.subplots(figsize=(11, 9))

    # generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(
        corr,
        annot=True,
        mask=mask,
        cmap=cmap,
        vmax=.3,
        center=0,
        square=True,
        linewidths=.5,
        cbar_kws={"shrink": .5}
    )

    return None


def draw_confusion_matrix(confusion: pd.DataFrame) -> None:
    """
    Draw a confusion matrix using seaborn.

    Args:
        confusion (DataFrame): The confusion matrix to draw.

    Returns:
        None
    """

    # create the groups to display
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']
    group_counts = ['{0:0.0f}'.format(value) for value in confusion.flatten()]
    group_percentages = ['{0:.2%}'.format(value) for value in confusion.flatten() / np.sum(confusion)]


    # labels to display
    labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)

    plt.figure(figsize=(5, 5))

    sns.heatmap(confusion, annot=labels, fmt='')

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()
    plt.show()

    return None


def remove_outliers_iqr(df:pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Removes outliers based on the IQR method.

    Args:

    Returns:
    """

    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df_filtered = df[(df[column_name] >= lower_bound) & (df[column_name] <= upper_bound)]

    return df_filtered


def remove_outliers_zscore(df, column_name, threshold=3) -> pd.DataFrame:
    """
    Removes outliers based on the z-score method.

    Args:

    Returns:
    """

    z_scores = np.abs((df[column_name] - df[column_name].mean()) / df[column_name].std())

    df_filtered = df[z_scores < threshold]

    return df_filtered


def cap_outliers_iqr(df, column_name) -> pd.DataFrame:
    """
    Caps outliers based on the IQR method.

    Args:

    Returns:
    """

    q1 = df[column_name].quantile(0.25)
    q3 = df[column_name].quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    df[column_name] = np.where(df[column_name] < lower_bound, lower_bound, df[column_name])
    df[column_name] = np.where(df[column_name] > upper_bound, upper_bound, df[column_name])

    return df


def show_comparison_table(metric_names: list[str], default_metrics: list[float], optimized_metrics: list[float]) -> None:
    """
    Creates and show a Markdown table comparing default and optimized model metrics.

    Args:
        metric_names (list[str]): List of metric names.
        default_metrics (list[float]): List of metric values for the default model.
        optimized_metrics (list[float]): List of metric values for the optimized model.

    Returns:
        str: A Markdown table as a string.
    """

    if len(metric_names) != len(default_metrics) or len(metric_names) != len(optimized_metrics):
        return "Error: Metric lists must have the same length."

    markdown_table = "| Metric | Default Model | Optimized Model |\n"
    markdown_table += "|---|---|---|\n"

    for i in range(len(metric_names)):
        markdown_table += f"| {metric_names[i]} | {np.round(default_metrics[i], 2)} | {np.round(optimized_metrics[i], 2)} |\n"

    # display the table
    display(Markdown(markdown_table))

    return None


def draw_comparison_confusion_matrices(
        confusion_1: pd.DataFrame,
        confusion_2: pd.DataFrame,
        confusion_matrix_1_name: str,
        confusion_matrix_2_name: str,
) -> None:
    """
    Draw a confusion matrix using seaborn.

    Args:
        confusion_1 (DataFrame): The confusion matrix of the first model.
        confusion_2 (DataFrame): The confusion matrix of the second model.
        confusion_matrix_1_name (str): Label to put on the heatmap of the first confusion matrix.
        confusion_matrix_2_name (str): Label to put on the heatmap of the second confusion matrix.

    Returns:
        None
    """

    _, axis = plt.subplots(1, 2, figsize = (20, 7))

    # create the groups to display
    group_names = ['True Neg', 'False Pos', 'False Neg', 'True Pos']

    # first confusion values
    conf_1_group_counts = ['{0:0.0f}'.format(value) for value in confusion_1.flatten()]
    conf_1_group_percentages = ['{0:.2%}'.format(value) for value in confusion_1.flatten() / np.sum(confusion_1)]

    # second confusion values
    conf_2_group_counts = ['{0:0.0f}'.format(value) for value in confusion_2.flatten()]
    conf_2_group_percentages = ['{0:.2%}'.format(value) for value in confusion_2.flatten() / np.sum(confusion_2)]


    # labels to display of the first confusion matrix
    conf_1_labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, conf_1_group_counts, conf_1_group_percentages)]
    conf_1_labels = np.asarray(conf_1_labels).reshape(2, 2)

    # labels to display of the second confusion matrix
    conf_2_labels = [f'{v1}\n{v2}\n{v3}' for v1, v2, v3 in zip(group_names, conf_2_group_counts, conf_2_group_percentages)]
    conf_2_labels = np.asarray(conf_2_labels).reshape(2, 2)

    plt.figure(figsize=(10, 5))

    # first heatmap
    sns.heatmap(ax=axis[0], data=confusion_1, annot=conf_1_labels, fmt='').set(xlabel = f'{confusion_matrix_1_name} - True label', ylabel = 'Predicted label')
    # second heatmap
    sns.heatmap(ax=axis[1], data=confusion_2, annot=conf_2_labels, fmt='').set(xlabel = f'{confusion_matrix_2_name} - True label', ylabel = 'Predicted label')

    plt.tight_layout()
    plt.show()

    return None

