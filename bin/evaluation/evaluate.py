import os
import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import random
import click
import re

# Constants for regex patterns
LABEL_REGEX = r"label:\s*([\w\-]+)"
CHARACTER_REPLACEMENTS = {"ف": "f", "س": "s", "ह": "h"}


def process_labels(df):
    """
    Processes the DataFrame to extract and clean labels from the output.

    Args:
        df (pd.DataFrame): DataFrame containing the 'output' column.

    Returns:
        pd.DataFrame: DataFrame with processed labels.
    """

    # Function to apply the regex if 'label:' exists in the text
    def extract_label(row):
        if "label:" in row["output"]:
            match = re.search(LABEL_REGEX, row["output"])
            return match.group(1) if match else row["processed_output"]
        return row["processed_output"]

    df["processed_output"] = df.apply(extract_label, axis=1)
    return df


def json_to_dataframe(directory, experiment):
    """
    Reads JSON files from the specified directory and converts them to a DataFrame.

    Args:
        directory (str): Directory containing JSON files.
        experiment (str): Name of the experiment for processing.

    Returns:
        pd.DataFrame: DataFrame containing combined data from JSON files.
    """
    records = []

    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as file:
                    data = json.load(file)
                    records.append(data)
            except json.JSONDecodeError as e:
                print(f"Error reading {filepath}: {e}")

    df = pd.DataFrame(records)
    df = process_labels(df)

    df["label"] = df["label"].str.lower().replace({"_": "", "-": ""}, regex=False)

    # Clean and transform 'processed_output'
    df["processed_output"] = (
        df["processed_output"]
        .str.lower()
        .replace({"_": "", "-": ""}, regex=False)
        .replace(CHARACTER_REPLACEMENTS, regex=False)
        .apply(lambda x: x.split("/")[0] if "/" in x else x)
        .apply(
            lambda x: x.replace("*", "")
            .replace("'", "")
            .replace('"', "")
            .split(",")[0]
            .strip()
        )
    )

    print("length", len(df))
    return df


def score_dataset(df):
    """
    Scores the dataset using various metrics.

    Args:
        df (pd.DataFrame): DataFrame containing the labels and predictions.

    Returns:
        dict: Dictionary containing calculated metrics.
    """

    def replace_invalid_labels(row, labels):
        return row if row in labels else random.choice(list(labels))

    labels = set(df.label.unique())
    df["processed_output"] = df["processed_output"].apply(
        lambda x: replace_invalid_labels(x, labels)
    )

    label_encoder = LabelEncoder()
    df["label"] = label_encoder.fit_transform(df["label"])
    df["processed_output"] = label_encoder.transform(df["processed_output"])

    # Calculate various metrics
    metrics = {
        "accuracy": accuracy_score(df["label"], df["processed_output"]),
        "precision_macro": precision_score(
            df["label"], df["processed_output"], average="macro"
        ),
        "recall_macro": recall_score(
            df["label"], df["processed_output"], average="macro"
        ),
        "f1_macro": f1_score(df["label"], df["processed_output"], average="macro"),
        "precision_micro": precision_score(
            df["label"], df["processed_output"], average="micro"
        ),
        "recall_micro": recall_score(
            df["label"], df["processed_output"], average="micro"
        ),
        "f1_micro": f1_score(df["label"], df["processed_output"], average="micro"),
        "precision_weighted": precision_score(
            df["label"], df["processed_output"], average="weighted"
        ),
        "recall_weighted": recall_score(
            df["label"], df["processed_output"], average="weighted"
        ),
        "f1_weighted": f1_score(
            df["label"], df["processed_output"], average="weighted"
        ),
        "confusion_matrix": confusion_matrix(
            df["label"], df["processed_output"]
        ).tolist(),
        "classification_report": classification_report(
            df["label"],
            df["processed_output"],
            target_names=label_encoder.classes_,
            output_dict=True,
        ),
    }
    return metrics


@click.command()
@click.option(
    "--experiment_dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory containing experiment files.",
)
@click.option(
    "--output_dir",
    type=click.Path(exists=True),
    required=True,
    help="Directory to save output metrics.",
)
def main(experiment_dir, output_dir):
    """
    Main function to evaluate experiments by processing JSON files and calculating metrics.

    Args:
        experiment_dir (str): Directory containing experiment JSON files.
        output_dir (str): Directory to save the output metrics as JSON files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for experiment in os.listdir(experiment_dir):
        if "xlsum" in experiment:
            print("Skipping scoring for", experiment)
            continue

        experiment_path = os.path.join(experiment_dir, experiment)
        experiment_dataframe = json_to_dataframe(experiment_path, experiment)
        metrics = score_dataset(experiment_dataframe)

        output_json_file = os.path.join(output_dir, experiment + ".json")
        with open(output_json_file, "w") as json_file:
            json.dump(metrics, json_file, indent=4)

        print(f"Metrics saved to '{output_json_file}'")


if __name__ == "__main__":
    main()
