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
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score, accuracy_score, multilabel_confusion_matrix
import random
import click
from datasets import load_metric

metric = load_metric("/workspace/llamalens/LlamaLens/bin/evaluation/rouge-metric/rouge")


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

    # Clean and transform 'response'
    df["response"] = df["response"].astype(str).str.lower().str.strip()
    df["output"] = df["output"].astype(str).str.lower().str.strip()


    print("Length of dataframe:", len(df))
    return df


def score_dataset(df):
    """
    Scores the dataset using various metrics.

    Args:
        df (pd.DataFrame): DataFrame containing the labels and predictions.

    Returns:
        dict: Dictionary containing calculated metrics.
    """
    
    dataset_name = df.dataset.iloc[0]
    lang = df.lang.iloc[0]
    if "Hostility" in dataset_name:
        df['output'] = df['output'].str.split(',').apply(lambda x: [item.strip() for item in x])
        df['response'] = df['response'].str.split(',').apply(lambda x: [item.strip() for item in x])

        # Initialize the MultiLabelBinarizer
        mlb = MultiLabelBinarizer()

        # Fit the binarizer on the true labels and transform both true and predicted labels
        y_true = mlb.fit_transform(df['output'])
        y_pred = mlb.transform(df['response'])

        # Get the list of all label classes
        all_labels = mlb.classes_
        print(f"Classes: {all_labels}")

        # Generate the classification report for multi-label classification
        report_dict = classification_report(y_true, y_pred, target_names=all_labels, output_dict=True)

        # Calculate additional F1 scores with different averaging methods
        report_dict['f1_micro'] = f1_score(y_true, y_pred, average='micro')
        report_dict['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        report_dict['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')

        # Calculate accuracy and add to the report dictionary
        report_dict['accuracy'] = accuracy_score(y_true, y_pred)
        return report_dict


    
    if "xlsum" in dataset_name:
        #df = df.sample(n=10, random_state=42)
        language = {"ar":"fa", "hi":"hi", "en":"en"}


        scores_list = []
        for ref, pred in zip(df["output"].tolist(), df["response"].tolist()):
            # Compute ROUGE score for each sample
            result = metric.compute(predictions=[pred], references=[ref], language=language[lang])
            result = {key: round(value.mid.fmeasure, 4) * 100 for key, value in result.items()}
            scores_list.append(result)
            print(result)
            print("\n\n")

        # Compute average scores
        # Assuming 'result' is a dictionary of metric names and their corresponding values.
        total_scores = {'rouge1': 0, 'rouge2': 0, 'rougeL': 0, 'rougeLsum': 0}
        num_jsons = len(scores_list)

        # Sum the values for each metric
        for score in scores_list:
            for key in total_scores:
                total_scores[key] += score[key]

        # Calculate the average for each metric
        average_scores = {key: value / num_jsons for key, value in total_scores.items()}

        return average_scores

    def replace_invalid_labels(row, labels):
        """
        Replaces invalid label with a random valid label.
        
        Args:
            row (str): The label to be checked.
            labels (set): The valid labels.
        
        Returns:
            str: The original or random valid label.
        """
        return row if row in labels else random.choice(list(labels))

    labels = set(df["output"].unique())
    
    # Apply the invalid label handling
    df["response"] = df["response"].apply(lambda x: replace_invalid_labels(x, labels))

    # Label encoding
    label_encoder = LabelEncoder()
    df["output"] = label_encoder.fit_transform(df["output"])
    df["response"] = label_encoder.transform(df["response"])

    # Calculate various metrics
    metrics = {
        "accuracy": accuracy_score(df["output"], df["response"]),
        "precision_macro": precision_score(df["output"], df["response"], average="macro"),
        "recall_macro": recall_score(df["output"], df["response"], average="macro"),
        "f1_macro": f1_score(df["output"], df["response"], average="macro"),
        "precision_micro": precision_score(df["output"], df["response"], average="micro"),
        "recall_micro": recall_score(df["output"], df["response"], average="micro"),
        "f1_micro": f1_score(df["output"], df["response"], average="micro"),
        "precision_weighted": precision_score(df["output"], df["response"], average="weighted"),
        "recall_weighted": recall_score(df["output"], df["response"], average="weighted"),
        "f1_weighted": f1_score(df["output"], df["response"], average="weighted"),
        "confusion_matrix": confusion_matrix(df["output"], df["response"]).tolist(),
        "classification_report": classification_report(
            df["output"],
            df["response"],
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

        experiment_path = os.path.join(experiment_dir, experiment)
        experiment_dataframe = json_to_dataframe(experiment_path, experiment)

        metrics = score_dataset(experiment_dataframe)

        output_json_file = os.path.join(output_dir, experiment + ".json")
        with open(output_json_file, "w", encoding="utf-8") as json_file:
            json.dump(metrics, json_file, indent=4, ensure_ascii=False)

        print(f"Metrics saved to '{output_json_file}'")


if __name__ == "__main__":
    main()
