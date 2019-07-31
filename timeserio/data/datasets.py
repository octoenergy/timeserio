import pandas as pd
import sklearn.datasets


def load_iris_df():
    """Return Iris dataset as a DataFrame."""
    iris = sklearn.datasets.load_iris()
    feature_names = [
        s.replace("(cm)", "cm").strip().replace(" ", "_")
        for s in iris['feature_names']
    ]
    df = pd.DataFrame(
        data=iris['data'],
        columns=feature_names
    )
    target_names = iris['target_names']
    targets = target_names[iris['target']]
    df["species"] = targets
    return df
