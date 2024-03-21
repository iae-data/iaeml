from iaeml.pure_python.data_structures.kdtree import KDTree
from typing import List, Dict, Union, Any
from collections import defaultdict
import itertools


class KNN:
    def __init__(self, k: int = 3, metric: str = "euclidean"):
        """ K-Nearest Neighbors classifier/regressor/multilabel

        Parameters
        ----------
        k : int, optional
            Number of nearest neighbors to consider, by default 3
        metric : str, optional
            Distance metric to use, by default "euclidean"

        Raises
        ------
        ValueError
            If k is not a positive integer
        ValueError
            If metric is not a valid distance metric

        Examples
        --------
        >>> from iaeml.pure_python.ml import KNN
        >>> knn = KNN(k=3, metric="euclidean")
        >>> knn.fit([[1, 2], [3, 4], [5, 6]], {"target": {"target_type": "regression", "list": [1, 2, 3]}})
        >>> knn.predict([1, 2])
        {'target': 2.0}
        >>> knn.predict([[1, 2], [3, 4]])
        {'target': [2.0, 2.0]}
        """
        self.k = k
        self.tree = KDTree(metric=metric)
        self.targets = {}
        self.all_possible_labels = set()

    def fit(self, data_points: List[List[float]], targets: Dict[str, Dict[str, Union[List, str]]]) -> None:
        """Fit the model to the data

        Parameters
        ----------
        data_points : List[List[float]]
            List of data points
        targets : Dict[str, Dict[str, Union[List, str]]]
            Dictionary of targets to predict

        Raises
        ------
        ValueError
            If data_points is empty
        ValueError
            If data_points and targets are not the same length
        ValueError
            If any target is not a valid target type
        ValueError
            If any target is not a valid target list

        Examples
        --------
        >>> from iaeml.pure_python.ml import KNN
        >>> knn = KNN(k=3, metric="euclidean")
        >>> knn.fit([[1, 2], [3, 4], [5, 6]], {"target": {"target_type": "regression", "list": [1, 2, 3]}})
        """
        self.tree = KDTree(data_points, metric=self.tree.distance.metric)
        self.targets = targets
        self._init_all_possible_labels(targets)

    def _init_all_possible_labels(self, targets: Dict[str, Dict[str, Union[List, str]]]) -> None:
        """ Initialize all possible labels for multilabel classification

        Parameters
        ----------
        targets : Dict[str, Dict[str, Union[List, str]]]
            Dictionary of targets to predict
        """
        for target_info in targets.values():
            if target_info["target_type"] == "multilabel":
                for label_set in target_info["list"]:
                    self.all_possible_labels.update(label_set)

    def predict(self, points: Union[List[float], List[List[float]]]) -> Dict[str, Any]:
        """Predict the target values for the given points

        Parameters
        ----------
        points : Union[List[float], List[List[float]]]
            List of points to predict
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of target names to predicted values

        Raises
        ------
        ValueError
            If points is empty
        ValueError
            If points is not a list of lists
        ValueError
            If any target is not a valid target type
        ValueError
            If any target is not a valid target list
        
        Examples
        --------
        >>> from iaeml.pure_python.ml import KNN
        >>> knn = KNN(k=3, metric="euclidean")
        >>> knn.fit([[1, 2], [3, 4], [5, 6]], {"target": {"target_type": "regression", "list": [1, 2, 3]}})
        >>> knn.predict([1, 2])
        {'target': 2.0}
        >>> knn.predict([[1, 2], [3, 4]])
        {'target': [2.0, 2.0]}
        """
        single_point = self._ensure_points_list(points)
        results = self._init_results_dict()
        for point in points:
            self._update_results_for_point(point, results)
        return self._format_results(single_point, results)

    def _ensure_points_list(self, points: Union[List[float], List[List[float]]]) -> bool:
        """ Ensure points is a list of lists

        Parameters
        ----------
        points : Union[List[float], List[List[float]]]
            List of points to predict

        Returns
        -------
        bool
            True if points is a single point, False otherwise
        """
        if not isinstance(points[0], (list, tuple)):
            points = [points]
            return True
        return False

    def _init_results_dict(self) -> Dict[str, Any]:
        """ Initialize results dictionary

        Returns
        -------
        Dict[str, Any]
            Dictionary of target names to results
        """
        results = {}
        for target_name in self.targets:
            target_type = self.targets[target_name]["target_type"]
            if target_type in ["classification", "multilabel"]:
                results[target_name] = self._init_classification_results_dict(target_name)
            else:
                results[target_name] = []
        return results

    def _init_classification_results_dict(self, target_name: str) -> Dict[str, List[float]]:
        """ Initialize classification results dictionary

        Parameters
        ----------
        target_name : str
            Name of target to predict

        Returns
        -------
        Dict[str, List[float]]
            Dictionary of labels to list of probabilities
        """
        return {label: [] for label in self._get_all_labels(target_name)}

    def _get_all_labels(self, target_name: str) -> set:
        """ Get all possible labels for the given target

        Parameters
        ----------
        target_name : str
            Name of target to predict

        Returns
        -------
        set
            Set of all possible labels for the given target
        """
        return set(itertools.chain.from_iterable(self.targets[target_name]["list"]))

    def _update_results_for_point(self, point: List[float], results: Dict[str, Any]) -> None:
        """ Update results dictionary for the given point

        Parameters
        ----------
        point : List[float]
            Point to predict
        results : Dict[str, Any]
            Dictionary of target names to results
        """

        for target_name, target_info in self.targets.items():
            nearest_labels = self._get_nearest_labels(point, target_name)
            target_type = target_info["target_type"]
            if target_type == "regression":
                results[target_name].append(self._predict_reg(nearest_labels))
            elif target_type in ["classification", "multilabel"]:
                self._update_classification_results(nearest_labels, results, target_name, target_type)

    def _get_nearest_labels(self, point: List[float], target_name: str) -> List:
        """ Get the labels of the k-nearest neighbors for the given point

        Parameters
        ----------
        point : List[float]
            Point to predict
        target_name : str
            Name of target to predict
        
        Returns
        -------
        List
            List of labels of the k-nearest neighbors for the given point
        """
        indices = self.tree.find_k_nearest_neighbors(point, self.k)[1]
        return [self.targets[target_name]["list"][i] for i in indices]

    def _update_classification_results(self, nearest_labels: List, results: Dict[str, Any],
                                       target_name: str, target_type: str) -> None:
        """ Update classification results dictionary for the given point

        Parameters
        ----------
        nearest_labels : List
            List of labels of the k-nearest neighbors for the given point
        results : Dict[str, Any]
            Dictionary of target names to results
        target_name : str
            Name of target to predict
        target_type : str
            Type of target to predict
        """
        prediction_method = self._predict_clf if target_type == "classification" else self._predict_multi
        classification_result = prediction_method(nearest_labels)
        for label, prob in classification_result.items():
            results[target_name][label].append(prob)

    def _format_results(self, single_point: bool, results: Dict[str, Any]) -> Dict[str, Any]:
        """ Format results dictionary

        Parameters
        ----------
        single_point : bool
            True if points is a single point, False otherwise
        results : Dict[str, Any]
            Dictionary of target names to results

        Returns
        -------
        Dict[str, Any]
            Dictionary of target names to results
        """
        if single_point:
            for target_name, value in results.items():
                if isinstance(value, list):
                    results[target_name] = value[0]
                else:  # dictionary
                    for key in value:
                        results[target_name][key] = value[key][0]
        return results

    @staticmethod
    def _predict_reg(values: List[float]) -> float:
        """ Predict the regression value for the given list of values

        Parameters
        ----------
        values : List[float]
            List of values to predict
        
        Returns
        -------
        float
            Predicted regression value
        """
        return sum(values) / len(values)

    @staticmethod
    def _predict_clf(labels: List[str]) -> Dict[str, float]:
        """ Predict the classification value for the given list of labels

        Parameters
        ----------
        labels : List[str]
            List of labels to predict

        Returns
        -------
        Dict[str, float]
            Dictionary of labels to probabilities
        """
        label_probabilities = defaultdict(int)
        for label in labels:
            label_probabilities[label] += 1
        total = len(labels)
        return {label: count/total for label, count in label_probabilities.items()}

    def _predict_multi(self, labels: List[set]) -> Dict[str, float]:
        """ Predict the multilabel classification value for the given list of labels

        Parameters
        ----------
        labels : List[set]
            List of labels to predict
        
        Returns
        -------
        Dict[str, float]
            Dictionary of labels to probabilities
        """
        label_counts = defaultdict(int)
        for label_set in labels:
            for label in label_set:
                label_counts[label] += 1
        # Convert counts to probabilities based on k-nearest neighbors
        label_probabilities = {label: count/len(labels) for label, count in label_counts.items()}
        # Ensure all possible labels have an associated probability
        for possible_label in self.all_possible_labels:
            if possible_label not in label_probabilities:
                label_probabilities[possible_label] = 0.0
        return label_probabilities