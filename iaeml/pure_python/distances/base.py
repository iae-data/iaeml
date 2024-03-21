class Distance:
    """
    Class to handle different distance metrics.
    
    Attributes:
    -----------
    metric : str
        The metric used to calculate distance. Currently supports "euclidean".
    """
    
    def __init__(
            self, 
            metric="euclidean"
    ):
        if metric not in ["euclidean"]:
            raise ValueError("Unsupported metric. Currently, only 'euclidean' is supported.")
        self.metric = metric

    def compute(
            self, 
            p1, 
            p2
    ):
        """
        Computes the distance between two points based on the specified metric.
        
        Parameters:
        -----------
        p1 : tuple
            The first point.
        p2 : tuple
            The second point.
            
        Returns:
        --------
        float
            The distance between the two points.
        """
        if self.metric == "euclidean":
            return self._euclidean(p1, p2)

    @staticmethod
    def _euclidean(p1, p2):
        return sum((x - y) ** 2 for x, y in zip(p1, p2)) ** 0.5