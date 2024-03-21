from iaeml.pure_python.distances import Distance

class Node:
    """
    A Node class to represent each individual point in the KDTree structure.
    
    Attributes:
    -----------
    point : tuple
        The point (x, y) that this node represents.
    left : Node or None
        The left child of this node.
    right : Node or None
        The right child of this node.
    """

    def __init__(self, point, left=None, right=None):
        """
        Initializes a new instance of the Node class.
        
        Parameters:
        -----------
        point : tuple
            The point (x, y) to store in this node.
        left : Node or None, optional
            The left child of this node. By default, None.
        right : Node or None, optional
            The right child of this node. By default, None.
        """
        self.point = point
        self.left = left
        self.right = right

class KDTree:
    """
    A simple Multidimensional KDTree implementation for efficient nearest neighbor search.

    Attributes:
    -----------
    root : Node or None
        The root node of the KDTree.
    """
    
    def __init__(self, data_points=None, metric="euclidean"):
        self.root = None
        self.distance = Distance(metric=metric)
        if data_points:
            for idx, point in enumerate(data_points):
                self._insert(point, idx)

    def _insert(self, point, idx, depth=0):
        if not self.root:
            self.root = Node((point, idx))
            return

        current = self.root
        num_dimensions = len(point)
        while current:
            axis = depth % num_dimensions  # Adjust axis based on point's dimensionality
            if point[axis] < current.point[0][axis]:
                if not current.left:
                    current.left = Node((point, idx))
                    return
                current = current.left
            else:
                if not current.right:
                    current.right = Node((point, idx))
                    return
                current = current.right
            depth += 1

    def find_k_nearest_neighbors(self, point, k, depth=0):
        if not self.root:
            return None

        num_dimensions = len(point)
        best_neighbors = [(float('inf'), None) for _ in range(k)]

        def _search(node, depth):
            if not node:
                return

            axis = depth % num_dimensions
            d = self.distance.compute(point, node.point[0])
            
            # Check if this node is one of the k-nearest neighbors
            if d < best_neighbors[-1][0]:
                best_neighbors.append((d, node.point[1]))
                best_neighbors.sort(key=lambda x: x[0])
                best_neighbors.pop()

            next_branch = None
            opposite_branch = None

            if point[axis] < node.point[0][axis]:
                next_branch = node.left
                opposite_branch = node.right
            else:
                next_branch = node.right
                opposite_branch = node.left

            _search(next_branch, depth+1)
            if abs(point[axis] - node.point[0][axis]) < best_neighbors[-1][0]:
                _search(opposite_branch, depth+1)

        _search(self.root, 0)
        distances, indices = zip(*best_neighbors)
        return distances, indices