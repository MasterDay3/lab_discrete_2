"""
Lab 2 template
"""


def read_incidence_matrix(filename: str) -> list[list[int]]:
    """
    :param str filename: path to file
    :returns list[list[int]]: the incidence matrix of a given graph
    """
    pass


def read_adjacency_matrix(filename: str) -> list[list[int]]:
    """
    :param str filename: path to file
    :returns list[list[int]]: the adjacency matrix of a given graph
    """
    pass


def read_adjacency_dict(filename: str) -> dict[int, list[int]]:
    """
    :param str filename: path to file
    :returns dict[int, list[int]]: the adjacency dict of a given graph
    """
    pass


def iterative_adjacency_dict_dfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    :param dict[int, list[int]] graph: the adjacency list of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> iterative_adjacency_dict_dfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
    [0, 1, 2]
    >>> iterative_adjacency_dict_dfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
    [0, 1, 2, 3]
    """
    visited = set()
    stack = [start]
    result = []
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            stack.extend(reversed(graph.get(vertex, [])))
    return result

def iterative_adjacency_matrix_dfs(graph: list[list[int]], start: int) -> list[int]:
    """
    :param list[list[int]] graph: the adjacency matrix of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> iterative_adjacency_matrix_dfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> iterative_adjacency_matrix_dfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    visited = set()
    stack = [start]
    result = []
    while stack: 
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            neighbors = []
            for i, connected in enumerate(graph[vertex]):
                if connected:
                    neighbors.append(i)
            stack.extend(reversed(neighbors))
    return result

def recursive_adjacency_dict_dfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    :param dict[int, list[int]] graph: the adjacency list of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> recursive_adjacency_dict_dfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
    [0, 1, 2]
    >>> recursive_adjacency_dict_dfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
    [0, 1, 2, 3]
    """
    visited = set()
    result = []
    def dfs(vertex):
        visited.add(vertex)
        result.append(vertex)
        for neighbor in graph.get(vertex, []):
            if neighbor not in visited:
                dfs(neighbor)
    dfs(start)
    return result

def recursive_adjacency_matrix_dfs(graph: list[list[int]], start: int) -> list[int]:
    """
    :param list[list[int]] graph: the adjacency matrix of a given graph
    :param int start: start vertex of search
    :returns list[int]: the dfs traversal of the graph
    >>> recursive_adjacency_matrix_dfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> recursive_adjacency_matrix_dfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    visited = set()
    result = []
    def dfs(vertex):
        visited.add(vertex)
        result.append(vertex)
        for neighbor, connected in enumerate(graph[vertex]):
            if connected and neighbor not in visited:
                dfs(neighbor)
    dfs(start)
    return result

def iterative_adjacency_dict_bfs(graph: dict[int, list[int]], start: int) -> list[int]:
    """
    :param dict[int, list[int]] graph: the adjacency list of a given graph
    :param int start: start vertex of search
    :returns list[int]: the bfs traversal of the graph
    >>> iterative_adjacency_dict_bfs({0: [1, 2], 1: [0, 2], 2: [0, 1]}, 0)
    [0, 1, 2]
    >>> iterative_adjacency_dict_bfs({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: []}, 0)
    [0, 1, 2, 3]
    """
    visited = set()
    queue = [start]
    result = []
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            for neighbor in graph.get(vertex, []):
                if neighbor not in visited:
                    queue.append(neighbor)
    return result

def iterative_adjacency_matrix_bfs(graph: list[list[int]], start: int) -> list[int]:
    """
    :param list[list[int]] graph: the adjacency matrix of a given graph
    :param int start: start vertex of search
    :returns list[int]: the bfs traversal of the graph
    >>> iterative_adjacency_matrix_bfs([[0, 1, 1], [1, 0, 1], [1, 1, 0]], 0)
    [0, 1, 2]
    >>> iterative_adjacency_matrix_bfs([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 0, 0, 0]], 0)
    [0, 1, 2, 3]
    """
    visited = set()
    queue = [start]
    result = []
    while queue:
        vertex = queue.pop(0)
        if vertex not in visited:
            visited.add(vertex)
            result.append(vertex)
            for neighbor, connected in enumerate(graph[vertex]):
                if connected and neighbor not in visited:
                    queue.append(neighbor)
    return result



def adjacency_matrix_radius(graph: list[list[int]]) -> int:
    """
    :param list[list[int]] graph: the adjacency matrix of a given graph
    :returns int: the radius of the graph
    >>> adjacency_matrix_radius([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
    1
    >>> adjacency_matrix_radius([[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 0]])
    1
    """
    lenght = len(graph)

    def bfs_distances(start: int) -> list[int]:
        distances = [float('inf')] * lenght
        distances[start] = 0
        queue = [start]
        while queue:
            vertex = queue.pop(0)
            for neighbor, connected in enumerate(graph[vertex]):
                if connected and distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[vertex] + 1
                    queue.append(neighbor)
        return distances

    eccentricities = []
    for i in range(lenght):
        dist = bfs_distances(i)
        eccentricities.append(max(dist))
    return min(eccentricities)


def adjacency_dict_radius(graph: dict[int, list[int]]) -> int:
    """
    :param dict[int, list[int]] graph: the adjacency list of a given graph
    :returns int: the radius of the graph
    >>> adjacency_dict_radius({0: [1, 2], 1: [0, 2], 2: [0, 1]})
    1
    >>> adjacency_dict_radius({0: [1, 2], 1: [0, 2, 3], 2: [0, 1], 3: [1]})
    1
    """
    def bfs_distances(start: int) -> dict[int, int]:
        distances = {}
        for i in graph:
            distances[i] = float('inf')
        distances[start] = 0
        queue = [start]
        while queue:
            vertex = queue.pop(0)
            for neighbor in graph[vertex]:
                if distances[neighbor] == float('inf'):
                    distances[neighbor] = distances[vertex] + 1
                    queue.append(neighbor)
        return distances

    eccentricities = []
    for ch in graph:
        dist = bfs_distances(ch)
        eccentricities.append(max(dist.values()))
    return min(eccentricities)




if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
