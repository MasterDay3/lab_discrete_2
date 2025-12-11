"""
Lab 2 template
"""
import time
import matplotlib.pyplot as plt

def read_edges(filename):
    """
    additional function to read the graph's edges
    >>> read_edges("input.dot")
    [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    """
    edges = []
    with open(filename, 'r', encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "->" in line:
                a, b = line.replace(";", "").split("->")
                a, b = int(a.strip()), int(b.strip())
                edges.append((a, b))
    return edges


def read_incidence_matrix(filename: str) -> list[list[int]]:
    """
    :param str filename: path to file
    :returns list[list[int]]: the incidence matrix of a given graph
    >>> read_incidence_matrix("input.dot")
    [[-1, -1, 1, 0, 1, 0], [1, 0, -1, -1, 0, 1], [0, 1, 0, 1, -1, -1]]
    """
    edges = read_edges(filename)
    vertices = sorted(set([u for u, _ in edges] + [v for _, v in edges]))
    n = len(vertices)
    m = len(edges)
    result = [[0]*m for _ in range(n)]
    for idx, (u, v) in enumerate(edges):
        result[u][idx] = -1
        result[v][idx] = +1
    return result


def read_adjacency_matrix(filename: str) -> list[list[int]]:
    """
    :param str filename: path to file
    :returns list[list[int]]: the adjacency matrix of a given graph
    >>> read_adjacency_matrix("input.dot")
    [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    """
    edges = read_edges(filename)
    vertices = sorted(set([u for u, _ in edges] + [v for _, v in edges]))
    n = len(vertices)
    result = [[0]*n for _ in range(n)]
    for u, v in edges:
        result[u][v] = 1
    return result


def read_adjacency_dict(filename: str) -> dict[int, list[int]]:
    """
    :param str filename: path to file
    :returns dict[int, list[int]]: the adjacency dict of a given graph
    >>> read_adjacency_dict("input.dot")
    {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    """
    edges = read_edges(filename)
    graph = {}
    for u, v in edges:
        if u not in graph:
            graph[u] = []
        graph[u].append(v)
    return graph

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

def find_cycles_adj_dict(graph, maxlen=10):
    """
    finds all the cycles in adjacency dictionary
    >>> find_cycles_adj_dict({0:[1,2],1:[0,2],2:[0,1]})
    [[0, 1], [0, 2, 1], [1, 2], [0, 2], [0, 1, 2]]
    """
    cycles = set()
    nodes = sorted(graph.keys())

    def dfs(start, current, visited, path):
        if len(path) > maxlen :
            return
        for nxt in graph.get(current, []):
            if nxt == start and len(path) >= 2:
                cycle = path[:]
                m = min(cycle)
                i = cycle.index(m)
                cycle = tuple(cycle[i:] + cycle[:i])
                cycles.add(cycle)
            if nxt not in visited and nxt >= start:
                visited.add(nxt)
                path.append(nxt)
                dfs(start, nxt, visited, path)
                path.pop()
                visited.remove(nxt)

    for start in nodes:
        dfs(start, start, {start}, [start])
    return [list(c) for c in cycles]
def find_cycles_adj_matrix(matrix):
    """
    finds all the cycles in adjacency matrix
    >>> find_cycles_adj_matrix([[0,1,1],[1,0,1],[1,1,0]])
    [[0, 1], [0, 2, 1], [1, 2], [0, 2], [0, 1, 2]]
    """
    n = len(matrix)
    graph = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if matrix[i][j]:
                graph[i].append(j)
    return find_cycles_adj_dict(graph)

def find_cycles_inc_matrix(matrix):
    """
    finds all the cycles in incedent matrix
    >>> find_cycles_inc_matrix([[-1,-1,1,0,1,0], [1,0,-1,-1,0,1], [0,1,0,1,-1,-1]])
    [[0, 1], [0, 2, 1], [1, 2], [0, 2], [0, 1, 2]]
    """
    n = len(matrix)
    m = len(matrix[0])
    graph = {i: [] for i in range(n)}
    for col in range(m):
        src = None
        dst = None
        for row in range(n):
            if matrix[row][col] == -1:
                src = row
            if matrix[row][col] == 1:
                dst = row
        if src is not None and dst is not None:
            graph[src].append(dst)
    return find_cycles_adj_dict(graph)

def read_incidence_from_adj_dict(graph):
    """
    Creates incidency matrix form adjacency graph
    """
    edges = []
    for u in graph:
        for v in graph[u]:
            edges.append((u,v))
    n = len(graph)
    m = len(edges)
    M = [[0]*m for _ in range(n)]
    for i,(u,v) in enumerate(edges):
        M[u][i] = -1
        M[v][i] = 1
    return M


def measure(f, *args):
    t = time.time()
    f(*args)
    return time.time() - t

sizes = []
times_dict = []
times_matrix = []
times_inc = []


def path_graph(n: int) -> dict[int, list[int]]:
    """
    Acycled graph.
    """
    graph = {i: [] for i in range(n)}
    for i in range(n - 1):
        graph[i].append(i + 1)
    return graph

def cycle_graph(n_in: int) -> dict[int, list[int]]:
    graph = {i: [] for i in range(n_in)}
    for i in range(n_in - 1):
        graph[i].append(i + 1)
    graph[n_in - 1].append(0)
    return graph


for n in [5, 10, 20, 40, 60, 80, 100]:
    g = path_graph(n)
    adj_dict = g
    adj_matrix = [[1 if j in g[i] else 0 for j in range(n)] for i in range(n)]
    inc_matrix = read_incidence_from_adj_dict(g)
    sizes.append(n)
    times_dict.append( measure(find_cycles_adj_dict, adj_dict) )
    times_matrix.append( measure(find_cycles_adj_matrix, adj_matrix) )
    times_inc.append( measure(find_cycles_inc_matrix, inc_matrix) )


plt.plot(sizes, times_dict, label="adj dict")
plt.plot(sizes, times_matrix, label="adj matrix")
plt.plot(sizes, times_inc, label="inc matrix")
plt.xlabel("Number of vertices")
plt.ylabel("Time (seconds)")
plt.legend()
plt.show()
sizes = []
times_dict = []
times_matrix = []
times_inc = []
for m in [5, 10, 20, 40, 60, 80, 100]:
    v = cycle_graph(m)
    adj_dict_2 = v
    adj_matrix = [[1 if j in v[i] else 0 for j in range(m)] for i in range(m)]
    inc_matrix = read_incidence_from_adj_dict(v)

    sizes.append(m)

    times_dict.append( measure(find_cycles_adj_dict, adj_dict_2) )
    times_matrix.append( measure(find_cycles_adj_matrix, adj_matrix) )
    times_inc.append( measure(find_cycles_inc_matrix, inc_matrix) )
plt.plot(sizes, times_dict, label="adj dict")
plt.plot(sizes, times_matrix, label="adj matrix")
plt.plot(sizes, times_inc, label="inc matrix")
plt.xlabel("Number of vertices")
plt.ylabel("Time (seconds)")
plt.legend()
plt.show()

if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)
