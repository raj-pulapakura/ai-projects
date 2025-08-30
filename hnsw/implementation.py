from heapq import heappop, heappush
import numpy as np


class Node:
    def __init__(self, vector, metadata=None):
        self.vector = vector
        self.metadata = metadata if metadata is not None else {}


class DistanceIdPair:
    def __init__(self, distance_to_query, node_id):
        self.distance_to_query = distance_to_query
        self.node_id = node_id

    def __lt__(self, other):
        return self.distance_to_query < other.distance_to_query


class HNSW:
    M: int
    M_0: int
    efConstruction: int
    mL: float
    nodes: list[Node]
    adj: list[dict[int, list[int]]]
    entry_point: int
    entry_level: int

    def __init__(self):
        self.M = 24  # max number of neighbours per node
        self.M_0 = self.M * 2
        # max number of neighbours per node for bottom layer (higher to allow for more dense, precise connections)
        self.efConstruction = 200  # number of top candidates to return when searching a layer for closest matches to query vector
        self.mL = 1 / np.log(self.M)  # normalization factor

        # Holds embeddings
        self.nodes = []
        # Stores the neighbours | list of dictionaries | self.adj[layer][node_id]
        self.adj = []

        self.entry_point = None
        self.entry_level = -1
        # entry_point and entry_level will be updated as construction happens

    def get_node_level(self):
        u = np.random.uniform(0, 1)
        level = np.floor(-np.log(u) * self.mL)
        return int(level)

    def search_layer(
        self, query: list, entry_id: int, layer: int, ef: int
    ) -> list[DistanceIdPair]:
        """
        query: the query vector
        entry_id: node id to start from on this layer
        layer: layer to search (0 = base)
        ef: beam size for this layer's search (ef >= 1)

        Returns: up to ef candidates as (distance, node_id) sorted by distance ascending
        """

        entry_node = self.nodes[entry_id]
        # distance from query to entry point
        d_entry = np.linalg.norm(query - entry_node.vector)

        # Min-heap
        candidates = [(d_entry, entry_id)]
        candidates = [DistanceIdPair(d_entry, entry_id)]
        # Nodes we have already visited
        visited = set([entry_id])
        # Return list
        top = [DistanceIdPair(d_entry, entry_id)]

        while len(candidates) > 0:
            # Since candidates is a min-heap, heappop will give us the nearest candidate, i.e. the node with the lowest distance (first value in tuple)
            nearest_candidate = heappop(candidates)

            # Find the worst of the best
            worst_of_top = top[-1]

            # If this candidate is even worse than the worst of the best, then early exit. We don't need to keep finding candidates
            if nearest_candidate.distance_to_query > worst_of_top.distance_to_query:
                break

            # Iterate through all neighbours of our candidate
            for neighbour_id in self.adj[layer][nearest_candidate.node_id]:
                # If we've already visited this node, then don't worry about it
                if neighbour_id in visited:
                    continue

                visited.add(neighbour_id)

                neighbour_node = self.nodes[neighbour_id]
                neighbour_dist_to_query = np.linalg.norm(neighbour_node.vector - query)
                worst_of_top = top[-1]

                # If top isn't filled up yet, or neighbours distance is shorter to query than the current worst in top, then:
                if (
                    len(top) < ef
                    or neighbour_dist_to_query < worst_of_top.distance_to_query
                ):
                    # add neighbour to candidates list -> we will explore this node even further
                    heappush(
                        candidates,
                        DistanceIdPair(neighbour_dist_to_query, neighbour_id),
                    )
                    # add neighbour to top list, maintaining sort order (ascending by distance)
                    if len(top) < ef:
                        top.append(
                            DistanceIdPair(neighbour_dist_to_query, neighbour_id)
                        )
                    else:
                        for i, node in enumerate(top):
                            node_dist_to_query = node.distance_to_query
                            if neighbour_dist_to_query < node_dist_to_query:
                                top.insert(
                                    i,
                                    DistanceIdPair(
                                        neighbour_dist_to_query, neighbour_id
                                    ),
                                )
                                break
                    # if number of nodes in top is greater than ef (which it will only every be by 1), then cut worst one.
                    if len(top) > ef:
                        top.pop()

        return top

    def select_neighbours(self, candidates: list[DistanceIdPair], budget: int):
        """
        candidates: the list of candidates for potential neighbours to query (assumed sorted by ascending distance)
            list of (dist_to_query, node_id) pairs
        budget: the number of nodes we want to reduce candidates to
        layer: the layer we are working on
        """

        # Initial list of selected nodes, constrained to max length of *budget*
        selected: list[DistanceIdPair] = []

        # Iterate through each candidate
        for c in candidates:
            c_vec = self.nodes[c.node_id].vector
            add_candidate = True

            # Go through each node currently in selected
            for s in selected:
                if not add_candidate:
                    break
                s_vec = self.nodes[s.node_id].vector
                c_dist_to_s = np.linalg.norm(c_vec - s_vec)
                # If the distance(candidate, selected) is less than distance(candidate, query)
                # i.e. candidate is closer to s than q
                # Then there's not point in adding candidate as a selected
                # Since s is already selected, and going from q -> s -> candidate is already good enough
                # There's no point in adding a connection q -> candidate, as we would rather save this connection
                # For a neighbour that is further away, creating long connections
                # Analogy: Let's say we already have a airline route from City Q to City S. City C is far away from City Q but close to City S
                # We wouldn't spend infrastructure on building a route from City Q to City C, as riders can just go City Q -> City S -> City C
                # We'd much rather save this "route slot" for a city in another direction, so City Q can access more places
                if c_dist_to_s < c.distance_to_query:
                    add_candidate = False
            if add_candidate:
                selected.append(c)
            if len(selected) == budget:
                return selected

        return selected

    def add_node(self, node: Node):
        node_id = len(self.nodes)
        self.nodes.append(node)
        return node_id

    def insert(self, node: Node):
        if len(self.adj) == 0:
            id = self.add_node(node)
            level_new = self.get_node_level()
            for _ in range(0, level_new + 1):
                self.adj.append({id: []})
            self.entry_level = level_new
            self.entry_point = id
            return id

        # build infra for new vec
        id = self.add_node(node)
        level_new = self.get_node_level()
        for l in range(0, level_new + 1):
            if l < len(self.adj):
                self.adj[l][id] = []
            else:
                self.adj.append({id: []})

        # greedy descent to layer just above level_new
        entry_point = self.entry_point
        entry_level = self.entry_level

        for l in range(entry_level, level_new + 1, -1):
            top = self.search_layer(
                query=node.vector, entry_id=entry_point, layer=l, ef=1
            )
            entry_point = top[0].node_id

        # layer by layer linking from level_new down to u
        for l in range(level_new, -1, -1):
            if len(self.adj[l].keys()) == 1:
                continue
            cands = self.search_layer(
                query=node.vector, entry_id=entry_point, layer=l, ef=self.efConstruction
            )
            budget = self.M_0 if l == 0 else self.M
            S = self.select_neighbours(cands, budget)
            for u in S:
                self.adj[l][u.node_id].append(id)
                self.adj[l][id].append(u.node_id)

            entry_point = S[0].node_id

        if level_new > self.entry_level:
            self.entry_level = level_new
            self.entry_point = id

    def search(self, query: list, k: int, efSearch: int = 200) -> list[Node]:
        """
        query: the query vector
        k: number of nearest neighbours to return
        efSearch: beam size for search (efSearch >= k recommended)
        """
        if len(self.adj) == 0:
            return []

        entry_point = self.entry_point
        entry_level = self.entry_level

        # Greedy descent to layer just above the base layer
        for l in range(entry_level, 0, -1):
            top = self.search_layer(query, entry_point, l, ef=1)
            entry_point = top[0].node_id

        # Base-layer search
        cands = self.search_layer(query, entry_point, 0, ef=efSearch)

        top_k = cands[:k]

        return [self.nodes[c.node_id] for c in top_k]

    def save(self, path: str):
        """
        Save the index to a file.
        """
        import pickle

        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str) -> "HNSW":
        """
        Load the index from a file.
        """
        import pickle

        with open(path, "rb") as f:
            return pickle.load(f)
