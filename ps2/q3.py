from typing import Dict, Set, List

graph = {
    1: {2, 4},
    2: {1, 3, 4, 5},
    3: {2, 4},
    4: {1, 2, 3, 6},
    5: {2, 6},
    6: {4, 5, 7},
    7: {6}
}


def search_v1(edges: Dict[int, Set[int]], curr: int, order: List[int]) -> List[int]:
    order.append(curr)

    while len(edges[curr]) > 0:
        neighbor = edges[curr].pop()
        edges[neighbor].remove(curr)
        search_v1(edges, neighbor, order)
        order.append(curr)

    return order


res = search_v1(graph, 1, [])
print(len(res))
print(res)
