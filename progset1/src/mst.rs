use crate::{
    graph_gen::Graph,
    prim_heap::{PrimHeap, Weight},
};
use std::{
    collections::{HashMap, HashSet},
    fmt::Debug,
    hash::Hash,
    rc::Rc,
};

#[derive(Debug)]
pub struct Prev<V> {
    pub vertex: Option<Rc<V>>,
    pub weight: Weight<f64>,
}

#[derive(Debug)]
pub struct Mst<V> {
    // All vertices have a preivous vertex except start. A tree can
    // be constructed from these edges.
    pub prevs: HashMap<Rc<V>, Prev<V>>,
    pub weight: f64,
}

impl<V> Mst<V>
where
    V: Debug + Default + PartialEq + Eq + Hash,
{
    /// Runs Prim's algorithm on the given graph from vertex `start`, returning
    /// the weight of the graph and a HashMap of relations. Each key in the map is
    /// a vertex, and the value is the other side of the edge taken to reach it.
    pub fn from_prim(graph: &Graph<V>, start: &Rc<V>) -> Self {
        // Data structures:
        let mut prevs: HashMap<Rc<V>, Prev<V>> = HashMap::new();
        let mut finished: HashSet<Rc<V>> = [start.clone()].into();
        let mut heap = PrimHeap::heapify(
            graph
                .keys()
                .map(|v| (v, Weight::new(f64::INFINITY)))
                .collect(),
        );
        let mut mst_weight: f64 = 0.;

        // Start has no predecessor and has weight 0
        heap.upsert_min(start, Weight::new(0.));
        prevs.insert(
            start.clone(),
            Prev {
                vertex: None,
                weight: Weight::new(0.),
            },
        );

        while let Some((vertex, weight)) = heap.take_min() {
            // Every vertex that exits the heap is finished.
            finished.insert(vertex.clone());
            mst_weight += weight.get();
            {
                #[cfg(debug_assertions)]
                if let Some(val) = prevs.get(&vertex) {
                    assert_eq!(
                        val.weight, weight,
                        "Final weight should equal popped weight."
                    );
                }
            }

            // Edges are only None if the vertex is disconnected.
            let edges = match graph.get(&vertex) {
                Some(e) => e,
                None => continue,
            };

            for neighbor in edges.iter() {
                // Only consider neighbors still in the heap.
                if finished.contains(&neighbor.vertex) {
                    continue;
                }
                let curr_nbr_weight = heap
                    .get_weight(&neighbor.vertex)
                    .expect("Every vertex not in assigned should have a weight in the heap");
                // If the popped vertex offers a shorter path to any of its live neighbors,
                // reparent the neighbors to use the new, shorter path.
                if &neighbor.weight < curr_nbr_weight {
                    heap.upsert_min(&neighbor.vertex, neighbor.weight);
                    prevs.insert(
                        neighbor.vertex.clone(),
                        Prev {
                            vertex: Some(vertex.clone()),
                            weight: neighbor.weight,
                        },
                    );
                }
            }
        }

        Mst {
            prevs,
            weight: mst_weight,
        }
    }
}

#[cfg(test)]
mod mst_tests {
    use crate::{
        decimal::Decimal,
        graph_gen::{CompleteUnitGraph, Graph, Vertex2D, Vertex3D, Vertex4D},
    };

    use super::{Mst, Prev};
    use std::{
        collections::{HashMap, HashSet},
        fmt::Debug,
        hash::Hash,
        rc::Rc,
    };

    /// The number of vertices per test graph.
    const NUM_VERTICES: usize = 64;

    /// Turn the prevs hashmap into a structure that's easier to search
    fn adj_graph_from_prevs<V>(prevs: &HashMap<Rc<V>, Prev<V>>) -> HashMap<Rc<V>, Vec<Rc<V>>>
    where
        V: PartialEq + Eq + Hash,
    {
        let mut adj: HashMap<Rc<V>, Vec<Rc<V>>> = HashMap::new();
        for (vertex, prev) in prevs.iter() {
            let prev_vertex = match &prev.vertex {
                Some(v) => v,
                None => continue,
            };
            adj.entry(prev_vertex.clone())
                .or_default()
                .push(vertex.clone());
        }
        adj
    }

    /// Computes the sum of edges in the output assignment
    fn weight_of_prev<V>(prevs: &HashMap<Rc<V>, Prev<V>>) -> f64 {
        prevs.values().map(|p| p.weight.get()).sum()
    }

    /// Traverses a graph, ensuring it has no cycles from the starting point.
    fn assert_no_cycles<V>(
        adj: &HashMap<Rc<V>, Vec<Rc<V>>>,
        curr: &Rc<V>,
        visited: &mut HashSet<Rc<V>>,
    ) where
        V: PartialEq + Eq + Hash + Debug,
    {
        assert!(
            visited.insert(curr.clone()),
            "Detected a cycle at {:?}",
            curr
        );
        let neighbors = match adj.get(curr) {
            Some(n) => n,
            None => return,
        };
        for neighbor in neighbors.iter() {
            assert_no_cycles(adj, neighbor, visited);
        }
    }

    /// Given a graph, generates an MST starting at each vertex in the
    /// graph verifying that all output the same weight. The output graph
    /// is also searched to verify that it's a tree.
    fn assert_mst<V>(graph: &Graph<V>)
    where
        V: PartialEq + Eq + Hash + Debug + Default,
    {
        let benchmark_start = graph.keys().next().expect("The graph shouldn't be empty");
        let benchmark_mst = Mst::from_prim(&graph, &benchmark_start);
        let expected_weight = Decimal::new(benchmark_mst.weight);

        for start in graph.keys() {
            let mst = Mst::from_prim(&graph, &start);
            let mst_adj = adj_graph_from_prevs(&mst.prevs);
            assert_no_cycles(&mst_adj, start, &mut HashSet::new());
            let this_weight = Decimal::new(mst.weight);
            assert_eq!(
                this_weight,
                Decimal::new(weight_of_prev(&mst.prevs)),
                "Returned weight should be the same as the weight of Prevs."
            );
            assert_eq!(
                this_weight, expected_weight,
                "Minimum weight should not be different no matter where we start"
            );
        }
    }

    #[test]
    fn mst_1d_graph() {
        let graph = CompleteUnitGraph::graph_1d(NUM_VERTICES);
        assert_mst(&graph);
    }

    #[test]
    fn mst_2d_graph() {
        let graph = CompleteUnitGraph::graph_nd::<Vertex2D>(NUM_VERTICES);
        assert_mst(&graph);
    }

    #[test]
    fn mst_3d_graph() {
        let graph = CompleteUnitGraph::graph_nd::<Vertex3D>(NUM_VERTICES);
        assert_mst(&graph);
    }

    #[test]
    fn mst_4d_graph() {
        let graph = CompleteUnitGraph::graph_nd::<Vertex4D>(NUM_VERTICES);
        assert_mst(&graph);
    }
}
