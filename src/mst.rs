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
            let Some(edges) = graph.get(&vertex) else {
                continue;
            };

            for neighbor in edges {
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
    use serde::Serialize;

    use crate::{
        decimal::Decimal,
        graph_gen::{
            CompleteUnitGraph, Graph, GraphDim, Vertex0D, Vertex2D, Vertex3D, Vertex4D,
            WeightedEdge,
        },
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
    fn assert_mst<V>(graph: &Graph<V>) -> Decimal
    where
        V: PartialEq + Eq + Hash + Debug + Default,
    {
        let benchmark_start = graph.keys().next().expect("The graph shouldn't be empty");
        let benchmark_mst = Mst::from_prim(&graph, &benchmark_start);
        // Use lower-precision decimals to prevent minor float differences from
        // crashing tests.
        let decimal_precision = 8;
        let expected_weight = Decimal::new_custom(benchmark_mst.weight, decimal_precision);

        for start in graph.keys() {
            let mst = Mst::from_prim(&graph, &start);
            let mst_adj = adj_graph_from_prevs(&mst.prevs);
            assert_no_cycles(&mst_adj, start, &mut HashSet::new());
            let this_weight = Decimal::new_custom(mst.weight, decimal_precision);
            assert_eq!(
                this_weight,
                Decimal::new_custom(weight_of_prev(&mst.prevs), decimal_precision),
                "Returned weight should be the same as the weight of Prevs."
            );
            assert_eq!(
                this_weight, expected_weight,
                "Minimum weight should not be different no matter where we start"
            );
        }
        expected_weight
    }

    #[test]
    fn mst_0d_graph() {
        let graph = CompleteUnitGraph::graph_nd::<Vertex0D>(NUM_VERTICES, None);
        assert_mst(&graph);
    }

    #[test]
    fn mst_2d_graph() {
        let graph = CompleteUnitGraph::graph_nd::<Vertex2D>(NUM_VERTICES, None);
        assert_mst(&graph);
    }

    #[test]
    fn mst_3d_graph() {
        let graph = CompleteUnitGraph::graph_nd::<Vertex3D>(NUM_VERTICES, None);
        assert_mst(&graph);
    }

    #[test]
    fn mst_4d_graph() {
        let graph = CompleteUnitGraph::graph_nd::<Vertex4D>(NUM_VERTICES, None);
        assert_mst(&graph);
    }

    fn find_heaviest_edge<T>(mst: &Mst<T>) -> f64 {
        let mut heaviest = 0f64;
        for edge in mst.prevs.iter() {
            heaviest = heaviest.max(*edge.1.weight.get());
        }
        return heaviest;
    }

    /// Shortens the process of getting a start vertex when one is definitely
    /// supposed to be there.
    fn expect_start<V>(graph: &Graph<V>) -> &Rc<V> {
        graph.keys().next().expect("Graph should not be empty")
    }

    #[derive(Serialize)]
    struct HeaviestEdge {
        dimension: usize,
        size: usize,
        heaviest: f64,
    }

    /// Asserts (with certain probability of failure) that the heaviest edge in
    /// graphs of various sizes and dimensions fit within the suggested trim limit.
    #[test]
    fn heaviest_edge() {
        /*
        let mut wtr = Writer::from_path("./file_io/heaviest_edge.csv")
            .expect("output csv path should be available");
        */
        for dimension in [0usize, 2, 3, 4] {
            for size in [
                64, 128, 256, 512, 1024, /*, 2048, 4096, 8192, 16384, 32768*/
            ] {
                let dim: GraphDim = dimension
                    .try_into()
                    .expect("Hard-coded graph dimension should be viable.");
                let heaviest = match dim {
                    GraphDim::ZeroD => {
                        let graph = CompleteUnitGraph::graph_nd::<Vertex0D>(size, None);
                        let mst = Mst::from_prim(&graph, expect_start(&graph));
                        find_heaviest_edge(&mst)
                    }
                    GraphDim::TwoD => {
                        let graph = CompleteUnitGraph::graph_nd::<Vertex2D>(size, None);
                        let mst = Mst::from_prim(&graph, expect_start(&graph));
                        find_heaviest_edge(&mst)
                    }
                    GraphDim::ThreeD => {
                        let graph = CompleteUnitGraph::graph_nd::<Vertex3D>(size, None);
                        let mst = Mst::from_prim(&graph, expect_start(&graph));
                        find_heaviest_edge(&mst)
                    }
                    GraphDim::FourD => {
                        let graph = CompleteUnitGraph::graph_nd::<Vertex4D>(size, None);
                        let mst = Mst::from_prim(&graph, expect_start(&graph));
                        find_heaviest_edge(&mst)
                    }
                };
                let guessed_bound = dim.get_max_edge_weight(size);
                /*
                wtr.serialize(HeaviestEdge {
                    dimension,
                    size,
                    heaviest,
                })
                .expect("Heaviest edge serialization should work");
                */
                println!(
                    "dimension: {dimension}, size: {size}, heaviest: {heaviest}, guessed: {}",
                    guessed_bound.get()
                );
                assert!(
                    guessed_bound.take() > heaviest,
                    "Would have trimmed an edge in the mst"
                );
            }
        }
        // wtr.flush().expect("Flushing should succeed");
    }

    /// Verify MST calculation is correct by comparing it to a known MST. The graph here
    /// corresponds with the example in CLRS 21.2.
    #[test]
    fn mst_known_graph() {
        let clrs_graph = create_clrs_graph();
        for start in clrs_graph.keys() {
            let res = Mst::from_prim(&clrs_graph, start);
            assert_eq!(res.weight as i64, 37, "Mst weight should equal CLRS");
        }
    }

    /// In a sequence indexed as ['a', 'b', 'c'], etc. beginning
    /// at 'a', returns the 0-indexed index of that character in an
    /// actual array. This function is really only useful to `create_clrs_graph`.
    fn char_ind(ch: char) -> usize {
        ch as usize - 'a' as usize
    }

    /// Generate the example graph exactly matching the one in CLRS Ch 21.2
    fn create_clrs_graph() -> Graph<Vertex0D> {
        let mut graph: Graph<Vertex0D> = HashMap::new();

        // Add vertices
        let mut vertices: Vec<Rc<Vertex0D>> = Vec::new();
        for id in ('a' as usize)..=('i' as usize) {
            let vertex = Rc::new(Vertex0D {
                id: Decimal::new(id as f64),
            });
            vertices.push(vertex.clone());
            graph.insert(vertex, Vec::new());
        }

        // Add all edges. The hard-coding is undesirable, but this ensures we have
        // exactly the CLRS graph.
        for vertex in vertices.iter() {
            let mut edges = vec![];
            match vertex.id.get() as u8 as char {
                'a' => {
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('b')].clone(),
                        weight: 4f64.into(),
                    });
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('h')].clone(),
                        weight: 8f64.into(),
                    })
                }
                'b' => {
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('a')].clone(),
                        weight: 4f64.into(),
                    });
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('c')].clone(),
                        weight: 8f64.into(),
                    });
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('h')].clone(),
                        weight: 11f64.into(),
                    });
                }
                'c' => {
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('b')].clone(),
                        weight: 8f64.into(),
                    });
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('d')].clone(),
                        weight: 7f64.into(),
                    });
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('f')].clone(),
                        weight: 4f64.into(),
                    });
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('i')].clone(),
                        weight: 2f64.into(),
                    });
                }
                'd' => {
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('c')].clone(),
                        weight: 7f64.into(),
                    });
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('e')].clone(),
                        weight: 9f64.into(),
                    });
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('f')].clone(),
                        weight: 14f64.into(),
                    });
                }
                'e' => {
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('d')].clone(),
                        weight: 9f64.into(),
                    });
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('f')].clone(),
                        weight: 10f64.into(),
                    });
                }
                'f' => {
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('c')].clone(),
                        weight: 4f64.into(),
                    });
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('d')].clone(),
                        weight: 14f64.into(),
                    });
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('e')].clone(),
                        weight: 10f64.into(),
                    });
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('g')].clone(),
                        weight: 2f64.into(),
                    });
                }
                'g' => {
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('f')].clone(),
                        weight: 2f64.into(),
                    });
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('h')].clone(),
                        weight: 1f64.into(),
                    });
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('i')].clone(),
                        weight: 6f64.into(),
                    });
                }
                'h' => {
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('a')].clone(),
                        weight: 8f64.into(),
                    });
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('b')].clone(),
                        weight: 11f64.into(),
                    });
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('g')].clone(),
                        weight: 1f64.into(),
                    });
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('i')].clone(),
                        weight: 7f64.into(),
                    });
                }
                'i' => {
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('c')].clone(),
                        weight: 2f64.into(),
                    });
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('g')].clone(),
                        weight: 6f64.into(),
                    });
                    edges.push(WeightedEdge {
                        vertex: vertices[char_ind('h')].clone(),
                        weight: 7f64.into(),
                    });
                }
                _ => panic!("Unexpected vertex id"),
            }
            graph
                .get_mut(vertex)
                .expect("Every vertex should be in the graph")
                .extend(edges.into_iter());
        }
        graph
    }
}
