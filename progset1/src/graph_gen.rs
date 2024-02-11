use crate::{decimal::Decimal, prim_heap::Weight};
use rand::{rngs::ThreadRng, thread_rng, Rng};
use std::{
    collections::{HashMap, HashSet},
    hash::Hash,
    rc::Rc,
};

/// Required behaviors for vertices on randomly-generated
/// n-dimensional graphs.
pub trait VertexCoord<T>
where
    T: PartialEq + Eq + Hash,
{
    /// How far apart two coordinates are in an n-dimensional space.
    fn dist(&self, other: &T) -> f64;

    /// Generates a new coordinate at a random location.
    fn new_rand(rng: &mut ThreadRng) -> T;
}

#[derive(Debug)]
pub struct WeightedEdge<T> {
    pub vertex: T,
    pub weight: Weight<f64>,
}

#[derive(Debug, PartialEq, Eq, Hash, Default)]
pub struct Vertex1D {
    pub id: Decimal,
}

#[derive(Debug, PartialEq, Eq, Hash, Default)]
pub struct Vertex2D {
    pub x: Decimal,
    pub y: Decimal,
}

#[derive(Debug, PartialEq, Eq, Hash, Default)]
pub struct Vertex3D {
    pub x: Decimal,
    pub y: Decimal,
    pub z: Decimal,
}

#[derive(Debug, PartialEq, Eq, Hash, Default)]
pub struct Vertex4D {
    pub r: Decimal,
    pub g: Decimal,
    pub b: Decimal,
    pub a: Decimal,
}

pub type Edges<V> = HashSet<WeightedEdge<V>>;
pub type Graph<V> = HashMap<Rc<V>, Edges<Rc<V>>>;

/// Used to generate complete graphs of various dimensions with
/// vertex coordinates between 0 and 1.
pub struct CompleteUnitGraph();

impl CompleteUnitGraph {
    /// Generates a complete graph of `num_vertices` with ids
    /// from 0 to `num_vertices - 1`.
    pub fn graph_1d(num_vertices: usize) -> Graph<Vertex1D> {
        let mut graph: Graph<Vertex1D> = HashMap::new();
        let mut rng = thread_rng();

        // Generate the vertices
        let mut vertices = HashSet::new();
        for id in 0..num_vertices {
            vertices.insert(Rc::new(Vertex1D {
                id: Decimal::new(id as f64),
            }));
        }

        // Add edges so that it's a complete graph.
        // Initialize the weights to -1. They'll be updated once
        // all edges are in place.
        for v_from in vertices.iter() {
            for v_to in vertices.iter() {
                if v_from == v_to {
                    continue;
                }
                // Ensure both sides of an edge have the same weight
                let weight = Self::get_existing_weight(&mut graph, v_from, v_to)
                    .unwrap_or_else(|| rng.gen::<f64>().into());
                let new_edge = WeightedEdge {
                    vertex: v_to.clone(),
                    weight,
                };
                if let Some(edges) = graph.get_mut(v_from) {
                    edges.insert(new_edge);
                    continue;
                };
                graph.insert(v_from.clone(), [new_edge].into());
            }
        }
        graph
    }

    /// Generates a complete n dimensional graph where each coordinate is generated
    /// by calling T::new_rand and edge distances are the distance between
    /// points.
    pub fn graph_nd<V>(num_vertices: usize) -> Graph<V>
    where
        V: VertexCoord<V> + PartialEq + Eq + Hash,
    {
        let mut graph: Graph<V> = HashMap::new();
        let mut vertices = HashSet::new();
        let mut rng = thread_rng();

        // Generate all vertices. Because of vertex random generation, we may
        // need to retry if duplicates are created. However, limit that to at
        // most 4x tries total to prevent an infinite loop somehow.
        for _ in 0..(num_vertices * 4) {
            if num_vertices <= vertices.len() {
                break;
            }
            vertices.insert(Rc::new(V::new_rand(&mut rng)));
        }

        // Add edges
        for v_from in vertices.iter() {
            for v_to in vertices.iter() {
                if v_from == v_to {
                    continue;
                }
                let weight = Self::get_existing_weight(&mut graph, v_from, v_to)
                    .unwrap_or_else(|| v_from.dist(v_to).into());
                let new_edge = WeightedEdge {
                    vertex: v_to.clone(),
                    weight,
                };
                if let Some(edges) = graph.get_mut(v_from) {
                    edges.insert(new_edge);
                    continue;
                };
                graph.insert(v_from.clone(), [new_edge].into());
            }
        }
        graph
    }

    /// Before adding a new edge, this function checks if a weighted edge
    /// beginning on the other side already exists. If so, that weight is
    /// returned. If not, None is returned.
    /// Even for n-dimensional graphs, this is helpful to avoid float
    /// imprecision when recalculating distance.
    ///
    /// The (old_to, old_from) naming implies that the caller is now suggesting
    /// an edge in the other direction where old_to == new_from and old_from == new_to.
    fn get_existing_weight<V>(
        graph: &mut Graph<V>,
        old_to: &Rc<V>,
        old_from: &Rc<V>,
    ) -> Option<Weight<f64>>
    where
        V: PartialEq + Eq + Hash,
    {
        if let Some(other_side) = graph.get(old_from) {
            let target_edge = WeightedEdge {
                vertex: old_to.clone(),
                // Equality on edges doesn't check weights
                weight: Weight::new(-1.),
            };
            if let Some(existing_edge) = other_side.get(&target_edge) {
                return Some(existing_edge.weight);
            }
        }
        None
    }
}

impl<T> PartialEq for WeightedEdge<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.vertex.eq(&other.vertex)
    }
}

impl<T> Eq for WeightedEdge<T> where T: Eq {}

impl<T> Hash for WeightedEdge<T>
where
    T: PartialEq + Eq + Hash,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.vertex.hash(state)
    }
}

impl VertexCoord<Vertex2D> for Vertex2D {
    fn dist(&self, other: &Vertex2D) -> f64 {
        let dst = [&other.x - &self.x, &other.y - &self.y]
            .into_iter()
            .map(|num| num.get().powi(2))
            .sum::<f64>()
            .sqrt();
        // Verifies correctness at 2d. Correctness beyond 2d is assumed because of this.
        debug_assert_eq!(
            Decimal::new(dst),
            Decimal::new(
                ((&other.x - &self.x).get().powi(2) + (&other.y - &self.y).get().powi(2)).sqrt()
            ),
            "Iterable 2d dist should equal formula 2d dist."
        );
        dst
    }

    fn new_rand(rng: &mut ThreadRng) -> Vertex2D {
        Vertex2D {
            x: rng.gen::<f64>().into(),
            y: rng.gen::<f64>().into(),
        }
    }
}

impl VertexCoord<Vertex3D> for Vertex3D {
    fn dist(&self, other: &Vertex3D) -> f64 {
        [&other.x - &self.x, &other.y - &self.y, &other.z - &self.z]
            .into_iter()
            .map(|num| num.get().powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn new_rand(rng: &mut ThreadRng) -> Vertex3D {
        Vertex3D {
            x: rng.gen::<f64>().into(),
            y: rng.gen::<f64>().into(),
            z: rng.gen::<f64>().into(),
        }
    }
}

impl VertexCoord<Vertex4D> for Vertex4D {
    fn dist(&self, other: &Vertex4D) -> f64 {
        [
            &other.r - &self.r,
            &other.g - &self.g,
            &other.b - &self.b,
            &other.a - &self.a,
        ]
        .into_iter()
        .map(|num| num.get().powi(2))
        .sum::<f64>()
        .sqrt()
    }

    fn new_rand(rng: &mut ThreadRng) -> Vertex4D {
        Vertex4D {
            r: rng.gen::<f64>().into(),
            g: rng.gen::<f64>().into(),
            b: rng.gen::<f64>().into(),
            a: rng.gen::<f64>().into(),
        }
    }
}

#[cfg(test)]
mod graph_tests {
    use super::{CompleteUnitGraph, Graph, Vertex2D, Vertex3D, Vertex4D, WeightedEdge};
    use crate::prim_heap::Weight;
    use std::hash::Hash;

    /// Returns the number of undirected edges in the graph and panics
    /// if there's an odd number of edges.
    fn count_edges<T>(graph: &Graph<T>) -> usize {
        let total = graph.values().fold(0, |acc, edges| acc + edges.len());
        assert_eq!(
            total % 2,
            0,
            "Graph is undirected, there should be one edge from each side"
        );
        total / 2
    }

    /// Verifies that the graph has the edge counts of a complete, undirected
    /// graph.
    fn assert_complete_graph<V>(graph: &Graph<V>, size: usize) {
        assert_eq!(graph.len(), size, "Graph size should equal input");
        let expected_num_edges = (size * (size - 1)) / 2;
        assert_eq!(
            expected_num_edges,
            count_edges(&graph),
            "Graph edges should indicate completeness"
        );
    }

    /// Verifies every edge in the graph has a counterpart in the
    /// other direction and that both sides have the same weight.
    fn assert_edges_well_defined<V>(graph: &Graph<V>)
    where
        V: PartialEq + Eq + Hash,
    {
        for (vertex, edges) in graph.iter() {
            for edge in edges.iter() {
                let other_side_weight = &graph
                    .get(&edge.vertex)
                    .expect("Other vertex should be in the graph")
                    .get(&WeightedEdge {
                        vertex: vertex.clone(),
                        // Again, equality only looks at the vertex.
                        weight: Weight::new(-1.),
                    })
                    .expect("Other vertex should have an edge pointing to curr vertex")
                    .weight;
                // While comparing floats is flakey, graph generation should ensure they're
                // identical.
                assert_eq!(&edge.weight, other_side_weight);
                assert!(
                    edge.weight >= Weight::new(0.),
                    "No negative-weight edges should remain"
                );
            }
        }
    }

    /// Generates all graph sizes and verifies invariants about them
    #[test]
    fn gen_graphs() {
        for size in 2..=16 {
            // The subscopes prevent copy-paste accidentally testing the wrong graph
            {
                let graph_1d = CompleteUnitGraph::graph_1d(size);
                assert_complete_graph(&graph_1d, size);
                assert_edges_well_defined(&graph_1d);
            }
            {
                let graph_2d = CompleteUnitGraph::graph_nd::<Vertex2D>(size);
                assert_complete_graph(&graph_2d, size);
                assert_edges_well_defined(&graph_2d);
            }
            {
                let graph_3d = CompleteUnitGraph::graph_nd::<Vertex3D>(size);
                assert_complete_graph(&graph_3d, size);
                assert_edges_well_defined(&graph_3d);
            }
            {
                let graph_4d = CompleteUnitGraph::graph_nd::<Vertex4D>(size);
                assert_complete_graph(&graph_4d, size);
                assert_edges_well_defined(&graph_4d);
            }
        }
    }
}
