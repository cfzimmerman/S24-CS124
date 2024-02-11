use crate::{decimal::Decimal, error::PsetRes, prim_heap::Weight};
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
    vertex: T,
    weight: Weight<f64>,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Vertex1D {
    id: Decimal,
}

#[derive(Debug, PartialEq, Eq, Hash)]
pub struct Vertex2D {
    x: Decimal,
    y: Decimal,
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

        // Add edges so that it's a complete graph
        for v_from in vertices.iter() {
            for v_to in vertices.iter() {
                if v_from == v_to {
                    continue;
                }
                let new_edge = WeightedEdge {
                    vertex: v_to.clone(),
                    weight: rng.gen::<f64>().into(),
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
    pub fn graph_nd<V>(num_vertices: usize) -> PsetRes<Graph<V>>
    where
        V: VertexCoord<V> + PartialEq + Eq + Hash,
    {
        let mut vertices = HashSet::new();
        let mut rng = thread_rng();

        // Generate all vertices. Because of vertex random generation, we may
        // need to retry if duplicates are created. However, limit that to at
        // most 4x tries total to prevent an infinite loop somehow.
        for _ in 0..(num_vertices * 4) {
            if vertices.len() > num_vertices {
                break;
            }
            vertices.insert(Rc::new(V::new_rand(&mut rng)));
        }

        let mut graph: Graph<V> = HashMap::new();
        for v_from in vertices.iter() {
            for v_to in vertices.iter() {
                if v_from == v_to {
                    continue;
                }
                let new_edge = WeightedEdge {
                    vertex: v_to.clone(),
                    weight: v_from.dist(v_to).into(),
                };
                if let Some(edges) = graph.get_mut(v_from) {
                    edges.insert(new_edge);
                    continue;
                };
                graph.insert(v_from.clone(), [new_edge].into());
            }
        }
        Ok(graph)
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
        ((&other.x - &self.x).get().powi(2) + (&other.y - &self.y).get().powi(2)).sqrt()
    }

    fn new_rand(rng: &mut ThreadRng) -> Vertex2D {
        Vertex2D {
            x: rng.gen::<f64>().into(),
            y: rng.gen::<f64>().into(),
        }
    }
}

#[cfg(test)]
mod graph_tests {
    use super::{CompleteUnitGraph, Graph};

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

    #[test]
    fn gen_graph_1d() {
        for size in 2..16 {
            let graph = CompleteUnitGraph::graph_1d(size);
            assert_eq!(graph.len(), size, "Graph size should equal input");
            let expected_num_edges = (size * (size - 1)) / 2;
            assert_eq!(
                expected_num_edges,
                count_edges(&graph),
                "Graph edges should indicate completeness"
            );
        }
    }
}
