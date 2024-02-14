use crate::{decimal::Decimal, error::PsetErr, prim_heap::Weight};
use rand::{rngs::ThreadRng, thread_rng, Rng};
use std::{collections::HashMap, hash::Hash, rc::Rc};

/// Required behaviors for vertices on randomly-generated
/// n-dimensional graphs.
pub trait VertexCoord<T>
where
    T: PartialEq + Eq + Hash,
{
    /// This is the longest possible edge across the state space.
    /// For a unit graph of `n` dimensions, this is `sqrt(n)`
    const DIMENSION: GraphDim;

    /// How far apart two coordinates are in an n-dimensional space.
    /// The ThreadRng is an ugly addition for the zero dimension case
    fn dist(&self, other: &T, rng: &mut ThreadRng) -> f64;

    /// Generates a new coordinate at a random location.
    fn new_rand(rng: &mut ThreadRng) -> T;
}

/// The various graph dimensions this program supports.
#[derive(Debug, Clone, Copy)]
pub enum GraphDim {
    ZeroD,
    TwoD,
    ThreeD,
    FourD,
}

#[derive(Debug)]
pub struct WeightedEdge<T> {
    pub vertex: T,
    pub weight: Weight<f64>,
}

#[derive(Debug, PartialEq, Eq, Hash, Default)]
pub struct Vertex0D {
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

pub type Edges<V> = Vec<WeightedEdge<V>>;
pub type Graph<V> = HashMap<Rc<V>, Edges<Rc<V>>>;

/// Used to generate complete graphs of various dimensions with
/// vertex coordinates between 0 and 1.
pub struct CompleteUnitGraph();

impl CompleteUnitGraph {
    /// Generates a complete n dimensional graph where each coordinate is generated
    /// by calling T::new_rand and edge distances are the distance between
    /// points.
    pub fn graph_nd<V>(num_vertices: usize, trim_threshold: Option<f64>) -> Graph<V>
    where
        V: VertexCoord<V> + PartialEq + Eq + Hash,
    {
        let mut graph: Graph<V> = HashMap::new();
        let mut rng = thread_rng();

        // Generate all vertices. We may produce slightly fewer than num_vertices due to
        // random generation, but the odds of frequent collisions are very small.
        let mut vertices = Vec::with_capacity(num_vertices);
        for _ in 0..num_vertices {
            let vertex = Rc::new(V::new_rand(&mut rng));
            vertices.push(vertex.clone());
            if graph
                .insert(vertex, Vec::with_capacity(num_vertices))
                .is_some()
            {
                eprintln!("graph_nd generated a duplicate vertex");
                continue;
            };
        }

        // Add edges
        let trim_threshold = trim_threshold.unwrap_or(f64::INFINITY);
        for (v1_ind, v1) in vertices.iter().enumerate() {
            let mut v1_edges = Vec::with_capacity(vertices.len());
            for v2 in vertices.iter().skip(v1_ind + 1) {
                let weight: Weight<f64> = v1.dist(v2, &mut rng).into();
                if weight.get() > &trim_threshold {
                    continue;
                }
                v1_edges.push(WeightedEdge {
                    vertex: v2.clone(),
                    weight,
                });
                graph
                    .get_mut(v2)
                    .expect("v2 should have already been added to the graph")
                    .push(WeightedEdge {
                        vertex: v1.clone(),
                        weight,
                    });
            }
            graph
                .get_mut(v1)
                .expect("v1 should have aready been added to the graph")
                .extend(v1_edges.into_iter());
        }
        graph
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
        self.vertex.hash(state);
    }
}

impl VertexCoord<Vertex0D> for Vertex0D {
    const DIMENSION: GraphDim = GraphDim::ZeroD;

    /// Dist for Vertex0D IGNORES other and just returns a random number.
    fn dist(&self, _: &Vertex0D, rng: &mut ThreadRng) -> f64 {
        rng.gen()
    }

    fn new_rand(rng: &mut ThreadRng) -> Vertex0D {
        Vertex0D {
            id: rng.gen::<f64>().into(),
        }
    }
}

impl VertexCoord<Vertex2D> for Vertex2D {
    const DIMENSION: GraphDim = GraphDim::TwoD;

    fn dist(&self, other: &Vertex2D, _: &mut ThreadRng) -> f64 {
        [&other.x - &self.x, &other.y - &self.y]
            .into_iter()
            .map(|num| num.get().powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn new_rand(rng: &mut ThreadRng) -> Vertex2D {
        let (x, y): (f64, f64) = rng.gen();
        Vertex2D {
            x: x.into(),
            y: y.into(),
        }
    }
}

impl VertexCoord<Vertex3D> for Vertex3D {
    const DIMENSION: GraphDim = GraphDim::ThreeD;

    fn dist(&self, other: &Vertex3D, _: &mut ThreadRng) -> f64 {
        [&other.x - &self.x, &other.y - &self.y, &other.z - &self.z]
            .into_iter()
            .map(|num| num.get().powi(2))
            .sum::<f64>()
            .sqrt()
    }

    fn new_rand(rng: &mut ThreadRng) -> Vertex3D {
        let (x, y, z): (f64, f64, f64) = rng.gen();
        Vertex3D {
            x: x.into(),
            y: y.into(),
            z: z.into(),
        }
    }
}

impl VertexCoord<Vertex4D> for Vertex4D {
    const DIMENSION: GraphDim = GraphDim::FourD;

    fn dist(&self, other: &Vertex4D, _: &mut ThreadRng) -> f64 {
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
        let (r, g, b, a): (f64, f64, f64, f64) = rng.gen();
        Vertex4D {
            r: r.into(),
            g: g.into(),
            b: b.into(),
            a: a.into(),
        }
    }
}

impl GraphDim {
    pub fn get_max_edge_weight(self, num_vertices: usize) -> Weight<f64> {
        let num_vertices = num_vertices as f64;
        // Estimated by power series trend lines from samples from 128 to 32768:
        // https://docs.google.com/spreadsheets/d/1ILvyZYYi5nrMkP_qPR2DdfvpdP5haJmWxkBTr-FHyEo/edit?usp=sharing
        match self {
            Self::ZeroD => Weight::from(3. * num_vertices.powf(-0.75)),
            Self::TwoD => Weight::from(1.4 * num_vertices.powf(-0.38)),
            Self::ThreeD => Weight::from(1.25 * num_vertices.powf(-0.26)),
            Self::FourD => Weight::from(1.45 * num_vertices.powf(-0.2)),
        }
    }
}

impl TryFrom<usize> for GraphDim {
    type Error = PsetErr;
    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::ZeroD),
            2 => Ok(Self::TwoD),
            3 => Ok(Self::ThreeD),
            4 => Ok(Self::FourD),
            _ => Err(PsetErr::Cxt(format!(
                "{value} does not correspond to a supported graph dimension: 0, 2, 3, 4."
            ))),
        }
    }
}

impl From<GraphDim> for usize {
    fn from(value: GraphDim) -> Self {
        match value {
            GraphDim::ZeroD => 0,
            GraphDim::TwoD => 2,
            GraphDim::ThreeD => 3,
            GraphDim::FourD => 4,
        }
    }
}

#[cfg(test)]
mod graph_tests {
    use super::{CompleteUnitGraph, Graph, Vertex0D, Vertex2D, Vertex3D, Vertex4D, WeightedEdge};
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
                    .iter()
                    .find(|el| {
                        el == &&WeightedEdge {
                            vertex: vertex.clone(),
                            // Again, equality only looks at the vertex.
                            weight: Weight::new(-1.),
                        }
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
                let graph_1d = CompleteUnitGraph::graph_nd::<Vertex0D>(size, None);
                assert_complete_graph(&graph_1d, size);
                assert_edges_well_defined(&graph_1d);
            }
            {
                let graph_2d = CompleteUnitGraph::graph_nd::<Vertex2D>(size, None);
                assert_complete_graph(&graph_2d, size);
                assert_edges_well_defined(&graph_2d);
            }
            {
                let graph_3d = CompleteUnitGraph::graph_nd::<Vertex3D>(size, None);
                assert_complete_graph(&graph_3d, size);
                assert_edges_well_defined(&graph_3d);
            }
            {
                let graph_4d = CompleteUnitGraph::graph_nd::<Vertex4D>(size, None);
                assert_complete_graph(&graph_4d, size);
                assert_edges_well_defined(&graph_4d);
            }
        }
    }
}
