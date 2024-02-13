use csv::Writer;
use progset1::{
    cli::{CliCommand, CollectStatsArgs, CollectedStat, GraphPerf},
    error::{PsetErr, PsetRes},
    graph_gen::{
        CompleteUnitGraph, Graph, GraphDim, Vertex0D, Vertex2D, Vertex3D, Vertex4D, VertexCoord,
    },
    mst::Mst,
    Weight,
};
use std::{
    env,
    fmt::Debug,
    hash::Hash,
    thread::{self, JoinHandle},
    time::{Duration, Instant},
};

/// Processes CLI commands related to Prim's algorithm. Call `cargo run` with no options
/// for a list of options.
fn main() -> PsetRes<()> {
    let args: Vec<String> = env::args().collect();
    let cmd: CliCommand = args.as_slice().try_into()?;
    match cmd {
        CliCommand::MstAverage(args) => {
            let avg_weight = mst_average(
                args.num_trials,
                args.num_vertices,
                args.graph_dimension,
                args.trimming,
            );
            println!(
                "\naverage: {:?}\nnumpoints: {}\nnumtrials: {}\ndimension: {:?}",
                avg_weight?, args.num_vertices, args.num_trials, args.graph_dimension
            );
        }
        CliCommand::CollectStats(args) => {
            collect_stats(&args)?;
        }
    }
    Ok(())
}

/// Returns the average weight of a `dimension` graph of size `num_vertices` over `num_trials`
/// passes. This function is parallelized to run all trials at once.
fn mst_average(
    num_trials: usize,
    num_vertices: usize,
    dimension: GraphDim,
    trimming: GraphPerf,
) -> PsetRes<Weight<f64>> {
    let mut total_weight = 0.;
    let mut total_time = Duration::ZERO;

    println!("\nspawning {num_trials} trials, cfg: {:?}", trimming);
    let mut handles: Vec<JoinHandle<(Weight<f64>, Duration)>> = Vec::with_capacity(num_trials);
    for _ in 0..num_trials {
        handles.push(thread::spawn(move || match dimension {
            GraphDim::ZeroD => run_trial::<Vertex0D>(num_vertices, trimming),
            GraphDim::TwoD => run_trial::<Vertex2D>(num_vertices, trimming),
            GraphDim::ThreeD => run_trial::<Vertex3D>(num_vertices, trimming),
            GraphDim::FourD => run_trial::<Vertex4D>(num_vertices, trimming),
        }));
    }

    for (ind, handle) in handles.into_iter().enumerate() {
        let (weight, time) = handle
            .join()
            .map_err(|e| PsetErr::Cxt(format!("{:?}", e)))?;
        total_weight += weight.get();
        total_time += time;
        println!(
            "trial: {ind}, num_vertices: {num_vertices}, dimension: {:?}, time: {:?}",
            dimension, time
        );
    }
    println!("average time: {:?}", total_time / num_trials as u32);
    Ok((total_weight / num_trials as f64).into())
}

/// Given a number of vertices and the graph variant, builds a graph of
/// that size, computes the MST, and returns the weight of the MST and
/// the time required to build it.
fn run_trial<V>(num_vertices: usize, trimming: GraphPerf) -> (Weight<f64>, Duration)
where
    V: VertexCoord<V> + Eq + Hash + Debug + Default,
{
    let timer = Instant::now();
    // Comment this out if no trimming
    let trim_threshold = match trimming {
        GraphPerf::WithoutTrim => None,
        GraphPerf::WithTrim => Some(V::DIMENSION.get_max_edge_weight(num_vertices).take()),
    };
    let graph: Graph<V> = CompleteUnitGraph::graph_nd(num_vertices, trim_threshold);
    let weight = match graph.keys().next() {
        Some(start) => Mst::from_prim(&graph, start).weight.into(),
        None => Weight::new(0.),
    };
    (weight, timer.elapsed())
}

/// Follows the provided config, generating outputs that are written to a CSV
/// at the given location.
fn collect_stats(args: &CollectStatsArgs) -> PsetRes<()> {
    let mut wtr = Writer::from_path(args.output_filepath.as_path())?;
    for graph_dim in args.config.graph_dimensions.iter().copied() {
        for graph_size in args.config.graph_sizes.iter().copied() {
            let dimension: GraphDim = graph_dim.try_into()?;
            let start = Instant::now();
            let weight = mst_average(
                args.config.trials_per_size,
                graph_size,
                dimension,
                GraphPerf::WithTrim,
            )?;
            let res = CollectedStat {
                graph_size,
                graph_dimension: graph_dim,
                num_trials: args.config.trials_per_size,
                runtime_secs: start.elapsed().as_secs(),
                weight: weight.take(),
            };
            wtr.serialize(&res)?;
            wtr.flush()?;
        }
    }
    wtr.flush()?;
    Ok(())
}
