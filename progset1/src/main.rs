use csv::Writer;
use progset1::{
    cli::{CliCommand, CollectStatsArgs, CollectedStat, GraphDim},
    error::PsetRes,
    graph_gen::{CompleteUnitGraph, Graph, Vertex0D, Vertex2D, Vertex3D, Vertex4D},
    mst::Mst,
};
use std::{
    env,
    time::{Duration, Instant},
};

/// Returns the average weight of a `dimension` graph of size `num_vertices` over `num_trials`
/// passes.
fn mst_average(num_trials: usize, num_vertices: usize, dimension: &GraphDim) -> f64 {
    let mut total_weight = 0.;
    let mut total_time = Duration::ZERO;
    for trial in 0..num_trials {
        let timer = Instant::now();
        // These cases are no fun, but I'd rather this than the performance hit of boxing and
        // dynamic dispatch on trait objects.
        match &dimension {
            GraphDim::ZeroD => {
                let graph: Graph<Vertex0D> = CompleteUnitGraph::graph_nd(num_vertices);
                if let Some(start) = graph.keys().next() {
                    let mst = Mst::from_prim(&graph, start);
                    total_weight += mst.weight;
                };
            }
            GraphDim::TwoD => {
                let graph: Graph<Vertex2D> = CompleteUnitGraph::graph_nd(num_vertices);
                if let Some(start) = graph.keys().next() {
                    let mst = Mst::from_prim(&graph, start);
                    total_weight += mst.weight;
                };
            }
            GraphDim::ThreeD => {
                let graph: Graph<Vertex3D> = CompleteUnitGraph::graph_nd(num_vertices);
                if let Some(start) = graph.keys().next() {
                    let mst = Mst::from_prim(&graph, start);
                    total_weight += mst.weight;
                };
            }
            GraphDim::FourD => {
                let graph: Graph<Vertex4D> = CompleteUnitGraph::graph_nd(num_vertices);
                if let Some(start) = graph.keys().next() {
                    let mst = Mst::from_prim(&graph, start);
                    total_weight += mst.weight;
                };
            }
        };
        let elapsed = timer.elapsed();
        total_time += elapsed;
        println!("trial: {}, duration: {:?}", trial, elapsed);
    }
    println!("average time: {:?}", total_time / num_trials as u32);
    total_weight / num_trials as f64
}

/// Follows the provided config, generating outputs that are written to a CSV
/// at the given location.
fn collect_stats(args: &CollectStatsArgs) -> PsetRes<()> {
    let mut wtr = Writer::from_path(args.output_filepath.as_path())?;
    for graph_dim in args.config.graph_dimensions.iter() {
        for graph_size in args.config.graph_sizes.iter() {
            println!("Dimension: {}, Size: {}", graph_dim, graph_size);
            let dimension: GraphDim = graph_dim.try_into()?;
            let start = Instant::now();
            let weight = mst_average(args.config.trials_per_size, *graph_size, &dimension);
            let res = CollectedStat {
                graph_size: *graph_size,
                graph_dimension: *graph_dim,
                num_trials: args.config.trials_per_size,
                runtime_secs: start.elapsed().as_secs(),
                weight,
            };
            println!("Finished in {} secs", res.runtime_secs);
            wtr.serialize(&res)?;
        }
    }
    wtr.flush()?;
    Ok(())
}

fn main() -> PsetRes<()> {
    let args: Vec<String> = env::args().collect();
    let cmd: CliCommand = args.as_slice().try_into()?;
    match cmd {
        CliCommand::MstAverage(args) => {
            let avg_weight = mst_average(args.num_trials, args.num_vertices, &args.graph_dimension);
            println!(
                "\naverage: {}\nnumpoints: {}\nnumtrials: {}\ndimension: {:?}",
                avg_weight, args.num_vertices, args.num_trials, args.graph_dimension
            );
        }
        CliCommand::CollectStats(args) => {
            collect_stats(&args)?;
        }
    }
    Ok(())
}
