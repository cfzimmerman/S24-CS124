# Average MST weight of n-dimension unit graphs
*Cory Zimmerman, February 2024*

This project generates multi-dimensional complete graphs and computes data about the MSTs of those graphs. Running the it requires a local installation of [Rust](https://www.rust-lang.org/tools/install). It can be run and tested with Cargo or built using Make as requested in the assignment pdf. Running the executable with no arguments will print options.

If you’re new to Rust, here’s an overview of the project structure:
- [src/main.rs](src/main.rs) is the executable’s entry point. It consumes the API defined in [src/lib.rs](src/lib.rs). 
- [src/lib.rs](src/lib.rs) defines the API from the library. Everything else in the project is library code. I’ve documented the various modules there.
- My unit tests live at the bottom of module files. They can be invoked with `make test` or `cargo test`. 
- I optimized graph generation pretty heavily. Flamegraphs (numerically ordered by creation) and notes can be found in [perf](perf).
- CLI variant `1` generates a CSV of graph stats. That can be found in [file_io](file_io).

### What algorithm and why?
Because complete graphs have many more edges than vertices, I chose Prim’s algorithm with `O(E log V)` binary heap complexity over Kruskal’s `O(E log E)` time complexity.

### Are the growth rates surprising? Can you come up with an explanation for them?
From the analysis in this [spreadsheet](https://docs.google.com/spreadsheets/d/19qFHtJvxhMoOoyb4R_D_k1B6aBKsx1p6voae2ElizY0/edit?usp=sharing), I estimate the following functional forms of the average weight where `n` is the number of vertices in the graph. Note that here and elsewhere, when I refer to graphs of dimension "0", that's just a reflection of the requested CLI spec. The "0 dimension" graph is fundamentally different from the 2, 3, and 4 dimension graphs. 

Estimated functions:
- 0d: `f(n) = 1.2`
- 2d: `f(n) = 0.68n^(0.496)`
- 3d: `f(n) = 0.708n^(0.659)`
- 4d: `f(n) = 0.78n^(0.739)`

The first graph was definitely surprising but makes sense upon consideration. The vertices in the "0d" graph have no spatial relation due to random edge selection. As the vertex count increases, there are correspondingly more edges, each with probability of being lighter than edges on the MST for a smaller `n`. Thus, as vertex density increases, the availability of increasingly lighter edges grows in parallel keeping the average weight generally constant.

The other three graphs occupy geometric vector spaces. As more vertices fill the graph, the vector space becomes increasingly saturated, and more spatial distance must be traversed in order for the spanning tree to visit every vertex. Because the 2d, 3d, and 4d graphs have different spatial volumes, it makes sense that the average MST weights are ordered smallest to largest.

### How long does it take your algorithm to run? Does this make sense? Do you notice things like the cache size of your computer having an effect.

This [CSV](/file_io/report.csv) has full timings. A complete graph has `n(n - 1) / 2` edges, which grows huge as input scales. For graphs as large as those tested, high runtimes are unsurprising. Across trials, I noticed that "0d" generation took extra time because it required random number generation per edge (rather than per vertex). 

On my laptop, this program is definitely memory bound, and I absolutely noticed that affect on runtime. On larger trials, RAM becomes saturated, and the OS begins swapping pages in and out of disk. I can see it in Activity Monitor, and it has a huge impact on runtime.

### Correctness, runtime, and results
The assignment requests a table of results. My submission is [this spreadsheet](https://docs.google.com/spreadsheets/d/19qFHtJvxhMoOoyb4R_D_k1B6aBKsx1p6voae2ElizY0/edit).

I didn't make any intentional changes to Prim's algorithm, and I only modified the definition of a binary heap to support `O(1)` lookup via hash table. Untrimmed graph generation just iterates every vertex for every pair of edges, so the correctness of my code should reduce to that of the algorithms themselves. The accuracy of my implementation is also indicated by several unit tests.

For untrimmed graphs with n vertices, generating the graph requires `O(n)` iterations. The number of edges is `n(n - 1) / 2 = O(n^2)` because it's complete. Prim's algorithm takes `O(E log V)`, which simplifies to `O(n^2 log n)` in this instance, yielding complexity of `O(n^2 log n)` on untrimmed graphs. 

Graph trimming is based on projections and heuristics with probablistic correctness, so I'll decline to provide an explicit runtime beyond noting that it drastically speeds up MST calculation  with a small chance of producing incorrect results. I arrived at my trimming functions around line 230 of [src/graph_gen.rs](src/graph_gen.rs#230) by writing data from the [heaviest_edge](src/mst.rs) test into a CSV and determining trend lines on this [spreadsheet](https://docs.google.com/spreadsheets/d/1ILvyZYYi5nrMkP_qPR2DdfvpdP5haJmWxkBTr-FHyEo/edit).

### Reflections:

Here are some of my different milestones and what I learned along the way:
- I started by building [PrimHeap](src/prim_heap.rs). Optimizing for `upsert_min` required more complex code and a hashmap supporting the heap vector, but it turned out well.
- I had initially written separate graph generation functions for different dimensions. I eventually used traits and generics to arrive at the single graph generator in [graph_gen](src/graph_gen.rs).
- With graphs and a heap, [Prim’s algorithm](src/mst.rs) was straightforward. It's well tested, and I’m confident of correctness on untrimmed graphs.
- At that point, I thought I was done. I added the [CLI parser](src/cli.rs) with handlers in [main](src/main.rs) and let it run overnight. By morning, my program was still working on 32k for dimension zero.
**todo: resume here**
- I thought the issue was code performance, so I optimized it pretty hard. Feel free to check out the notes and flame graphs in the perf folder. By focusing mostly on graph generation, I brought single-threaded runtimes down by 20 to 80 percent depending on graph dimension. With this, a single threaded run hit 32k vertices pretty comfortably but seemed to have a sharp growth trajectory.
- Running five trials of the same thing is a huge invitation to parallelize, so I added thread spawning for each trial. This made 16k even faster, but running 5 threads at 32k caused my OS to kill my process within a minute due to colossal memory usage. On single threaded attempts, I could also definitely see the point at which the OA started swapping pages between RAN and disk 
- This stumped me. After eventually rereading the assignment, I decided to try trimming. Trimming everything by 50 percent didn’t help much. I added a test function to generate maximum mst edge weights in graphs of different sizes and dimensions and spent a while trying to determine a formula for max edge weight given size and dimension. Those trials didn’t turn out, so I eventually just wrote the data to a CSV and used Google Sheets to determine a reasonable trend line. The power series model did best. I used those new trim factors to produce [file_io/results.csv](file_io/results.csv).
