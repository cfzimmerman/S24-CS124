### Performance diffs

- `sudo cargo flamegraph -- 0 8192 5 0`

  - 17.51 seconds: initial
  - 14.80 secs avg: moved HashSet to outside the inner loop to only insert once per node in 1d graph_gen.
  - 9.85 secs avg: changed intermediate HashSet into a pre-sized vector, collected into HashSet at the end of the inner loop in 1d graph_gen.
  - 9.44 secs avg: generate both sides of an edge for each pass, decreasing total iterations of edge generation in 1d graph_gen.
  - 2.85 secs avg: store edges in a vec instead of a hash set.

- `sudo cargo flamegraph -- 0 16384 5 4`
  - 23.54 secs: initial
  - 18.69 secs: merge 0d improvements to nd
  - 11.36 secs (56.81 secs total): parallelized trials
  - 166.43 ms (832.15 ms total): with trimming, validated correct up to 32k
  - 160.55 ms (802.74 ms total): with tighter trimming estimate
