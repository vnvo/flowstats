# flowstats

Streaming algorithms for Rust.

[![Crates.io](https://img.shields.io/crates/v/flowstats.svg)](https://crates.io/crates/flowstats)
[![Documentation](https://docs.rs/flowstats/badge.svg)](https://docs.rs/flowstats)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](README.md#license)

## Overview

`flowstats` provides space-efficient probabilistic data structures for processing data streams. These algorithms trade a small amount of accuracy for dramatic reductions in memory usage, making them ideal for:

- **Real-time analytics** - Count distinct users, track popular items
- **Monitoring systems** - Compute percentiles, detect anomalies
- **Data pipelines** - Aggregate statistics across distributed workers
- **Resource-constrained environments** - Process unbounded streams with fixed memory

Additionally: 
- All structures are **mergeable** for distributed computation
- It supports `no_std` environments, see no_std support section below.

## Features

| Category | Algorithm | Use Case |
|----------|-----------|----------|
| **Cardinality** | HyperLogLog | Count distinct elements (~1% error in 16KB) |
| **Frequency** | Count-Min Sketch | Estimate item frequencies |
| **Frequency** | Space-Saving | Find top-k frequent items |
| **Quantiles** | t-digest | Estimate percentiles (p50, p99, etc.) |
| **Membership** | Bloom Filter | Test set membership (no false negatives) |
| **Sampling** | Reservoir Sampler | Uniform random sample from stream |
| **Statistics** | RunningStats | Mean, variance, min, max in one pass |

## Installation

```toml
[dependencies]
flowstats = "0.1"
```

Or with specific features:

```toml
[dependencies]
flowstats = { version = "0.1", default-features = false, features = ["cardinality", "frequency"] }
```

## Quick Start

### Count Distinct Elements (HyperLogLog)

```rust
use flowstats::HyperLogLog;
use flowstats::traits::CardinalitySketch;

let mut hll = HyperLogLog::new(14); // ~0.8% error, 16KB memory

for user_id in user_events {
    hll.insert(&user_id);
}

println!("Unique users: {}", hll.estimate()); // ≈ actual count ± 0.8%
```

### Estimate Percentiles (t-digest)

```rust
use flowstats::TDigest;
use flowstats::traits::QuantileSketch;

let mut digest = TDigest::new(100.0);

for latency in request_latencies {
    digest.add(latency);
}

println!("p50: {:?}", digest.quantile(0.5));
println!("p99: {:?}", digest.quantile(0.99));
```

### Find Top-K Items (Space-Saving)

```rust
use flowstats::SpaceSaving;

let mut top_k = SpaceSaving::new(100); // Track top 100

for page in page_views {
    top_k.add(page);
}

for (page, count) in top_k.top_k(10) {
    println!("{}: {} views", page, count);
}
```

### Test Set Membership (Bloom Filter)

```rust
use flowstats::BloomFilter;

let mut seen = BloomFilter::new(100_000, 0.01); // 1% false positive rate

for item in stream {
    if seen.contains(item.as_bytes()) {
        // Probably seen before (might be false positive)
    } else {
        // Definitely new
        seen.insert(item.as_bytes());
    }
}
```

### Estimate Frequencies (Count-Min Sketch)

```rust
use flowstats::CountMinSketch;

let mut cms = CountMinSketch::new(0.001, 0.01); // ε=0.1%, δ=1%

for event in events {
    cms.add(event.as_bytes());
}

let freq = cms.estimate(b"login");
println!("Login events: ≤ {}", freq); // Upper bound on true count
```

### Random Sampling (Reservoir)

```rust
use flowstats::ReservoirSampler;

let mut sampler = ReservoirSampler::new(1000); // Keep 1000 samples

for record in unlimited_stream {
    sampler.add(record);
}

// Each record had equal probability of being sampled
let sample = sampler.sample();
```

### Running Statistics

```rust
use flowstats::RunningStats;

let mut stats = RunningStats::new();

for value in measurements {
    stats.add(value);
}

println!("Mean: {}", stats.mean());
println!("Stddev: {}", stats.stddev());
println!("Min: {:?}, Max: {:?}", stats.min(), stats.max());
```

## Distributed Aggregation

All sketches support merging for distributed computation:

```rust
use flowstats::HyperLogLog;
use flowstats::traits::{CardinalitySketch, Sketch};

// Worker 1
let mut hll1 = HyperLogLog::new(14);
for id in shard_1 { hll1.insert(&id); }

// Worker 2
let mut hll2 = HyperLogLog::new(14);
for id in shard_2 { hll2.insert(&id); }

// Coordinator
hll1.merge(&hll2).unwrap();
println!("Total unique: {}", hll1.estimate());
```

## Algorithm Selection Guide

| Need | Algorithm | Memory | Error |
|------|-----------|--------|-------|
| Count unique items | HyperLogLog | 16 KB | ~0.8% |
| Percentiles (p50, p99) | t-digest | ~10 KB | ~1% at tails |
| Top-k items | Space-Saving | O(k) | Guaranteed if freq > n/k |
| Item frequencies | Count-Min Sketch | ~100 KB | Overestimates by ε·n |
| Set membership | Bloom Filter | ~1 MB/100k items | 1% false positive |
| Random sample | Reservoir | O(k) | Exact uniform sample |
| Mean/variance | RunningStats | 40 bytes | Exact |

## Feature Flags

```toml
[features]
default = ["std", "cardinality", "frequency", "quantiles"]
full = ["cardinality", "frequency", "quantiles", "membership", "sampling", "statistics"]

# Algorithm families
cardinality = []   # HyperLogLog
frequency = []     # Count-Min Sketch, Space-Saving
quantiles = []     # t-digest
membership = []    # Bloom Filter
sampling = []      # Reservoir Sampler
statistics = []    # RunningStats

# Optional
std = []           # Use standard library (disable for no_std)
serde = ["dep:serde"]  # Serialization support
```

## `no_std` Support

`flowstats` supports `no_std` environments that have the `alloc` crate.

**Cargo.toml:**
```toml
[dependencies]
flowstats = { version = "0.1", default-features = false, features = ["cardinality"] }
```

**Usage:**
```rust
#![no_std]
extern crate alloc;

use alloc::format;
use flowstats::HyperLogLog;
use flowstats::traits::CardinalitySketch;

// You must provide a global allocator for your target

fn count_distinct() -> f64 {
    let mut hll = HyperLogLog::new(10); // Use lower precision for memory-constrained environments
    
    for i in 0..100u32 {
        hll.insert(&format!("{}", i));
    }
    
    hll.estimate()
}
```

**Requirements:**
- Target must support the `alloc` crate (heap allocation)
- You must provide a global allocator (`#[global_allocator]`)
- `libm` is included automatically for floating-point math

## Parallel Processing

Sketches are not internally synchronized, but they're designed for parallel workloads via the merge pattern:

```rust
use std::thread;
use flowstats::HyperLogLog;
use flowstats::traits::{CardinalitySketch, Sketch};

// One sketch per thread, merge at the end
let handles: Vec<_> = data_shards.into_iter().map(|shard| {
    thread::spawn(move || {
        let mut hll = HyperLogLog::new(14);
        for item in shard {
            hll.insert(&item);
        }
        hll
    })
}).collect();

// Merge results
let mut combined = HyperLogLog::new(14);
for handle in handles {
    combined.merge(&handle.join().unwrap()).unwrap();
}
```

For concurrent access to a single sketch, wrap in `Arc<Mutex<_>>`.

## Error Handling

- **NaN values**: Silently ignored in t-digest and RunningStats
- **Merge errors**: Return `MergeError` for incompatible configurations
- **Serde validation**: Deserialize validates invariants

## Roadmap
- Cuckoo Filter (better than Bloom for deletions)
- Theta Sketch (set operations on cardinalities)
- MinHash (Jaccard similarity)
- DDSketch (relative error quantiles)
- KLL Sketch (alternative quantiles)
- SIMD optimizations

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## References

- [HyperLogLog: the analysis of a near-optimal cardinality estimation algorithm](http://algo.inria.fr/flajolet/Publications/FlFuGaMe07.pdf)
- [Computing Extremely Accurate Quantiles Using t-Digests](https://arxiv.org/abs/1902.04023)
- [An Improved Data Stream Summary: The Count-Min Sketch](https://dimacs.rutgers.edu/~graham/pubs/papers/cm-full.pdf)
- [Efficient Computation of Frequent and Top-k Elements in Data Streams](https://www.cse.ust.hk/~raMDNL/topk.pdf)