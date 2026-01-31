//! Quantile estimation algorithms
//!
//! This module provides implementations of sketches for estimating quantiles
//! (percentiles) in a data stream.
//!
//! # Algorithms
//!
//! - [`TDigest`]: Mergeable quantile sketch with high accuracy at extremes
//!
//! # Example
//!
//! ```
//! use flowstats::quantiles::TDigest;
//! use flowstats::traits::QuantileSketch;
//!
//! let mut digest = TDigest::new(100.0);
//!
//! for value in [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0] {
//!     digest.add(value);
//! }
//!
//! let median = digest.median();
//! println!("Median: {:?}", median);
//! ```

mod tdigest;

pub use tdigest::{Centroid, TDigest};