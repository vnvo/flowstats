//! Statistical summaries for streaming data
//!
//! This module provides algorithms for computing statistics over streams
//! in a single pass with constant memory.
//!
//! # Example
//!
//! ```
//! use flowstats::statistics::RunningStats;
//!
//! let mut stats = RunningStats::new();
//!
//! for value in [1.0, 2.0, 3.0, 4.0, 5.0] {
//!     stats.add(value);
//! }
//!
//! println!("Mean: {}", stats.mean());
//! println!("Stddev: {}", stats.stddev());
//! println!("Min: {:?}", stats.min());
//! println!("Max: {:?}", stats.max());
//! ```

mod moments;

pub use moments::RunningStats;