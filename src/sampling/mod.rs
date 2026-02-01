//! Stream sampling algorithms
//!
//! This module provides algorithms for maintaining random samples from streams.
//! Useful when you need representative samples but can't store all data.
//!
//! # Example
//!
//! ```
//! use flowstats::sampling::ReservoirSampler;
//!
//! let mut sampler = ReservoirSampler::<i32>::new(10);
//!
//! // Stream millions of items, keep uniform sample of 10
//! for i in 0..1_000_000 {
//!     sampler.add(i);
//! }
//!
//! // Each item had equal probability of being sampled
//! let sample = sampler.sample();
//! assert_eq!(sample.len(), 10);
//! ```

mod reservoir;

pub use reservoir::ReservoirSampler;
