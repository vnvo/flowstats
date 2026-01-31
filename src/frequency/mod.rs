//! Frequency estimation algorithms
//!
//! This module provides implementations of sketches for estimating item
//! frequencies in a data stream.
//!
//! # Algorithms
//!
//! - [`CountMinSketch`]: Classic count-min sketch with optional conservative update
//! - [`SpaceSaving`]: Top-K / heavy hitters tracking
//!
//! # Example
//!
//! ```
//! use flowstats::frequency::CountMinSketch;
//! use flowstats::traits::FrequencySketch;
//!
//! let mut cms = CountMinSketch::new(0.01, 0.001); // 1% error, 0.1% probability
//!
//! cms.add(b"item1", 5);
//! cms.add(b"item2", 3);
//!
//! let count = cms.estimate(b"item1");
//! println!("Estimated count: {}", count);
//! ```

mod count_min;
mod space_saving;

pub use count_min::CountMinSketch;
pub use space_saving::SpaceSaving;
