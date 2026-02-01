//! Cardinality (distinct count) estimation algorithms
//!
//! This module provides implementations of sketches for estimating the number
//! of distinct elements in a data stream.
//!
//! # Algorithms
//!
//! - [`HyperLogLog`]: The classic HLL algorithm with bias correction
//!
//! # Example
//!
//! ```
//! use flowstats::cardinality::HyperLogLog;
//! use flowstats::traits::CardinalitySketch;
//!
//! let mut hll = HyperLogLog::new(14); // ~0.8% error
//!
//! for i in 0..10000 {
//!     hll.insert(&i.to_string());
//! }
//!
//! let estimate = hll.estimate();
//! println!("estimated distinct count: {}", estimate);
//! ```

mod hyperloglog;

pub use hyperloglog::HyperLogLog;

/// Compute the required precision for a target error rate
///
/// HLL error is approximately 1.04 / sqrt(2^p)
pub fn precision_for_error(target_error: f64) -> u8 {
    // error = 1.04 / sqrt(m) where m = 2^p
    // sqrt(m) = 1.04 / error
    // m = (1.04 / error)^2
    // p = log2(m)
    let m = (1.04 / target_error).powi(2);
    let p = m.log2().ceil() as u8;
    p.clamp(4, 18)
}

/// Compute the memory usage for a given precision
pub fn memory_for_precision(precision: u8) -> usize {
    1usize << precision
}

/// Compute the expected error for a given precision
pub fn error_for_precision(precision: u8) -> f64 {
    let m = (1usize << precision) as f64;
    1.04 / m.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_for_error() {
        // 1% error should give precision around 14
        let p = precision_for_error(0.01);
        assert!(p >= 13 && p <= 15);

        // 0.5% error should give higher precision
        let p2 = precision_for_error(0.005);
        assert!(p2 > p);
    }

    #[test]
    fn test_error_for_precision() {
        let e14 = error_for_precision(14);
        assert!(e14 > 0.007 && e14 < 0.009); // ~0.81%

        let e12 = error_for_precision(12);
        assert!(e12 > e14); // Lower precision = higher error
    }
}
