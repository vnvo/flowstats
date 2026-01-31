//! HyperLogLog cardinality estimator
//!
//! Implementation of the HyperLogLog algorithm with bias correction
//! for small cardinalities.

use crate::traits::{CardinalitySketch, ErrorBounds, MergeError, Sketch};
use xxhash_rust::xxh3::xxh3_64;

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// HyperLogLog cardinality estimator
///
/// Estimates the number of distinct elements with configurable precision.
/// Memory usage is 2^precision bytes.
///
/// # Error Rate
///
/// The relative standard error is approximately 1.04 / sqrt(m) where m = 2^precision.
///
/// | Precision | Memory | Error |
/// |-----------|--------|-------|
/// | 10 | 1 KB | ~3.25% |
/// | 12 | 4 KB | ~1.63% |
/// | 14 | 16 KB | ~0.81% |
/// | 16 | 64 KB | ~0.41% |
/// | 18 | 256 KB | ~0.20% |
///
/// # Example
///
/// ```
/// use flowstats::cardinality::HyperLogLog;
/// use flowstats::traits::CardinalitySketch;
///
/// let mut hll = HyperLogLog::new(12);
///
/// for i in 0..10000 {
///     hll.insert(&format!("user_{}", i));
/// }
///
/// let count = hll.estimate();
/// println!("Approximately {} distinct users", count);
/// ```
#[derive(Clone, Debug)]
pub struct HyperLogLog {
    /// Precision parameter (4-18)
    precision: u8,
    /// Registers (one byte per register)
    registers: Vec<u8>,
    /// Number of items inserted
    count: u64,
}

impl HyperLogLog {
    /// Create a new HyperLogLog with the given precision
    ///
    /// Precision must be between 4 and 18 inclusive.
    /// Higher precision gives better accuracy but uses more memory.
    ///
    /// # Panics
    ///
    /// Panics if precision is not in range [4, 18]
    pub fn new(precision: u8) -> Self {
        assert!(
            (4..=18).contains(&precision),
            "precision must be between 4 and 18"
        );

        let m = 1usize << precision;
        Self {
            precision,
            registers: vec![0u8; m],
            count: 0,
        }
    }

    /// Create a HyperLogLog targeting a specific error rate
    ///
    /// The error rate is approximate and represents the relative standard error.
    pub fn with_error(target_error: f64) -> Self {
        let precision = super::precision_for_error(target_error);
        Self::new(precision)
    }

    /// Get the precision parameter
    pub fn precision(&self) -> u8 {
        self.precision
    }

    /// Get the number of registers (m = 2^precision)
    pub fn num_registers(&self) -> usize {
        self.registers.len()
    }

    /// Insert an item by its bytes
    pub fn insert(&mut self, item: &str) {
        self.insert_bytes(item.as_bytes());
    }

    /// Insert raw bytes
    pub fn insert_bytes(&mut self, bytes: &[u8]) {
        let hash = xxh3_64(bytes);
        self.insert_hash(hash);
    }

    /// Insert a pre-computed hash value
    pub fn insert_hash(&mut self, hash: u64) {
        self.count += 1;

        // Use first p bits for register index
        let idx = (hash >> (64 - self.precision)) as usize;

        // Count leading zeros in remaining bits + 1
        let w = hash << self.precision | (1u64 << (self.precision - 1));
        let rho = w.leading_zeros() as u8 + 1;

        // Update register if new value is larger
        if rho > self.registers[idx] {
            self.registers[idx] = rho;
        }
    }

    /// Raw estimate using harmonic mean
    fn raw_estimate(&self) -> f64 {
        let m = self.registers.len() as f64;

        // Compute harmonic mean of 2^(-register[i])
        let sum: f64 = self.registers.iter().map(|&r| 2f64.powi(-(r as i32))).sum();

        // Apply alpha constant
        let alpha = self.alpha_m();
        alpha * m * m / sum
    }

    /// Alpha constant for given m
    fn alpha_m(&self) -> f64 {
        let m = self.registers.len();
        match m {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m as f64),
        }
    }

    /// Count registers with value 0
    fn count_zeros(&self) -> usize {
        self.registers.iter().filter(|&&r| r == 0).count()
    }

    /// Linear counting estimate for small cardinalities
    fn linear_counting(&self, zeros: usize) -> f64 {
        let m = self.registers.len() as f64;
        m * (m / zeros as f64).ln()
    }

    /// Bias correction for HyperLogLog++
    fn bias_correction(&self, raw: f64) -> f64 {
        // Simplified bias correction
        // Full HLL++ uses empirical bias tables
        let m = self.registers.len() as f64;

        // Threshold for switching to linear counting
        let threshold = 2.5 * m;

        if raw <= threshold {
            let zeros = self.count_zeros();
            if zeros > 0 {
                // Use linear counting for small cardinalities
                let lc = self.linear_counting(zeros);
                if lc <= threshold {
                    return lc;
                }
            }
        }

        // Large range: no correction needed
        // (In full HLL++, we'd also handle extremely large cardinalities here)
        raw
    }
}

impl Sketch for HyperLogLog {
    type Item = [u8];

    fn update(&mut self, item: &[u8]) {
        self.insert_bytes(item);
    }

    fn merge(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.precision != other.precision {
            return Err(MergeError::IncompatibleConfig {
                expected: format!("precision={}", self.precision),
                found: format!("precision={}", other.precision),
            });
        }

        // Take element-wise max
        for (a, &b) in self.registers.iter_mut().zip(other.registers.iter()) {
            *a = (*a).max(b);
        }

        self.count += other.count;
        Ok(())
    }

    fn clear(&mut self) {
        self.registers.fill(0);
        self.count = 0;
    }

    fn size_bytes(&self) -> usize {
        self.registers.len() + core::mem::size_of::<Self>()
    }

    fn count(&self) -> u64 {
        self.count
    }
}

impl CardinalitySketch for HyperLogLog {
    fn estimate(&self) -> f64 {
        if self.count == 0 {
            return 0.0;
        }

        let raw = self.raw_estimate();
        self.bias_correction(raw)
    }

    fn error_bounds(&self, confidence: f64) -> ErrorBounds {
        let estimate = self.estimate();
        let rse = self.relative_error();

        // Convert confidence to z-score (approximate)
        let z = match confidence {
            c if c >= 0.99 => 2.576,
            c if c >= 0.95 => 1.96,
            c if c >= 0.90 => 1.645,
            c if c >= 0.80 => 1.282,
            _ => 1.0,
        };

        let margin = z * rse * estimate;
        ErrorBounds::new(
            (estimate - margin).max(0.0),
            estimate,
            estimate + margin,
            confidence,
        )
    }

    fn relative_error(&self) -> f64 {
        let m = self.registers.len() as f64;
        1.04 / m.sqrt()
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for HyperLogLog {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("HyperLogLog", 3)?;
        state.serialize_field("precision", &self.precision)?;
        state.serialize_field("registers", &self.registers)?;
        state.serialize_field("count", &self.count)?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for HyperLogLog {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct HllData {
            precision: u8,
            registers: Vec<u8>,
            count: u64,
        }

        let data = HllData::deserialize(deserializer)?;
        Ok(HyperLogLog {
            precision: data.precision,
            registers: data.registers,
            count: data.count,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let mut hll = HyperLogLog::new(12);

        for i in 0..10000 {
            hll.insert(&format!("item_{}", i));
        }

        let estimate = hll.estimate();
        // Should be within 10% of actual
        assert!(estimate > 9000.0 && estimate < 11000.0);
    }

    #[test]
    fn test_empty() {
        let hll = HyperLogLog::new(12);
        assert_eq!(hll.estimate(), 0.0);
    }

    #[test]
    fn test_duplicates() {
        let mut hll = HyperLogLog::new(12);

        // Insert same item many times
        for _ in 0..10000 {
            hll.insert("same_item");
        }

        let estimate = hll.estimate();
        // Should be close to 1
        assert!(estimate >= 0.5 && estimate <= 2.0);
    }

    #[test]
    fn test_merge() {
        let mut hll1 = HyperLogLog::new(12);
        let mut hll2 = HyperLogLog::new(12);

        // Insert different items
        for i in 0..5000 {
            hll1.insert(&format!("a_{}", i));
        }
        for i in 0..5000 {
            hll2.insert(&format!("b_{}", i));
        }

        let est1 = hll1.estimate();
        let est2 = hll2.estimate();

        hll1.merge(&hll2).unwrap();
        let merged_est = hll1.estimate();

        // Merged should be approximately sum (no overlap)
        assert!(merged_est > est1);
        assert!(merged_est > est2);
        assert!(merged_est > 9000.0 && merged_est < 11000.0);
    }

    #[test]
    fn test_merge_incompatible() {
        let mut hll1 = HyperLogLog::new(12);
        let hll2 = HyperLogLog::new(14);

        assert!(hll1.merge(&hll2).is_err());
    }

    #[test]
    fn test_precision() {
        let hll = HyperLogLog::new(14);
        assert_eq!(hll.precision(), 14);
        assert_eq!(hll.num_registers(), 16384);
    }

    #[test]
    fn test_error_bounds() {
        let mut hll = HyperLogLog::new(14);

        for i in 0..100000 {
            hll.insert(&format!("item_{}", i));
        }

        let bounds = hll.error_bounds(0.95);
        assert!(bounds.lower < bounds.estimate);
        assert!(bounds.estimate < bounds.upper);

        // True value (100000) should be within bounds most of the time
        // (This is probabilistic, but with 14-bit precision it should be close)
        assert!(bounds.lower < 110000.0);
        assert!(bounds.upper > 90000.0);
    }

    #[test]
    fn test_small_cardinalities() {
        let mut hll = HyperLogLog::new(12);

        // Small number of items - linear counting should kick in
        for i in 0..100 {
            hll.insert(&format!("item_{}", i));
        }

        let estimate = hll.estimate();
        // Linear counting is more accurate for small cardinalities
        assert!(estimate > 80.0 && estimate < 120.0);
    }

    #[test]
    fn test_clear() {
        let mut hll = HyperLogLog::new(12);

        for i in 0..1000 {
            hll.insert(&format!("item_{}", i));
        }

        assert!(hll.estimate() > 0.0);

        hll.clear();
        assert_eq!(hll.estimate(), 0.0);
        assert_eq!(hll.count(), 0);
    }

    #[test]
    fn test_with_error() {
        let hll = HyperLogLog::with_error(0.01); // Target 1% error
        assert!(hll.precision() >= 13); // Should select appropriate precision
    }
}
