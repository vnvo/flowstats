//! HyperLogLog cardinality estimator
//!
//! Implementation of the HyperLogLog algorithm with linear counting
//! correction for small cardinalities.

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
    /// Number of insert operations (not distinct items)
    num_inserts: u64,
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
            num_inserts: 0,
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
        self.num_inserts += 1;

        let p = self.precision as u32;

        // Use top p bits for register index
        let idx = (hash >> (64 - p)) as usize;

        // Remaining bits moved to MSB positions
        let w = hash << p;

        // rho = leading zeros + 1, clamped to valid range [1, 64-p+1]
        // This handles the edge case where remaining bits are all zeros
        let max_rho = (64 - p + 1) as u8;
        let rho = ((w.leading_zeros() + 1) as u8).min(max_rho);

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

    /// Small-range correction using linear counting
    ///
    /// For small cardinalities (raw estimate <= 2.5m), uses linear counting
    /// which is more accurate. This is a standard HyperLogLog correction,
    /// not the full HLL++ empirical bias tables.
    ///
    /// Note: Large-range correction (for cardinalities near 2^32) is not
    /// implemented as it's rarely needed for typical use cases.
    fn bias_correction(&self, raw: f64) -> f64 {
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

        // No correction for normal/large ranges
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

        self.num_inserts += other.num_inserts;
        Ok(())
    }

    fn clear(&mut self) {
        self.registers.fill(0);
        self.num_inserts = 0;
    }

    fn size_bytes(&self) -> usize {
        self.registers.len() + core::mem::size_of::<Self>()
    }

    fn count(&self) -> u64 {
        self.num_inserts
    }
}

impl CardinalitySketch for HyperLogLog {
    fn estimate(&self) -> f64 {
        if self.num_inserts == 0 {
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
        state.serialize_field("num_inserts", &self.num_inserts)?;
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
            num_inserts: u64,
        }

        let data = HllData::deserialize(deserializer)?;

        // Validate precision is in valid range
        if !(4..=18).contains(&data.precision) {
            return Err(serde::de::Error::custom(
                "precision must be between 4 and 18",
            ));
        }

        // Validate registers length matches precision
        let expected_len = 1usize << data.precision;
        if data.registers.len() != expected_len {
            return Err(serde::de::Error::custom(format!(
                "invalid register length: expected {}, got {}",
                expected_len,
                data.registers.len()
            )));
        }

        Ok(HyperLogLog {
            precision: data.precision,
            registers: data.registers,
            num_inserts: data.num_inserts,
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

    #[test]
    fn test_rho_edge_case_all_zeros() {
        // Test that rho doesn't overflow when remaining bits are all zeros
        let mut hll = HyperLogLog::new(14);

        // Hash where remaining bits after index extraction are all zeros
        // For precision 14, index uses top 14 bits, leaving 50 bits
        // If those 50 bits are all zero, rho should be clamped to 51 (64-14+1)
        let hash_with_zero_suffix =
            0b1111111111111100_0000000000000000_0000000000000000_0000000000000000u64;
        hll.insert_hash(hash_with_zero_suffix);

        // Max valid rho for precision 14 is 64-14+1 = 51
        let max_rho = 64 - 14 + 1;

        // The register value should be at most max_rho
        let idx = (hash_with_zero_suffix >> (64 - 14)) as usize;
        assert!(
            hll.registers[idx] <= max_rho as u8,
            "rho {} exceeds max valid rho {}",
            hll.registers[idx],
            max_rho
        );

        // Estimate should still be reasonable (not skewed by invalid rho)
        let estimate = hll.estimate();
        assert!(estimate >= 0.5 && estimate <= 5.0, "estimate={}", estimate);
    }

    #[test]
    fn test_rho_various_precisions() {
        // Test rho clamping works for different precisions
        for precision in [4u8, 10, 14, 18] {
            let mut hll = HyperLogLog::new(precision);

            // Insert hash with all zeros in remaining bits
            let hash = ((1u64 << precision) - 1) << (64 - precision); // Only index bits set
            hll.insert_hash(hash);

            let max_rho = (64 - precision + 1) as u8;
            let idx = (hash >> (64 - precision)) as usize;

            assert!(
                hll.registers[idx] <= max_rho,
                "precision={}: rho {} exceeds max {}",
                precision,
                hll.registers[idx],
                max_rho
            );
        }
    }
}
