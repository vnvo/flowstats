//! t-digest quantile estimator
//!
//! Implementation of the t-digest algorithm for streaming quantile estimation.
//! t-digest provides excellent accuracy at the extremes (p01, p99) while
//! being fully mergeable for distributed computation.

use crate::math;
use crate::traits::{MergeError, QuantileSketch, Sketch};

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// A centroid in the t-digest
///
/// Centroids represent clusters of values with a mean and count.
#[derive(Clone, Debug, PartialEq)]
pub struct Centroid {
    /// Mean value of the centroid
    pub mean: f64,
    /// Number of values in the centroid
    pub weight: u64,
}

impl Centroid {
    /// Create a new centroid
    pub fn new(mean: f64, weight: u64) -> Self {
        Self { mean, weight }
    }

    /// Add a value to the centroid
    pub fn add(&mut self, value: f64, count: u64) {
        let new_weight = self.weight + count;
        self.mean = (self.mean * self.weight as f64 + value * count as f64) / new_weight as f64;
        self.weight = new_weight;
    }
}

/// t-digest quantile sketch
///
/// The t-digest is a data structure for estimating quantiles of a distribution
/// from a stream of values. It provides:
///
/// - O(1) amortized time per insertion
/// - Accurate estimation especially at the tails (p01, p99)
/// - Full mergeability for distributed computation
/// - Bounded memory usage
///
/// # Compression Parameter
///
/// The compression parameter controls the tradeoff between accuracy and memory:
/// - Higher compression = more centroids = better accuracy = more memory
/// - Typical values: 100-500
/// - Default: 100
///
/// # Example
///
/// ```
/// use flowstats::quantiles::TDigest;
/// use flowstats::traits::QuantileSketch;
///
/// let mut digest = TDigest::new(100.0);
///
/// // Add some values
/// for i in 1..=1000 {
///     digest.add(i as f64);
/// }
///
/// // Query quantiles
/// let p50 = digest.quantile(0.5);  // median
/// let p99 = digest.quantile(0.99); // 99th percentile
/// ```
#[derive(Clone, Debug)]
pub struct TDigest {
    /// Compression parameter (higher = more accuracy, more memory)
    compression: f64,
    /// Centroids (sorted by mean)
    centroids: Vec<Centroid>,
    /// Buffer for unmerged values
    buffer: Vec<f64>,
    /// Buffer capacity before forcing compression
    buffer_capacity: usize,
    /// Total count of values
    count: u64,
    /// Minimum value seen
    min: f64,
    /// Maximum value seen
    max: f64,
}

impl TDigest {
    /// Create a new t-digest with the given compression parameter
    ///
    /// # Arguments
    ///
    /// * `compression` - Controls accuracy vs memory tradeoff. Typical values: 100-500.
    ///   Higher values give better accuracy but use more memory.
    pub fn new(compression: f64) -> Self {
        let buffer_capacity = (compression * 2.0) as usize;
        Self {
            compression,
            centroids: Vec::with_capacity(compression as usize),
            buffer: Vec::with_capacity(buffer_capacity),
            buffer_capacity,
            count: 0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Get the compression parameter
    pub fn compression(&self) -> f64 {
        self.compression
    }

    /// Get the current number of centroids
    pub fn num_centroids(&self) -> usize {
        self.centroids.len()
    }

    /// Add a single value
    ///
    /// NaN values are ignored to prevent corrupting the digest.
    pub fn push(&mut self, value: f64) {
        // Reject NaN to prevent poisoning the digest
        if value.is_nan() {
            return;
        }

        self.buffer.push(value);
        self.count += 1;
        self.min = self.min.min(value);
        self.max = self.max.max(value);

        if self.buffer.len() >= self.buffer_capacity {
            self.compress();
        }
    }

    /// Force compression of the buffer
    pub fn compress(&mut self) {
        if self.buffer.is_empty() {
            return;
        }

        // Sort buffer using total_cmp for defined ordering (handles edge cases)
        self.buffer.sort_by(|a, b| a.total_cmp(b));

        // Create centroids from buffer values
        let mut new_centroids: Vec<Centroid> =
            self.buffer.drain(..).map(|v| Centroid::new(v, 1)).collect();

        // Merge with existing centroids
        if !self.centroids.is_empty() {
            new_centroids.extend(self.centroids.drain(..));
            new_centroids.sort_by(|a, b| a.mean.total_cmp(&b.mean));
        }

        // Compress centroids using the scale function
        self.centroids = self.compress_centroids(new_centroids);
    }

    /// Compress centroids using the t-digest scale function
    fn compress_centroids(&self, sorted_centroids: Vec<Centroid>) -> Vec<Centroid> {
        if sorted_centroids.is_empty() {
            return Vec::new();
        }

        let total_weight: u64 = sorted_centroids.iter().map(|c| c.weight).sum();
        let mut result = Vec::with_capacity((self.compression * 2.0) as usize);

        let mut current = sorted_centroids[0].clone();
        let mut weight_so_far = 0u64;

        for centroid in sorted_centroids.into_iter().skip(1) {
            let proposed_weight = current.weight + centroid.weight;
            let q0 = weight_so_far as f64 / total_weight as f64;
            let q1 = (weight_so_far + proposed_weight) as f64 / total_weight as f64;

            // Scale function: k = compression * (asin(2*q - 1) / pi + 0.5)
            let k0 = self.scale(q0);
            let k1 = self.scale(q1);

            if k1 - k0 <= 1.0 {
                // Merge centroids
                current.add(centroid.mean, centroid.weight);
            } else {
                // Start new centroid
                weight_so_far += current.weight;
                result.push(current);
                current = centroid;
            }
        }

        result.push(current);
        result
    }

    /// Scale function for t-digest (arcsin approximation)
    #[inline]
    fn scale(&self, q: f64) -> f64 {
        self.compression * (asin_approx(2.0 * q - 1.0) / core::f64::consts::PI + 0.5)
    }

    /// Get quantile at rank (0.0 to 1.0)
    fn quantile_impl(&self, q: f64) -> Option<f64> {
        if self.count == 0 {
            return None;
        }

        // Ensure data is compressed
        if !self.buffer.is_empty() {
            // For immutable query, we need to work around this
            // In practice, users should call compress() or we clone
        }

        let centroids = &self.centroids;
        if centroids.is_empty() {
            // All data is in buffer
            if self.buffer.is_empty() {
                return None;
            }
            // This shouldn't happen if compress is called properly
            return Some(self.min);
        }

        let q = q.clamp(0.0, 1.0);
        let target_rank = q * self.count as f64;

        // Handle extremes
        if q <= 0.0 {
            return Some(self.min);
        }
        if q >= 1.0 {
            return Some(self.max);
        }

        // Binary search through centroids
        let mut cumulative = 0.0;

        for (i, centroid) in centroids.iter().enumerate() {
            let next_cumulative = cumulative + centroid.weight as f64;

            if target_rank <= next_cumulative {
                // Interpolate within this centroid
                if i == 0 {
                    // First centroid: interpolate from min
                    let prev_mean = self.min;
                    let delta = centroid.mean - prev_mean;
                    let fraction = (target_rank - cumulative) / centroid.weight as f64;
                    return Some(prev_mean + delta * fraction);
                } else {
                    // Interpolate between this and previous centroid
                    let prev = &centroids[i - 1];
                    let prev_mid = cumulative - prev.weight as f64 / 2.0;
                    let curr_mid = cumulative + centroid.weight as f64 / 2.0;

                    if target_rank <= cumulative {
                        // In the overlap with previous centroid
                        let t = (target_rank - prev_mid) / (cumulative - prev_mid);
                        return Some(prev.mean + (centroid.mean - prev.mean) * t);
                    } else {
                        // In this centroid
                        let t = (target_rank - cumulative) / (curr_mid - cumulative);
                        return Some(
                            centroid.mean
                                + if i + 1 < centroids.len() {
                                    (centroids[i + 1].mean - centroid.mean) * t
                                } else {
                                    (self.max - centroid.mean) * t
                                },
                        );
                    }
                }
            }

            cumulative = next_cumulative;
        }

        Some(self.max)
    }

    /// Get the rank of a value (0.0 to 1.0)
    fn rank_impl(&self, value: &f64) -> f64 {
        if self.count == 0 {
            return 0.0;
        }

        if *value <= self.min {
            return 0.0;
        }
        if *value >= self.max {
            return 1.0;
        }

        let centroids = &self.centroids;
        if centroids.is_empty() {
            return 0.5; // No centroids yet
        }

        let mut cumulative = 0.0;

        for (i, centroid) in centroids.iter().enumerate() {
            if *value < centroid.mean {
                // Value falls before this centroid
                if i == 0 {
                    // Before first centroid: interpolate from min
                    let fraction = (*value - self.min) / (centroid.mean - self.min);
                    return fraction * (centroid.weight as f64 / 2.0) / self.count as f64;
                } else {
                    // Between previous and this centroid
                    let prev = &centroids[i - 1];
                    let prev_weight_half = prev.weight as f64 / 2.0;
                    let curr_weight_half = centroid.weight as f64 / 2.0;

                    let fraction = (*value - prev.mean) / (centroid.mean - prev.mean);
                    return (cumulative - prev_weight_half
                        + fraction * (prev_weight_half + curr_weight_half))
                        / self.count as f64;
                }
            }

            cumulative += centroid.weight as f64;
        }

        // Value is at or beyond the last centroid
        1.0
    }
}

impl Default for TDigest {
    fn default() -> Self {
        Self::new(100.0)
    }
}

impl Sketch for TDigest {
    type Item = f64;

    fn update(&mut self, item: &f64) {
        self.push(*item);
    }

    fn merge(&mut self, other: &Self) -> Result<(), MergeError> {
        // Check compatibility
        if (self.compression - other.compression).abs() > f64::EPSILON {
            return Err(MergeError::IncompatibleConfig {
                expected: format!("compression={}", self.compression),
                found: format!("compression={}", other.compression),
            });
        }

        // Merge centroids
        let mut all_centroids = self.centroids.clone();
        all_centroids.extend(other.centroids.iter().cloned());
        all_centroids.sort_by(|a, b| a.mean.total_cmp(&b.mean));

        // Add buffered values as centroids
        for &v in &self.buffer {
            all_centroids.push(Centroid::new(v, 1));
        }
        for &v in &other.buffer {
            all_centroids.push(Centroid::new(v, 1));
        }

        if !all_centroids.is_empty() {
            all_centroids.sort_by(|a, b| a.mean.total_cmp(&b.mean));
        }

        // Update stats
        self.count += other.count;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self.buffer.clear();

        // Compress merged centroids
        self.centroids = self.compress_centroids(all_centroids);

        Ok(())
    }

    fn clear(&mut self) {
        self.centroids.clear();
        self.buffer.clear();
        self.count = 0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
    }

    fn size_bytes(&self) -> usize {
        core::mem::size_of::<Self>()
            + self.centroids.capacity() * core::mem::size_of::<Centroid>()
            + self.buffer.capacity() * core::mem::size_of::<f64>()
    }

    fn count(&self) -> u64 {
        self.count
    }
}

impl QuantileSketch for TDigest {
    type Value = f64;

    fn add(&mut self, value: f64) {
        self.push(value);
    }

    fn quantile(&self, rank: f64) -> Option<f64> {
        // Need to handle buffer - in production we'd use interior mutability
        // or require explicit compress() call
        if !self.buffer.is_empty() {
            // Create temporary compressed version
            let mut temp = self.clone();
            temp.compress();
            return temp.quantile_impl(rank);
        }
        self.quantile_impl(rank)
    }

    fn rank(&self, value: &f64) -> f64 {
        if !self.buffer.is_empty() {
            let mut temp = self.clone();
            temp.compress();
            return temp.rank_impl(value);
        }
        self.rank_impl(value)
    }

    fn min(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.min)
        }
    }

    fn max(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.max)
        }
    }
}

/// Approximate arcsin for the scale function
#[inline]
fn asin_approx(x: f64) -> f64 {
    math::asin(x)
}

#[cfg(feature = "serde")]
impl serde::Serialize for TDigest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("TDigest", 6)?;
        state.serialize_field("compression", &self.compression)?;
        state.serialize_field("centroids", &self.centroids)?;
        state.serialize_field("buffer", &self.buffer)?;
        state.serialize_field("count", &self.count)?;
        state.serialize_field("min", &self.min)?;
        state.serialize_field("max", &self.max)?;
        state.end()
    }
}

#[cfg(feature = "serde")]
impl serde::Serialize for Centroid {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeTuple;
        let mut tuple = serializer.serialize_tuple(2)?;
        tuple.serialize_element(&self.mean)?;
        tuple.serialize_element(&self.weight)?;
        tuple.end()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let mut digest = TDigest::new(100.0);

        for i in 1..=100 {
            digest.add(i as f64);
        }

        let median = digest.median().unwrap();
        // Median of 1-100 is 50.5
        assert!(median > 45.0 && median < 55.0);
    }

    #[test]
    fn test_empty() {
        let digest = TDigest::new(100.0);
        assert!(digest.median().is_none());
        assert!(digest.min().is_none());
        assert!(digest.max().is_none());
    }

    #[test]
    fn test_single_value() {
        let mut digest = TDigest::new(100.0);
        digest.add(42.0);

        assert_eq!(digest.min(), Some(42.0));
        assert_eq!(digest.max(), Some(42.0));
    }

    #[test]
    fn test_quantiles() {
        let mut digest = TDigest::new(100.0);

        // Add values 1 to 1000
        for i in 1..=1000 {
            digest.add(i as f64);
        }

        // Check various quantiles
        let p10 = digest.quantile(0.1).unwrap();
        let p50 = digest.quantile(0.5).unwrap();
        let p90 = digest.quantile(0.9).unwrap();
        let p99 = digest.quantile(0.99).unwrap();

        // Should be approximately correct
        assert!(p10 > 50.0 && p10 < 150.0, "p10={}", p10);
        assert!(p50 > 450.0 && p50 < 550.0, "p50={}", p50);
        assert!(p90 > 850.0 && p90 < 950.0, "p90={}", p90);
        assert!(p99 > 950.0 && p99 <= 1000.0, "p99={}", p99);
    }

    #[test]
    fn test_extremes() {
        let mut digest = TDigest::new(100.0);

        for i in 1..=1000 {
            digest.add(i as f64);
        }

        let p0 = digest.quantile(0.0).unwrap();
        let p1 = digest.quantile(1.0).unwrap();

        assert_eq!(p0, 1.0);
        assert_eq!(p1, 1000.0);
    }

    #[test]
    fn test_merge() {
        let mut d1 = TDigest::new(100.0);
        let mut d2 = TDigest::new(100.0);

        // Different ranges
        for i in 1..=500 {
            d1.add(i as f64);
        }
        for i in 501..=1000 {
            d2.add(i as f64);
        }

        d1.merge(&d2).unwrap();

        assert_eq!(d1.count(), 1000);
        assert_eq!(d1.min(), Some(1.0));
        assert_eq!(d1.max(), Some(1000.0));

        let median = d1.median().unwrap();
        assert!(median > 450.0 && median < 550.0);
    }

    #[test]
    fn test_clear() {
        let mut digest = TDigest::new(100.0);

        for i in 1..=100 {
            digest.add(i as f64);
        }

        digest.clear();

        assert_eq!(digest.count(), 0);
        assert!(digest.median().is_none());
    }

    #[test]
    fn test_rank() {
        let mut digest = TDigest::new(100.0);

        for i in 1..=100 {
            digest.add(i as f64);
        }

        // 50 should be around rank 0.5
        let rank = digest.rank(&50.0);
        assert!(rank > 0.4 && rank < 0.6, "rank={}", rank);

        // 1 should be near 0
        let rank_min = digest.rank(&1.0);
        assert!(rank_min < 0.1, "rank_min={}", rank_min);

        // 100 should be near 1
        let rank_max = digest.rank(&100.0);
        assert!(rank_max > 0.9, "rank_max={}", rank_max);
    }

    #[test]
    fn test_compression_parameter() {
        let d1 = TDigest::new(50.0);
        let d2 = TDigest::new(500.0);

        assert!(d2.buffer_capacity > d1.buffer_capacity);
    }

    #[test]
    fn test_nan_ignored() {
        let mut digest = TDigest::new(100.0);

        digest.add(1.0);
        digest.add(f64::NAN);
        digest.add(2.0);
        digest.add(f64::NAN);
        digest.add(3.0);

        // NaNs should be ignored
        assert_eq!(digest.count(), 3);
        assert_eq!(digest.min(), Some(1.0));
        assert_eq!(digest.max(), Some(3.0));

        // Quantiles should work correctly
        let median = digest.median().unwrap();
        assert!(!median.is_nan());
    }

    #[test]
    fn test_merge_incompatible_compression() {
        let mut d1 = TDigest::new(100.0);
        let d2 = TDigest::new(200.0);

        d1.add(1.0);

        let result = d1.merge(&d2);
        assert!(result.is_err());
    }

    #[test]
    fn test_default() {
        let digest = TDigest::default();
        assert!((digest.compression() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_infinity() {
        let mut digest = TDigest::new(100.0);

        digest.add(1.0);
        digest.add(f64::INFINITY);
        digest.add(2.0);

        // Infinity is a valid f64 value
        assert_eq!(digest.count(), 3);
        assert_eq!(digest.max(), Some(f64::INFINITY));
    }
}
