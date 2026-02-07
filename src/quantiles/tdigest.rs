//! t-digest quantile estimator
//!
//! Implementation of the t-digest algorithm for streaming quantile estimation.
//! t-digest provides excellent accuracy at the extremes (p01, p99) while
//! being fully mergeable for distributed computation.
//!
//! # Performance Note
//!
//! This implementation uses interior mutability (`RefCell`) to lazily compress
//! the internal buffer on query. This means `quantile()` and `rank()` calls
//! on `&self` will compress any buffered data in-place rather than cloning
//! the entire digest, providing significant performance improvements for
//! single-threaded read-heavy workloads.
//!
//! # Thread Safety
//!
//! `TDigest` is `Send` but **not `Sync`** due to the internal `RefCell`.
//! For concurrent read access, wrap in `Arc<Mutex<_>>` or `Arc<RwLock<_>>`.

use crate::math;
use crate::traits::{MergeError, QuantileSketch, Sketch};
use core::cell::RefCell;

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

// Helper macro for format! in both std and no_std
macro_rules! fmt {
    ($($arg:tt)*) => {{
        #[cfg(feature = "std")]
        { format!($($arg)*) }
        #[cfg(not(feature = "std"))]
        { alloc::format!($($arg)*) }
    }};
}

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

    /// Add a value to the centroid, updating the weighted mean
    pub fn add(&mut self, value: f64, count: u64) {
        let new_weight = self.weight + count;
        self.mean = (self.mean * self.weight as f64 + value * count as f64) / new_weight as f64;
        self.weight = new_weight;
    }
}

/// Mutable interior data for the t-digest.
///
/// Separated from the outer struct so we can wrap it in `RefCell`
/// and lazily compress on read without requiring `&mut self`.
#[derive(Clone, Debug)]
struct TDigestInner {
    /// Centroids (sorted by mean when compressed)
    centroids: Vec<Centroid>,
    /// Buffer for unmerged values
    buffer: Vec<f64>,
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
/// # Thread Safety
///
/// `TDigest` is `Send` but **not `Sync`** due to internal use of `RefCell`
/// for lazy buffer compression. For concurrent read access, wrap in
/// `Arc<Mutex<_>>` or `Arc<RwLock<_>>`.
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
#[derive(Debug)]
pub struct TDigest {
    /// Compression parameter (higher = more accuracy, more memory)
    compression: f64,
    /// Interior mutable state: centroids + buffer
    inner: RefCell<TDigestInner>,
    /// Buffer capacity before forcing compression
    buffer_capacity: usize,
    /// Total count of values
    count: u64,
    /// Minimum value seen
    min: f64,
    /// Maximum value seen
    max: f64,
}

impl Clone for TDigest {
    fn clone(&self) -> Self {
        Self {
            compression: self.compression,
            inner: RefCell::new(self.inner.borrow().clone()),
            buffer_capacity: self.buffer_capacity,
            count: self.count,
            min: self.min,
            max: self.max,
        }
    }
}

impl TDigest {
    /// Create a new t-digest with the given compression parameter
    ///
    /// # Arguments
    ///
    /// * `compression` - Controls accuracy vs memory tradeoff. Typical values: 100-500.
    ///   Must be finite and positive.
    ///
    /// # Panics
    ///
    /// Panics if `compression` is not finite or not positive.
    pub fn new(compression: f64) -> Self {
        assert!(
            compression.is_finite() && compression > 0.0,
            "compression must be finite and positive, got {}",
            compression
        );

        let buffer_capacity = (compression * 2.0) as usize;
        Self {
            compression,
            inner: RefCell::new(TDigestInner {
                centroids: Vec::with_capacity(compression as usize),
                buffer: Vec::with_capacity(buffer_capacity),
            }),
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

    /// Get the current number of compressed centroids
    ///
    /// Note: this does not include values still in the uncompressed buffer.
    /// Call `compress()` first if you need the fully compressed count.
    pub fn num_centroids(&self) -> usize {
        self.inner.borrow().centroids.len()
    }

    /// Add a single value (internal)
    ///
    /// NaN values are ignored to prevent corrupting the digest.
    fn push(&mut self, value: f64) {
        if value.is_nan() {
            return;
        }

        // get_mut() bypasses RefCell runtime checks — zero overhead since we have &mut self
        let inner = self.inner.get_mut();
        inner.buffer.push(value);
        self.count += 1;
        self.min = self.min.min(value);
        self.max = self.max.max(value);

        if inner.buffer.len() >= self.buffer_capacity {
            Self::compress_inner(inner, self.compression);
        }
    }

    /// Force compression of the buffer into centroids
    pub fn compress(&mut self) {
        let inner = self.inner.get_mut();
        Self::compress_inner(inner, self.compression);
    }

    /// Internal: compress buffer into centroids.
    /// Static method to avoid RefCell borrow conflicts.
    fn compress_inner(inner: &mut TDigestInner, compression: f64) {
        if inner.buffer.is_empty() {
            return;
        }

        // Sort buffer
        inner.buffer.sort_by(|a, b| a.total_cmp(b));

        // Create centroids from buffer values
        let mut new_centroids: Vec<Centroid> = inner
            .buffer
            .drain(..)
            .map(|v| Centroid::new(v, 1))
            .collect();

        // Merge with existing centroids
        if !inner.centroids.is_empty() {
            new_centroids.extend(inner.centroids.drain(..));
            new_centroids.sort_by(|a, b| a.mean.total_cmp(&b.mean));
        }

        inner.centroids = Self::compress_centroids_static(new_centroids, compression);
    }

    /// Ensure the buffer is compressed (for &self query methods).
    /// Uses RefCell::borrow_mut() for interior mutability.
    fn ensure_compressed(&self) {
        if self.inner.borrow().buffer.is_empty() {
            return;
        }
        let mut inner = self.inner.borrow_mut();
        if !inner.buffer.is_empty() {
            Self::compress_inner(&mut inner, self.compression);
        }
    }

    /// Compress centroids using the t-digest scale function
    fn compress_centroids_static(
        sorted_centroids: Vec<Centroid>,
        compression: f64,
    ) -> Vec<Centroid> {
        if sorted_centroids.is_empty() {
            return Vec::new();
        }

        let total_weight: u64 = sorted_centroids.iter().map(|c| c.weight).sum();
        let mut result = Vec::with_capacity((compression * 2.0) as usize);

        let mut current = sorted_centroids[0].clone();
        let mut weight_so_far = 0u64;

        for centroid in sorted_centroids.into_iter().skip(1) {
            let proposed_weight = current.weight + centroid.weight;
            let q0 = weight_so_far as f64 / total_weight as f64;
            let q1 = (weight_so_far + proposed_weight) as f64 / total_weight as f64;

            let k0 = Self::scale_static(q0, compression);
            let k1 = Self::scale_static(q1, compression);

            if k1 - k0 <= 1.0 {
                current.add(centroid.mean, centroid.weight);
            } else {
                weight_so_far += current.weight;
                result.push(current);
                current = centroid;
            }
        }

        result.push(current);
        result
    }

    /// Scale function for t-digest (arcsin family).
    /// Clamps argument to [-1, 1] to prevent NaN from floating-point drift.
    #[inline]
    fn scale_static(q: f64, compression: f64) -> f64 {
        let x = (2.0 * q - 1.0).clamp(-1.0, 1.0);
        compression * (math::asin(x) / core::f64::consts::PI + 0.5)
    }

    /// Quantile query using consistent midpoint interpolation.
    ///
    /// Each centroid represents a point mass at its mean. The cumulative
    /// distribution is modeled as a piecewise linear function passing through
    /// each centroid's midpoint (cumulative_weight_before + weight/2, mean).
    /// The curve is anchored at (0, min) and (total_count, max).
    fn quantile_impl(
        centroids: &[Centroid],
        count: u64,
        min: f64,
        max: f64,
        q: f64,
    ) -> Option<f64> {
        if count == 0 {
            return None;
        }

        let q = q.clamp(0.0, 1.0);

        if centroids.is_empty() {
            // No centroids: linearly interpolate between min and max
            return Some(min + (max - min) * q);
        }

        // Handle exact extremes
        if q <= 0.0 {
            return Some(min);
        }
        if q >= 1.0 {
            return Some(max);
        }

        let target_rank = q * count as f64;

        // Piecewise linear model through midpoint anchors:
        //   anchor_start = (rank=0, value=min)
        //   anchor_i     = (rank=cumulative_before_i + weight_i/2, value=mean_i)
        //   anchor_end   = (rank=count, value=max)
        //
        // We walk adjacent anchor pairs and interpolate linearly.

        let mut prev_rank = 0.0_f64;
        let mut prev_mean = min;
        let mut cumulative = 0.0_f64;

        for centroid in centroids.iter() {
            let mid_rank = cumulative + centroid.weight as f64 / 2.0;

            if target_rank < mid_rank {
                // Interpolate between previous anchor and this centroid's midpoint
                let denom = mid_rank - prev_rank;
                if denom <= 0.0 {
                    return Some(prev_mean);
                }
                let t = (target_rank - prev_rank) / denom;
                return Some(prev_mean + t * (centroid.mean - prev_mean));
            }

            cumulative += centroid.weight as f64;
            prev_rank = mid_rank;
            prev_mean = centroid.mean;
        }

        // target_rank is past the last centroid midpoint — interpolate to max
        let denom = count as f64 - prev_rank;
        if denom <= 0.0 {
            return Some(max);
        }
        let t = (target_rank - prev_rank) / denom;
        Some(prev_mean + t * (max - prev_mean))
    }

    /// Rank query using consistent midpoint interpolation (inverse of quantile_impl).
    ///
    /// Uses the same piecewise linear model anchored at centroid midpoints.
    fn rank_impl(centroids: &[Centroid], count: u64, min: f64, max: f64, value: &f64) -> f64 {
        if count == 0 {
            return 0.0;
        }

        // Check >= max before <= min so that when min == max (all values equal),
        // we return 1.0 (CDF convention: P(X ≤ x) = 1.0 when x ≥ max).
        if *value >= max {
            return 1.0;
        }
        if *value <= min {
            return 0.0;
        }

        if centroids.is_empty() {
            // Linear interpolation between min and max
            let denom = max - min;
            if denom <= 0.0 {
                return 0.5;
            }
            return (*value - min) / denom;
        }

        // Walk the same piecewise linear model as quantile_impl
        let mut prev_rank = 0.0_f64;
        let mut prev_mean = min;
        let mut cumulative = 0.0_f64;

        for centroid in centroids.iter() {
            let mid_rank = cumulative + centroid.weight as f64 / 2.0;

            if *value < centroid.mean {
                // value falls in the segment [prev_mean, centroid.mean]
                let denom = centroid.mean - prev_mean;
                if denom <= 0.0 {
                    return prev_rank / count as f64;
                }
                let t = (*value - prev_mean) / denom;
                let interpolated_rank = prev_rank + t * (mid_rank - prev_rank);
                return interpolated_rank / count as f64;
            }

            cumulative += centroid.weight as f64;
            prev_rank = mid_rank;
            prev_mean = centroid.mean;
        }

        // value is past the last centroid mean — interpolate to max
        let denom = max - prev_mean;
        if denom <= 0.0 {
            return 1.0;
        }
        let t = (*value - prev_mean) / denom;
        let interpolated_rank = prev_rank + t * (count as f64 - prev_rank);
        interpolated_rank / count as f64
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
        // Check compatibility with relative tolerance.
        // Compression is typically an integer-ish value (100, 200, etc.),
        // so 1e-6 relative tolerance is generous enough for serialization
        // roundtrips but still catches genuinely different configs.
        let avg = (self.compression + other.compression) * 0.5;
        let diff = self.compression - other.compression;
        let abs_diff = if diff < 0.0 { -diff } else { diff };
        if avg > 0.0 && abs_diff / avg > 1e-6 {
            return Err(MergeError::IncompatibleConfig {
                expected: fmt!("compression={}", self.compression),
                found: fmt!("compression={}", other.compression),
            });
        }

        let self_inner = self.inner.get_mut();
        let other_inner = other.inner.borrow();

        let mut all_centroids = core::mem::take(&mut self_inner.centroids);
        all_centroids.extend(other_inner.centroids.iter().cloned());

        for &v in &self_inner.buffer {
            all_centroids.push(Centroid::new(v, 1));
        }
        for &v in &other_inner.buffer {
            all_centroids.push(Centroid::new(v, 1));
        }

        if !all_centroids.is_empty() {
            all_centroids.sort_by(|a, b| a.mean.total_cmp(&b.mean));
        }

        self.count += other.count;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self_inner.buffer.clear();

        self_inner.centroids = Self::compress_centroids_static(all_centroids, self.compression);

        Ok(())
    }

    fn clear(&mut self) {
        let inner = self.inner.get_mut();
        inner.centroids.clear();
        inner.buffer.clear();
        self.count = 0;
        self.min = f64::INFINITY;
        self.max = f64::NEG_INFINITY;
    }

    fn size_bytes(&self) -> usize {
        let inner = self.inner.borrow();
        core::mem::size_of::<Self>()
            + inner.centroids.capacity() * core::mem::size_of::<Centroid>()
            + inner.buffer.capacity() * core::mem::size_of::<f64>()
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
        self.ensure_compressed();
        let inner = self.inner.borrow();
        Self::quantile_impl(&inner.centroids, self.count, self.min, self.max, rank)
    }

    fn rank(&self, value: &f64) -> f64 {
        self.ensure_compressed();
        let inner = self.inner.borrow();
        Self::rank_impl(&inner.centroids, self.count, self.min, self.max, value)
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

#[cfg(feature = "serde")]
impl serde::Serialize for TDigest {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let inner = self.inner.borrow();
        let mut state = serializer.serialize_struct("TDigest", 6)?;
        state.serialize_field("compression", &self.compression)?;
        state.serialize_field("centroids", &inner.centroids)?;
        state.serialize_field("buffer", &inner.buffer)?;
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

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Basic functionality ----

    #[test]
    fn test_basic() {
        let mut digest = TDigest::new(100.0);
        for i in 1..=100 {
            digest.add(i as f64);
        }
        let median = digest.median().unwrap();
        assert!(median > 45.0 && median < 55.0);
    }

    #[test]
    fn test_empty() {
        let digest = TDigest::new(100.0);
        assert!(digest.median().is_none());
        assert!(digest.min().is_none());
        assert!(digest.max().is_none());
        assert_eq!(digest.count(), 0);
    }

    #[test]
    fn test_single_value() {
        let mut digest = TDigest::new(100.0);
        digest.add(42.0);
        assert_eq!(digest.min(), Some(42.0));
        assert_eq!(digest.max(), Some(42.0));
        // quantile of a single value should always be that value
        assert_eq!(digest.quantile(0.0), Some(42.0));
        assert_eq!(digest.quantile(0.5), Some(42.0));
        assert_eq!(digest.quantile(1.0), Some(42.0));
    }

    #[test]
    fn test_two_values() {
        let mut digest = TDigest::new(100.0);
        digest.add(10.0);
        digest.add(20.0);
        assert_eq!(digest.quantile(0.0), Some(10.0));
        assert_eq!(digest.quantile(1.0), Some(20.0));
        let median = digest.quantile(0.5).unwrap();
        assert!(median >= 10.0 && median <= 20.0, "median={}", median);
    }

    #[test]
    fn test_quantiles() {
        let mut digest = TDigest::new(100.0);
        for i in 1..=1000 {
            digest.add(i as f64);
        }
        let p10 = digest.quantile(0.1).unwrap();
        let p50 = digest.quantile(0.5).unwrap();
        let p90 = digest.quantile(0.9).unwrap();
        let p99 = digest.quantile(0.99).unwrap();
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
        assert_eq!(digest.quantile(0.0).unwrap(), 1.0);
        assert_eq!(digest.quantile(1.0).unwrap(), 1000.0);
    }

    // ---- Merge ----

    #[test]
    fn test_merge() {
        let mut d1 = TDigest::new(100.0);
        let mut d2 = TDigest::new(100.0);
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
    fn test_merge_incompatible_compression() {
        let mut d1 = TDigest::new(100.0);
        let d2 = TDigest::new(200.0);
        d1.add(1.0);
        assert!(d1.merge(&d2).is_err());
    }

    #[test]
    fn test_merge_tolerates_serialization_roundtrip() {
        // Two digests with compression that differs by less than relative 1e-6
        // should merge successfully (review point #7)
        let mut d1 = TDigest::new(100.0);
        let d2 = TDigest::new(100.0 + 1e-12);
        d1.add(1.0);
        assert!(d1.merge(&d2).is_ok(), "should tolerate tiny float drift");
    }

    // ---- Clear ----

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

    // ---- Rank ----

    #[test]
    fn test_rank() {
        let mut digest = TDigest::new(100.0);
        for i in 1..=100 {
            digest.add(i as f64);
        }
        let rank = digest.rank(&50.0);
        assert!(rank > 0.4 && rank < 0.6, "rank={}", rank);
        let rank_min = digest.rank(&1.0);
        assert!(rank_min < 0.05, "rank_min={}", rank_min);
        let rank_max = digest.rank(&100.0);
        assert!(rank_max > 0.95, "rank_max={}", rank_max);
    }

    // ---- Edge cases: NaN, Infinity, parameter validation ----

    #[test]
    fn test_nan_ignored() {
        let mut digest = TDigest::new(100.0);
        digest.add(1.0);
        digest.add(f64::NAN);
        digest.add(2.0);
        digest.add(f64::NAN);
        digest.add(3.0);
        assert_eq!(digest.count(), 3);
        assert_eq!(digest.min(), Some(1.0));
        assert_eq!(digest.max(), Some(3.0));
        let median = digest.median().unwrap();
        assert!(!median.is_nan());
    }

    #[test]
    fn test_infinity() {
        let mut digest = TDigest::new(100.0);
        digest.add(1.0);
        digest.add(f64::INFINITY);
        digest.add(2.0);
        assert_eq!(digest.count(), 3);
        assert_eq!(digest.max(), Some(f64::INFINITY));
    }

    #[test]
    #[should_panic(expected = "compression must be finite and positive")]
    fn test_negative_compression_panics() {
        TDigest::new(-1.0);
    }

    #[test]
    #[should_panic(expected = "compression must be finite and positive")]
    fn test_zero_compression_panics() {
        TDigest::new(0.0);
    }

    #[test]
    #[should_panic(expected = "compression must be finite and positive")]
    fn test_nan_compression_panics() {
        TDigest::new(f64::NAN);
    }

    #[test]
    #[should_panic(expected = "compression must be finite and positive")]
    fn test_inf_compression_panics() {
        TDigest::new(f64::INFINITY);
    }

    // ---- Default / compression ----

    #[test]
    fn test_default() {
        let digest = TDigest::default();
        let diff = digest.compression() - 100.0;
        assert!(diff < f64::EPSILON && diff > -f64::EPSILON);
    }

    #[test]
    fn test_compression_parameter() {
        let d1 = TDigest::new(50.0);
        let d2 = TDigest::new(500.0);
        assert!(d2.buffer_capacity > d1.buffer_capacity);
    }

    // ---- RefCell optimization tests ----

    #[test]
    fn test_query_does_not_require_mut() {
        let mut digest = TDigest::new(100.0);
        for i in 1..=100 {
            digest.add(i as f64);
        }
        let digest_ref: &TDigest = &digest;
        assert!(digest_ref.quantile(0.5).is_some());
        assert!(digest_ref.rank(&50.0) > 0.0);
    }

    #[test]
    fn test_lazy_compress_on_query() {
        let mut digest = TDigest::new(100.0);
        for i in 1..=10 {
            digest.add(i as f64);
        }
        assert!(!digest.inner.borrow().buffer.is_empty());
        let median = digest.quantile(0.5).unwrap();
        assert!(median > 3.0 && median < 8.0, "median={}", median);
        assert!(digest.inner.borrow().buffer.is_empty());
    }

    // ---- Deterministic small-dataset tests (review point #11) ----

    #[test]
    fn test_deterministic_small_dataset() {
        // Known exact quantiles for [1, 2, 3, 4, 5]
        let mut digest = TDigest::new(100.0);
        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            digest.add(v);
        }

        assert_eq!(digest.quantile(0.0).unwrap(), 1.0);
        assert_eq!(digest.quantile(1.0).unwrap(), 5.0);

        // Median of [1,2,3,4,5] is 3.0
        let p50 = digest.quantile(0.5).unwrap();
        assert!((p50 - 3.0).abs() < 0.5, "p50={}", p50);
    }

    #[test]
    fn test_deterministic_known_percentiles() {
        // 10 values: 10, 20, 30, ..., 100
        let mut digest = TDigest::new(200.0);
        for i in 1..=10 {
            digest.add(i as f64 * 10.0);
        }

        assert_eq!(digest.quantile(0.0).unwrap(), 10.0);
        assert_eq!(digest.quantile(1.0).unwrap(), 100.0);

        // Median should be close to 55
        let p50 = digest.quantile(0.5).unwrap();
        assert!(p50 > 45.0 && p50 < 65.0, "p50={}", p50);
    }

    // ---- Monotonicity tests (review point #11) ----

    #[test]
    fn test_quantile_monotonicity() {
        let mut digest = TDigest::new(100.0);
        for i in 1..=10000 {
            digest.add(i as f64);
        }

        let mut prev = f64::NEG_INFINITY;
        for i in 0..=100 {
            let q = i as f64 / 100.0;
            let val = digest.quantile(q).unwrap();
            assert!(
                val >= prev,
                "monotonicity violation: q({})={} < q({})={}",
                q,
                val,
                (i - 1) as f64 / 100.0,
                prev
            );
            assert!(!val.is_nan(), "NaN at q={}", q);
            prev = val;
        }
    }

    #[test]
    fn test_rank_monotonicity() {
        let mut digest = TDigest::new(100.0);
        for i in 1..=10000 {
            digest.add(i as f64);
        }

        let mut prev_rank = -1.0_f64;
        for i in 0..=100 {
            let val = i as f64 * 100.0;
            let r = digest.rank(&val);
            assert!(
                r >= prev_rank,
                "rank monotonicity violation: rank({})={} < rank({})={}",
                val,
                r,
                (i - 1) as f64 * 100.0,
                prev_rank
            );
            assert!(!r.is_nan(), "NaN rank at value={}", val);
            prev_rank = r;
        }
    }

    // ---- Rank/quantile consistency (review point #11) ----

    #[test]
    fn test_rank_quantile_consistency() {
        // rank(quantile(q)) ≈ q
        let mut digest = TDigest::new(200.0);
        for i in 1..=10000 {
            digest.add(i as f64);
        }

        for i in 1..=99 {
            let q = i as f64 / 100.0;
            let val = digest.quantile(q).unwrap();
            let r = digest.rank(&val);
            let error = if r > q { r - q } else { q - r };
            assert!(
                error < 0.05,
                "rank/quantile inconsistency: q={}, quantile(q)={}, rank(quantile(q))={}, error={}",
                q,
                val,
                r,
                error
            );
        }
    }

    // ---- Degenerate distributions (review point #11) ----

    #[test]
    fn test_all_values_equal() {
        let mut digest = TDigest::new(100.0);
        for _ in 0..1000 {
            digest.add(42.0);
        }

        assert_eq!(digest.min(), Some(42.0));
        assert_eq!(digest.max(), Some(42.0));

        // All quantiles should return 42.0
        for i in 0..=10 {
            let q = i as f64 / 10.0;
            let val = digest.quantile(q).unwrap();
            assert_eq!(val, 42.0, "q={} returned {}", q, val);
        }

        // Rank of 42.0 should be 1.0 (value >= max)
        assert_eq!(digest.rank(&42.0), 1.0);
        // Rank of anything below should be 0.0
        assert_eq!(digest.rank(&41.0), 0.0);
        // No NaN anywhere
        assert!(!digest.quantile(0.5).unwrap().is_nan());
    }

    #[test]
    fn test_many_duplicates() {
        let mut digest = TDigest::new(100.0);
        // 10000 zeros and 10000 ones
        for _ in 0..10000 {
            digest.add(0.0);
        }
        for _ in 0..10000 {
            digest.add(1.0);
        }

        assert_eq!(digest.min(), Some(0.0));
        assert_eq!(digest.max(), Some(1.0));

        let p25 = digest.quantile(0.25).unwrap();
        let p50 = digest.quantile(0.50).unwrap();
        let p75 = digest.quantile(0.75).unwrap();

        // p25 should be close to 0, p75 close to 1
        assert!(p25 >= 0.0 && p25 <= 0.5, "p25={}", p25);
        assert!(p50 >= 0.0 && p50 <= 1.0, "p50={}", p50);
        assert!(p75 >= 0.5 && p75 <= 1.0, "p75={}", p75);

        // No NaN
        assert!(!p25.is_nan());
        assert!(!p50.is_nan());
        assert!(!p75.is_nan());
    }

    #[test]
    fn test_two_point_distribution() {
        let mut digest = TDigest::new(100.0);
        for _ in 0..5000 {
            digest.add(0.0);
        }
        for _ in 0..5000 {
            digest.add(100.0);
        }

        assert_eq!(digest.rank(&0.0), 0.0);
        assert_eq!(digest.rank(&100.0), 1.0);
        let r50 = digest.rank(&50.0);
        assert!(r50 > 0.3 && r50 < 0.7, "rank(50)={}", r50);

        // Quantile monotonicity holds
        let mut prev = f64::NEG_INFINITY;
        for i in 0..=20 {
            let q = i as f64 / 20.0;
            let val = digest.quantile(q).unwrap();
            assert!(val >= prev, "monotonicity at q={}: {} < {}", q, val, prev);
            assert!(!val.is_nan());
            prev = val;
        }
    }

    #[test]
    fn test_division_by_zero_no_nan() {
        // Construct scenarios where denominators could be zero
        let mut digest = TDigest::new(100.0);
        for _ in 0..100 {
            digest.add(5.0);
        }

        for i in 0..=20 {
            let q = i as f64 / 20.0;
            let val = digest.quantile(q).unwrap();
            assert!(!val.is_nan(), "NaN at q={}", q);
        }

        for v in [4.0, 5.0, 6.0, 0.0, 100.0] {
            let r = digest.rank(&v);
            assert!(!r.is_nan(), "NaN rank at value={}", v);
        }
    }

    #[test]
    fn test_tight_cluster_no_nan() {
        // Values extremely close together
        let mut digest = TDigest::new(100.0);
        let base = 1e15;
        for i in 0..1000 {
            digest.add(base + i as f64 * 1e-10);
        }

        for i in 0..=10 {
            let q = i as f64 / 10.0;
            let val = digest.quantile(q).unwrap();
            assert!(!val.is_nan(), "NaN at q={}", q);
        }

        let r = digest.rank(&base);
        assert!(!r.is_nan());
    }
}
