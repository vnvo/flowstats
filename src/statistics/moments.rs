//! Running statistics (mean, variance, min, max)
//!
//! Computes streaming statistics using Welford's numerically stable online algorithm.
//! Supports merging for distributed computation.

use crate::traits::{MergeError, Sketch};

/// Running statistics calculator using Welford's algorithm
///
/// Computes mean, variance, standard deviation, min, and max in a single pass
/// with O(1) memory. Uses Welford's numerically stable algorithm to avoid
/// catastrophic cancellation.
///
/// # Example
///
/// ```
/// use flowstats::statistics::RunningStats;
///
/// let mut stats = RunningStats::new();
///
/// for value in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
///     stats.add(value);
/// }
///
/// assert!((stats.mean() - 5.0).abs() < 0.001);
/// assert!((stats.variance() - 4.0).abs() < 0.001);
/// assert!((stats.stddev() - 2.0).abs() < 0.001);
/// assert_eq!(stats.min(), Some(2.0));
/// assert_eq!(stats.max(), Some(9.0));
/// ```
///
/// # Distributed Usage
///
/// ```
/// use flowstats::statistics::RunningStats;
/// use flowstats::traits::Sketch;
///
/// let mut stats1 = RunningStats::new();
/// let mut stats2 = RunningStats::new();
///
/// // Worker 1
/// for v in [1.0, 2.0, 3.0] {
///     stats1.add(v);
/// }
///
/// // Worker 2
/// for v in [4.0, 5.0, 6.0] {
///     stats2.add(v);
/// }
///
/// // Merge
/// stats1.merge(&stats2).unwrap();
/// assert!((stats1.mean() - 3.5).abs() < 0.001);
/// ```
#[derive(Clone, Debug)]
pub struct RunningStats {
    /// Number of values seen
    count: u64,
    /// Running mean
    mean: f64,
    /// Sum of squared differences from mean (M2 in Welford's algorithm)
    m2: f64,
    /// Minimum value
    min: f64,
    /// Maximum value
    max: f64,
}

impl Default for RunningStats {
    fn default() -> Self {
        Self::new()
    }
}

impl RunningStats {
    /// Create a new empty statistics accumulator
    pub fn new() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    /// Add a value to the statistics
    ///
    /// Uses Welford's online algorithm for numerical stability.
    /// NaN values are ignored to prevent poisoning the statistics.
    pub fn add(&mut self, value: f64) {
        // Ignore NaN to prevent poisoning statistics
        if value.is_nan() {
            return;
        }

        self.count += 1;

        // Update min/max
        if value < self.min {
            self.min = value;
        }
        if value > self.max {
            self.max = value;
        }

        // Welford's algorithm
        let delta = value - self.mean;
        self.mean += delta / self.count as f64;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Get the number of values
    pub fn len(&self) -> u64 {
        self.count
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Get the mean (average)
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.mean
        }
    }

    /// Get the population variance
    ///
    /// This is the variance assuming the data represents the entire population.
    /// Use `sample_variance()` if the data is a sample.
    pub fn variance(&self) -> f64 {
        if self.count < 1 {
            0.0
        } else {
            self.m2 / self.count as f64
        }
    }

    /// Get the sample variance
    ///
    /// This is the unbiased variance estimator (Bessel's correction).
    /// Use `variance()` for population variance.
    pub fn sample_variance(&self) -> f64 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / (self.count - 1) as f64
        }
    }

    /// Get the population standard deviation
    pub fn stddev(&self) -> f64 {
        self.variance().sqrt()
    }

    /// Get the sample standard deviation
    pub fn sample_stddev(&self) -> f64 {
        self.sample_variance().sqrt()
    }

    /// Get the minimum value
    pub fn min(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.min)
        }
    }

    /// Get the maximum value
    pub fn max(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.max)
        }
    }

    /// Get the range (max - min)
    pub fn range(&self) -> Option<f64> {
        if self.count == 0 {
            None
        } else {
            Some(self.max - self.min)
        }
    }

    /// Get the sum of all values
    pub fn sum(&self) -> f64 {
        self.mean * self.count as f64
    }

    /// Merge with another RunningStats using parallel algorithm
    ///
    /// Uses Chan et al.'s parallel algorithm for combining statistics.
    pub fn merge_stats(&mut self, other: &Self) {
        if other.count == 0 {
            return;
        }

        if self.count == 0 {
            *self = other.clone();
            return;
        }

        let combined_count = self.count + other.count;
        let delta = other.mean - self.mean;

        // Combined mean
        let combined_mean = self.mean + delta * (other.count as f64 / combined_count as f64);

        // Combined M2 (Chan et al.'s parallel algorithm)
        let combined_m2 = self.m2
            + other.m2
            + delta * delta * (self.count as f64 * other.count as f64 / combined_count as f64);

        // Update
        self.count = combined_count;
        self.mean = combined_mean;
        self.m2 = combined_m2;
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
    }
}

impl Sketch for RunningStats {
    type Item = f64;

    fn update(&mut self, item: &Self::Item) {
        self.add(*item);
    }

    fn merge(&mut self, other: &Self) -> Result<(), MergeError> {
        self.merge_stats(other);
        Ok(())
    }

    fn clear(&mut self) {
        *self = Self::new();
    }

    fn size_bytes(&self) -> usize {
        core::mem::size_of::<Self>()
    }

    fn count(&self) -> u64 {
        self.count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let mut stats = RunningStats::new();

        stats.add(2.0);
        stats.add(4.0);
        stats.add(4.0);
        stats.add(4.0);
        stats.add(5.0);
        stats.add(5.0);
        stats.add(7.0);
        stats.add(9.0);

        assert_eq!(stats.len(), 8);
        assert!((stats.mean() - 5.0).abs() < 0.001);
        assert!((stats.variance() - 4.0).abs() < 0.001);
        assert!((stats.stddev() - 2.0).abs() < 0.001);
        assert_eq!(stats.min(), Some(2.0));
        assert_eq!(stats.max(), Some(9.0));
    }

    #[test]
    fn test_single_value() {
        let mut stats = RunningStats::new();
        stats.add(42.0);

        assert_eq!(stats.len(), 1);
        assert!((stats.mean() - 42.0).abs() < 0.001);
        assert!((stats.variance() - 0.0).abs() < 0.001);
        assert_eq!(stats.min(), Some(42.0));
        assert_eq!(stats.max(), Some(42.0));
    }

    #[test]
    fn test_empty() {
        let stats = RunningStats::new();

        assert!(stats.is_empty());
        assert_eq!(stats.mean(), 0.0);
        assert_eq!(stats.variance(), 0.0);
        assert_eq!(stats.min(), None);
        assert_eq!(stats.max(), None);
    }

    #[test]
    fn test_merge() {
        let mut stats1 = RunningStats::new();
        let mut stats2 = RunningStats::new();

        // Split data: [1,2,3] and [4,5,6]
        for v in [1.0, 2.0, 3.0] {
            stats1.add(v);
        }
        for v in [4.0, 5.0, 6.0] {
            stats2.add(v);
        }

        stats1.merge(&stats2).unwrap();

        assert_eq!(stats1.len(), 6);
        assert!((stats1.mean() - 3.5).abs() < 0.001);
        assert_eq!(stats1.min(), Some(1.0));
        assert_eq!(stats1.max(), Some(6.0));
        assert!((stats1.sum() - 21.0).abs() < 0.001);
    }

    #[test]
    fn test_merge_empty() {
        let mut stats1 = RunningStats::new();
        let stats2 = RunningStats::new();

        stats1.add(1.0);
        stats1.add(2.0);

        stats1.merge(&stats2).unwrap();

        assert_eq!(stats1.len(), 2);
        assert!((stats1.mean() - 1.5).abs() < 0.001);
    }

    #[test]
    fn test_sample_variance() {
        let mut stats = RunningStats::new();

        // Dataset: [2, 4, 4, 4, 5, 5, 7, 9]
        // Mean = 40/8 = 5.0
        // Population variance = 32/8 = 4.0
        // Sample variance = 32/7 ≈ 4.571 (Bessel's correction)
        for v in [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0] {
            stats.add(v);
        }

        // Population variance = 4.0
        assert!((stats.variance() - 4.0).abs() < 0.001);

        // Sample variance = 4.0 * 8/7 ≈ 4.571
        assert!((stats.sample_variance() - 4.571).abs() < 0.01);
    }

    #[test]
    fn test_clear() {
        let mut stats = RunningStats::new();

        stats.add(1.0);
        stats.add(2.0);
        stats.add(3.0);

        stats.clear();

        assert!(stats.is_empty());
        assert_eq!(stats.min(), None);
    }

    #[test]
    fn test_numerical_stability() {
        // Test with large values that could cause numerical issues
        let mut stats = RunningStats::new();

        let base = 1e12;
        for i in 0..1000 {
            stats.add(base + i as f64);
        }

        // Mean should be base + 499.5
        let expected_mean = base + 499.5;
        assert!(
            (stats.mean() - expected_mean).abs() < 1.0,
            "Mean: {} expected: {}",
            stats.mean(),
            expected_mean
        );
    }

    #[test]
    fn test_nan_ignored() {
        let mut stats = RunningStats::new();

        stats.add(1.0);
        stats.add(f64::NAN);
        stats.add(2.0);
        stats.add(f64::NAN);
        stats.add(3.0);

        // NaNs should be ignored
        assert_eq!(stats.len(), 3);
        assert!((stats.mean() - 2.0).abs() < 0.001);
        assert_eq!(stats.min(), Some(1.0));
        assert_eq!(stats.max(), Some(3.0));

        // Mean and variance should not be NaN
        assert!(!stats.mean().is_nan());
        assert!(!stats.variance().is_nan());
    }

    #[test]
    fn test_infinity() {
        let mut stats = RunningStats::new();

        stats.add(1.0);
        stats.add(f64::INFINITY);
        stats.add(2.0);

        // Infinity is a valid f64 value, should be included
        assert_eq!(stats.len(), 3);
        assert_eq!(stats.max(), Some(f64::INFINITY));
    }
}