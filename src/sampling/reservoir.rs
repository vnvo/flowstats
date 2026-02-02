//! Reservoir sampling for uniform random sampling from streams
//!
//! Reservoir sampling maintains a fixed-size uniform random sample from
//! a stream of unknown length. Each item in the stream has equal probability
//! of being in the final sample.

use crate::traits::{MergeError, Sketch};

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Simple xorshift64 PRNG for no_std compatibility
#[derive(Clone, Debug)]
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x853c49e6748fea9b } else { seed },
        }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Generate random usize in [0, bound)
    fn next_bounded(&mut self, bound: usize) -> usize {
        (self.next() as usize) % bound
    }
}

/// Reservoir sampler using Algorithm R
///
/// Maintains a uniform random sample of fixed size from a stream of arbitrary length.
/// Each item in the stream has probability k/n of being in the final sample,
/// where k is the reservoir size and n is the total number of items seen.
///
/// # Algorithm
///
/// Algorithm R (Vitter, 1985):
/// 1. Fill reservoir with first k items
/// 2. For each subsequent item i (1-indexed):
///    - Generate random j in [0, i)
///    - If j < k, replace reservoir[j] with item i
///
/// # Example
///
/// ```
/// use flowstats::sampling::ReservoirSampler;
///
/// let mut sampler = ReservoirSampler::<i32>::new(5);
///
/// // Stream 100 items
/// for i in 0..100 {
///     sampler.add(i);
/// }
///
/// // Get uniform random sample of 5 items
/// let sample = sampler.sample();
/// assert_eq!(sample.len(), 5);
/// ```
#[derive(Clone, Debug)]
pub struct ReservoirSampler<T: Clone + core::fmt::Debug> {
    /// Maximum sample size
    capacity: usize,
    /// Current sample
    reservoir: Vec<T>,
    /// Number of items seen
    count: u64,
    /// Random number generator
    rng: Xorshift64,
}

impl<T: Clone + core::fmt::Debug> ReservoirSampler<T> {
    /// Create a new reservoir sampler with given capacity
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of items to sample
    pub fn new(capacity: usize) -> Self {
        Self::with_seed(capacity, 0x12345678)
    }

    /// Create a new reservoir sampler with given capacity and seed
    ///
    /// # Arguments
    ///
    /// * `capacity` - Maximum number of items to sample
    /// * `seed` - Seed for random number generator (for reproducibility)
    pub fn with_seed(capacity: usize, seed: u64) -> Self {
        assert!(capacity > 0, "capacity must be positive");

        Self {
            capacity,
            reservoir: Vec::with_capacity(capacity),
            count: 0,
            rng: Xorshift64::new(seed),
        }
    }

    /// Add an item to the sampler
    pub fn add(&mut self, item: T) {
        self.count += 1;

        if self.reservoir.len() < self.capacity {
            // Reservoir not full - just add
            self.reservoir.push(item);
        } else {
            // Reservoir full - maybe replace
            let j = self.rng.next_bounded(self.count as usize);
            if j < self.capacity {
                self.reservoir[j] = item;
            }
        }
    }

    /// Get the current sample
    pub fn sample(&self) -> &[T] {
        &self.reservoir
    }

    /// Get the sample as a mutable slice
    pub fn sample_mut(&mut self) -> &mut [T] {
        &mut self.reservoir
    }

    /// Consume the sampler and return the sample
    pub fn into_sample(self) -> Vec<T> {
        self.reservoir
    }

    /// Get the reservoir capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get the current sample size
    pub fn len(&self) -> usize {
        self.reservoir.len()
    }

    /// Check if reservoir is empty
    pub fn is_empty(&self) -> bool {
        self.reservoir.is_empty()
    }

    /// Check if reservoir is full
    pub fn is_full(&self) -> bool {
        self.reservoir.len() >= self.capacity
    }

    /// Get the number of items seen
    pub fn items_seen(&self) -> u64 {
        self.count
    }

    /// Get the sampling probability for the current state
    ///
    /// This is the probability that any given item from the stream
    /// is in the current sample.
    pub fn sampling_probability(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            ((self.capacity as f64) / (self.count as f64)).min(1.0)
        }
    }
}

impl<T: Clone + core::fmt::Debug> Sketch for ReservoirSampler<T> {
    type Item = T;

    fn update(&mut self, item: &Self::Item) {
        self.add(item.clone());
    }

    fn merge(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.capacity != other.capacity {
            return Err(MergeError::IncompatibleConfig {
                expected: format!("capacity={}", self.capacity),
                found: format!("capacity={}", other.capacity),
            });
        }

        // Merge by sampling from combined pool
        // This is a simplified merge - for exact merging, you'd need
        // weighted sampling based on items_seen
        for item in &other.reservoir {
            self.count += 1;
            let j = self.rng.next_bounded(self.count as usize);
            if j < self.capacity && j < self.reservoir.len() {
                self.reservoir[j] = item.clone();
            }
        }

        Ok(())
    }

    fn clear(&mut self) {
        self.reservoir.clear();
        self.count = 0;
    }

    fn size_bytes(&self) -> usize {
        self.reservoir.capacity() * core::mem::size_of::<T>() + 32
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
        let mut sampler = ReservoirSampler::<i32>::new(5);

        for i in 0..10 {
            sampler.add(i);
        }

        assert_eq!(sampler.len(), 5);
        assert_eq!(sampler.items_seen(), 10);
    }

    #[test]
    fn test_underfilled() {
        let mut sampler = ReservoirSampler::<i32>::new(10);

        for i in 0..5 {
            sampler.add(i);
        }

        assert_eq!(sampler.len(), 5);
        assert!(!sampler.is_full());

        // Should contain exactly items 0-4
        let sample = sampler.sample();
        for i in 0..5 {
            assert!(sample.contains(&i));
        }
    }

    #[test]
    fn test_reproducibility() {
        let mut sampler1 = ReservoirSampler::<i32>::with_seed(5, 42);
        let mut sampler2 = ReservoirSampler::<i32>::with_seed(5, 42);

        for i in 0..100 {
            sampler1.add(i);
            sampler2.add(i);
        }

        assert_eq!(sampler1.sample(), sampler2.sample());
    }

    #[test]
    fn test_uniformity() {
        // Statistical test: each item should appear with roughly equal frequency
        let mut counts = [0usize; 10];
        let iterations = 10000;

        for i in 0..iterations {
            let seed = (i as u64)
                .wrapping_mul(0x9e3779b97f4a7c15)
                .wrapping_add(0x853c49e6748fea9b);
            let mut sampler = ReservoirSampler::<usize>::with_seed(1, seed as u64);
            for i in 0..10 {
                sampler.add(i);
            }
            counts[sampler.sample()[0]] += 1;
        }

        // Each item should appear roughly iterations/10 times
        let expected = iterations / 10;
        for (i, &count) in counts.iter().enumerate() {
            let deviation = (count as i64 - expected as i64).abs() as f64 / expected as f64;
            assert!(
                deviation < 0.1,
                "Item {} appeared {} times (expected ~{})",
                i,
                count,
                expected
            );
        }
    }

    #[test]
    fn test_clear() {
        let mut sampler = ReservoirSampler::<i32>::new(5);

        for i in 0..10 {
            sampler.add(i);
        }

        sampler.clear();

        assert!(sampler.is_empty());
        assert_eq!(sampler.count(), 0);
    }

    #[test]
    fn test_into_sample() {
        let mut sampler = ReservoirSampler::<String>::new(3);

        sampler.add("a".to_string());
        sampler.add("b".to_string());
        sampler.add("c".to_string());

        let sample = sampler.into_sample();
        assert_eq!(sample.len(), 3);
    }
}
