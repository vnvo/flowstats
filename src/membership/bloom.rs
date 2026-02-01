//! Bloom filter for probabilistic set membership
//!
//! A Bloom filter is a space-efficient probabilistic data structure that tests
//! whether an element is a member of a set. False positives are possible, but
//! false negatives are not.

use crate::math;
use crate::traits::{DecodeError, MembershipSketch, MergeError, Sketch};
use xxhash_rust::xxh3::xxh3_64_with_seed;

#[cfg(feature = "std")]
use std::vec::Vec;

#[cfg(not(feature = "std"))]
extern crate alloc;
#[cfg(not(feature = "std"))]
use alloc::vec::Vec;

/// Bloom filter for set membership testing
///
/// # Example
///
/// ```
/// use flowstats::membership::BloomFilter;
///
/// // Create filter for ~1000 items with 1% false positive rate
/// let mut bloom = BloomFilter::new(1000, 0.01);
///
/// bloom.insert(b"apple");
/// bloom.insert(b"banana");
///
/// assert!(bloom.contains(b"apple"));   // true - definitely inserted
/// assert!(bloom.contains(b"banana"));  // true - definitely inserted
/// assert!(!bloom.contains(b"cherry")); // probably false (might be false positive)
/// ```
///
/// # False Positive Rate
///
/// The actual false positive rate depends on the number of items inserted.
/// If you insert more items than the expected capacity, the false positive
/// rate will increase.
#[derive(Clone, Debug)]
pub struct BloomFilter {
    /// Bit array
    bits: Vec<u64>,
    /// Number of bits (m)
    num_bits: usize,
    /// Number of hash functions (k)
    num_hashes: usize,
    /// Number of items inserted
    count: u64,
    /// Seeds for hash functions
    seeds: Vec<u64>,
}

impl BloomFilter {
    /// Create a new Bloom filter with expected capacity and false positive rate
    ///
    /// # Arguments
    ///
    /// * `expected_items` - Expected number of items to insert
    /// * `false_positive_rate` - Desired false positive rate (e.g., 0.01 for 1%)
    ///
    /// # Panics
    ///
    /// Panics if `expected_items` is 0 or `false_positive_rate` is not in (0, 1)
    pub fn new(expected_items: usize, false_positive_rate: f64) -> Self {
        assert!(expected_items > 0, "expected_items must be positive");
        assert!(
            false_positive_rate > 0.0 && false_positive_rate < 1.0,
            "false_positive_rate must be in (0, 1)"
        );

        // Optimal number of bits: m = -n * ln(p) / (ln(2)^2)
        let ln2_squared = core::f64::consts::LN_2 * core::f64::consts::LN_2;
        let num_bits =
            math::ceil(-(expected_items as f64) * math::ln(false_positive_rate) / ln2_squared)
                as usize;
        let num_bits = num_bits.max(64); // Minimum 64 bits

        // Optimal number of hash functions: k = (m/n) * ln(2)
        let num_hashes =
            math::ceil((num_bits as f64 / expected_items as f64) * core::f64::consts::LN_2)
                as usize;
        let num_hashes = num_hashes.max(1).min(32); // Clamp to [1, 32]

        Self::with_params(num_bits, num_hashes)
    }

    /// Create a Bloom filter with specific parameters
    ///
    /// # Arguments
    ///
    /// * `num_bits` - Number of bits in the filter
    /// * `num_hashes` - Number of hash functions
    pub fn with_params(num_bits: usize, num_hashes: usize) -> Self {
        assert!(num_bits > 0, "num_bits must be positive");
        assert!(num_hashes > 0, "num_hashes must be positive");

        // Round up to multiple of 64 for word alignment
        let num_bits = (num_bits + 63) / 64 * 64;
        let num_words = num_bits / 64;

        // Generate seeds for hash functions
        let seeds: Vec<u64> = (0..num_hashes)
            .map(|i| (i as u64).wrapping_mul(0x9e3779b97f4a7c15))
            .collect();

        Self {
            bits: vec![0u64; num_words],
            num_bits,
            num_hashes,
            count: 0,
            seeds,
        }
    }

    /// Insert an item into the filter
    pub fn insert(&mut self, item: &[u8]) {
        self.count += 1;

        for &seed in &self.seeds {
            let hash = xxh3_64_with_seed(item, seed);
            let bit_idx = (hash as usize) % self.num_bits;
            let word_idx = bit_idx / 64;
            let bit_offset = bit_idx % 64;
            self.bits[word_idx] |= 1u64 << bit_offset;
        }
    }

    /// Check if an item might be in the filter
    ///
    /// Returns `true` if the item might be in the set (possibly a false positive),
    /// or `false` if the item is definitely not in the set.
    pub fn contains(&self, item: &[u8]) -> bool {
        for &seed in &self.seeds {
            let hash = xxh3_64_with_seed(item, seed);
            let bit_idx = (hash as usize) % self.num_bits;
            let word_idx = bit_idx / 64;
            let bit_offset = bit_idx % 64;
            if (self.bits[word_idx] & (1u64 << bit_offset)) == 0 {
                return false;
            }
        }
        true
    }

    /// Get the number of bits in the filter
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Get the number of hash functions
    pub fn num_hashes(&self) -> usize {
        self.num_hashes
    }

    /// Get the number of bits set to 1
    pub fn bits_set(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Estimate the current false positive rate
    ///
    /// This is based on the actual fill ratio of the filter.
    pub fn estimated_false_positive_rate(&self) -> f64 {
        let fill_ratio = self.bits_set() as f64 / self.num_bits as f64;
        math::powi(fill_ratio, self.num_hashes as i32)
    }

    /// Estimate the number of items in the filter
    ///
    /// Uses the fill ratio to estimate cardinality.
    pub fn estimated_count(&self) -> f64 {
        let bits_set = self.bits_set() as f64;
        let m = self.num_bits as f64;
        let k = self.num_hashes as f64;

        if bits_set >= m {
            return f64::INFINITY;
        }

        // n ≈ -m/k * ln(1 - X/m) where X is bits set
        -(m / k) * math::ln(1.0 - bits_set / m)
    }
}

impl Sketch for BloomFilter {
    type Item = [u8];

    fn update(&mut self, item: &Self::Item) {
        self.insert(item);
    }

    fn merge(&mut self, other: &Self) -> Result<(), MergeError> {
        if self.num_bits != other.num_bits || self.num_hashes != other.num_hashes {
            return Err(MergeError::IncompatibleConfig {
                expected: format!("bits={}, hashes={}", self.num_bits, self.num_hashes),
                found: format!("bits={}, hashes={}", other.num_bits, other.num_hashes),
            });
        }

        for (a, b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a |= *b;
        }
        self.count += other.count;

        Ok(())
    }

    fn clear(&mut self) {
        for word in &mut self.bits {
            *word = 0;
        }
        self.count = 0;
    }

    fn size_bytes(&self) -> usize {
        self.bits.len() * 8 + self.seeds.len() * 8 + 32
    }

    fn count(&self) -> u64 {
        self.count
    }
}

impl MembershipSketch for BloomFilter {
    fn contains(&self, item: &Self::Item) -> bool {
        self.contains(item)
    }

    fn false_positive_rate(&self) -> f64 {
        self.estimated_false_positive_rate()
    }

    fn len(&self) -> usize {
        self.count as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic() {
        let mut bloom = BloomFilter::new(1000, 0.01);

        bloom.insert(b"apple");
        bloom.insert(b"banana");
        bloom.insert(b"cherry");

        assert!(bloom.contains(b"apple"));
        assert!(bloom.contains(b"banana"));
        assert!(bloom.contains(b"cherry"));
    }

    #[test]
    fn test_no_false_negatives() {
        let mut bloom = BloomFilter::new(1000, 0.01);

        // Insert many items
        for i in 0..1000 {
            let item = format!("item_{}", i);
            bloom.insert(item.as_bytes());
        }

        // All inserted items must be found (no false negatives)
        for i in 0..1000 {
            let item = format!("item_{}", i);
            assert!(bloom.contains(item.as_bytes()), "Missing item_{}", i);
        }
    }

    #[test]
    fn test_false_positive_rate() {
        let mut bloom = BloomFilter::new(1000, 0.01);

        // Insert 1000 items
        for i in 0..1000 {
            let item = format!("item_{}", i);
            bloom.insert(item.as_bytes());
        }

        // Test 10000 items that were NOT inserted
        let mut false_positives = 0;
        for i in 0..10000 {
            let item = format!("other_{}", i);
            if bloom.contains(item.as_bytes()) {
                false_positives += 1;
            }
        }

        // False positive rate should be roughly 1% (allow some margin)
        let fp_rate = false_positives as f64 / 10000.0;
        assert!(fp_rate < 0.03, "FP rate too high: {}", fp_rate);
    }

    #[test]
    fn test_merge() {
        let mut bloom1 = BloomFilter::new(1000, 0.01);
        let mut bloom2 = BloomFilter::new(1000, 0.01);

        bloom1.insert(b"apple");
        bloom2.insert(b"banana");

        bloom1.merge(&bloom2).unwrap();

        assert!(bloom1.contains(b"apple"));
        assert!(bloom1.contains(b"banana"));
    }

    #[test]
    fn test_merge_incompatible() {
        let mut bloom1 = BloomFilter::new(1000, 0.01);
        let bloom2 = BloomFilter::new(2000, 0.01);

        assert!(bloom1.merge(&bloom2).is_err());
    }

    #[test]
    fn test_clear() {
        let mut bloom = BloomFilter::new(100, 0.01);

        bloom.insert(b"apple");
        assert!(bloom.contains(b"apple"));

        bloom.clear();
        assert!(!bloom.contains(b"apple"));
        assert_eq!(bloom.count(), 0);
    }

    #[test]
    fn test_estimated_count() {
        let mut bloom = BloomFilter::new(1000, 0.01);

        for i in 0..500 {
            let item = format!("item_{}", i);
            bloom.insert(item.as_bytes());
        }

        let estimated = bloom.estimated_count();
        // Should be roughly 500, allow 20% error
        assert!(
            estimated > 400.0 && estimated < 600.0,
            "Estimate: {}",
            estimated
        );
    }
}
