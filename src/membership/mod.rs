//! Membership testing data structures
//!
//! This module provides probabilistic data structures for testing set membership.
//! These structures trade a small probability of false positives for significant
//! space savings compared to exact set representations.
//!
//! # Example
//!
//! ```
//! use flowstats::membership::BloomFilter;
//!
//! let mut bloom = BloomFilter::new(1000, 0.01);
//! bloom.insert(b"hello");
//! assert!(bloom.contains(b"hello"));
//! ```

mod bloom;

pub use bloom::BloomFilter;
