//! Correctness and invariant tests for flowstats
//!
//! These tests verify critical invariants, merge semantics, and edge cases
//! across all algorithm families. They complement the unit tests in each module
//! by focusing on properties that must always hold.
//!
//! Run with: cargo test --test correctness --features full

// Require all features
#[cfg(not(all(
    feature = "cardinality",
    feature = "frequency",
    feature = "quantiles",
    feature = "membership",
    feature = "sampling",
    feature = "statistics"
)))]
compile_error!(
    "Correctness tests require all features. Run: cargo test --test correctness --features full"
);

use flowstats::cardinality::HyperLogLog;
use flowstats::frequency::CountMinSketch;
use flowstats::membership::BloomFilter;
use flowstats::quantiles::TDigest;
use flowstats::sampling::ReservoirSampler;
use flowstats::statistics::RunningStats;
use flowstats::traits::{CardinalitySketch, FrequencySketch, QuantileSketch, Sketch};

// ============================================================================
// Reservoir Sampler
// ============================================================================

mod reservoir {
    use super::*;

    #[test]
    fn merge_preserves_total_count() {
        let mut s1 = ReservoirSampler::<u64>::with_seed(10, 42);
        let mut s2 = ReservoirSampler::<u64>::with_seed(10, 99);

        for i in 0..10_000 {
            s1.add(i);
        }
        for i in 10_000..20_000 {
            s2.add(i);
        }

        assert_eq!(s1.items_seen(), 10_000);
        assert_eq!(s2.items_seen(), 10_000);

        s1.merge(&s2).unwrap();

        assert_eq!(
            s1.items_seen(),
            20_000,
            "After merging two samplers that each saw 10,000 items, \
             total count should be 20,000 but got {}",
            s1.items_seen()
        );
    }

    #[test]
    fn merge_produces_unbiased_sample() {
        let capacity = 1000;
        let n_per_side = 100_000u64;
        let trials = 50;
        let mut total_from_side2 = 0usize;

        for trial in 0..trials {
            let seed1 = 1000 + trial * 7;
            let seed2 = 2000 + trial * 13;

            let mut s1 = ReservoirSampler::<u64>::with_seed(capacity, seed1);
            let mut s2 = ReservoirSampler::<u64>::with_seed(capacity, seed2);

            for i in 0..n_per_side {
                s1.add(i);
            }
            for i in n_per_side..(2 * n_per_side) {
                s2.add(i);
            }

            s1.merge(&s2).unwrap();

            let from_side2 = s1.sample().iter().filter(|&&x| x >= n_per_side).count();
            total_from_side2 += from_side2;
        }

        // Expected: ~50% from each side
        let expected = (capacity as f64 * trials as f64) / 2.0;
        let actual = total_from_side2 as f64;
        let relative_error = (actual - expected).abs() / expected;

        assert!(
            relative_error < 0.15,
            "Merged reservoir is biased: {:.1}% of items from side 2 \
             (expected ~50%, relative error {:.1}%)",
            (actual / (capacity as f64 * trials as f64)) * 100.0,
            relative_error * 100.0,
        );
    }

    #[test]
    fn merge_sampling_probability_correct() {
        let mut s1 = ReservoirSampler::<u64>::with_seed(100, 1);
        let mut s2 = ReservoirSampler::<u64>::with_seed(100, 2);

        for i in 0..50_000 {
            s1.add(i);
        }
        for i in 50_000..100_000 {
            s2.add(i);
        }

        s1.merge(&s2).unwrap();

        let prob = s1.sampling_probability();
        assert!(
            (prob - 0.001).abs() < 0.0005,
            "sampling_probability after merge should be ~0.001 but got {}",
            prob
        );
    }

    #[test]
    fn merge_underfilled_reservoirs_preserves_all_items() {
        let mut s1 = ReservoirSampler::<u64>::with_seed(100, 1);
        let mut s2 = ReservoirSampler::<u64>::with_seed(100, 2);

        for i in 0..30 {
            s1.add(i);
        }
        for i in 30..50 {
            s2.add(i);
        }

        s1.merge(&s2).unwrap();

        assert_eq!(
            s1.items_seen(),
            50,
            "Merge of underfilled reservoirs: count should be 50, got {}",
            s1.items_seen()
        );

        // Capacity 100, only 50 items total — all should be present
        assert_eq!(s1.len(), 50);

        let sample = s1.sample();
        for i in 0..50u64 {
            assert!(
                sample.contains(&i),
                "Item {} missing from merged underfilled reservoir",
                i
            );
        }
    }

    #[test]
    fn merge_with_empty_is_identity() {
        let mut s1 = ReservoirSampler::<u64>::with_seed(10, 1);
        let s2 = ReservoirSampler::<u64>::with_seed(10, 2);

        for i in 0..1000 {
            s1.add(i);
        }

        let count_before = s1.items_seen();
        let sample_before: Vec<u64> = s1.sample().to_vec();

        s1.merge(&s2).unwrap();

        assert_eq!(
            s1.items_seen(),
            count_before,
            "Merging with empty sampler should not change count"
        );
        assert_eq!(
            s1.sample(),
            sample_before.as_slice(),
            "Merging with empty sampler should not change sample"
        );
    }

    #[test]
    fn merge_empty_into_empty() {
        let mut s1 = ReservoirSampler::<u64>::with_seed(10, 1);
        let s2 = ReservoirSampler::<u64>::with_seed(10, 2);

        s1.merge(&s2).unwrap();

        assert_eq!(s1.items_seen(), 0);
        assert!(s1.is_empty());
    }

    #[test]
    fn merge_empty_into_populated() {
        let mut empty = ReservoirSampler::<u64>::with_seed(10, 1);
        let mut populated = ReservoirSampler::<u64>::with_seed(10, 2);

        for i in 0..1000 {
            populated.add(i);
        }

        empty.merge(&populated).unwrap();

        assert_eq!(empty.items_seen(), 1000);
        assert_eq!(empty.len(), 10);
    }

    #[test]
    fn sample_size_never_exceeds_capacity() {
        let capacity = 50;
        let mut s = ReservoirSampler::<u64>::with_seed(capacity, 1);

        for i in 0..1_000_000 {
            s.add(i);
            assert!(
                s.len() <= capacity,
                "Sample size {} exceeds capacity {} after {} inserts",
                s.len(),
                capacity,
                i + 1
            );
        }
    }
}

// ============================================================================
// TDigest
// ============================================================================

mod tdigest {
    use super::*;

    #[test]
    fn quantile_correct_with_unflushed_buffer() {
        // compression=100 → buffer_capacity=200
        // Add fewer than 200 items so compress() is never auto-triggered
        let mut digest = TDigest::new(100.0);

        for i in 1..=50 {
            digest.add(i as f64);
        }

        let median = digest.median();
        assert!(
            median.is_some(),
            "median() should return Some for non-empty digest"
        );

        let median = median.unwrap();
        assert!(
            median > 20.0 && median < 31.0,
            "Median of 1..=50 should be near 25.5, got {}",
            median
        );

        let p10 = digest.quantile(0.1).unwrap();
        let p90 = digest.quantile(0.9).unwrap();

        assert!(
            p10 > 2.0 && p10 < 10.0,
            "p10 of 1..=50 should be ~5, got {}",
            p10
        );
        assert!(
            p90 > 40.0 && p90 < 50.0,
            "p90 of 1..=50 should be ~45, got {}",
            p90
        );
    }

    #[test]
    fn repeated_quantile_queries_are_consistent() {
        let mut digest = TDigest::new(100.0);

        for i in 1..=100 {
            digest.add(i as f64);
        }

        // Don't call compress() — leave data in buffer
        let first = digest.quantile(0.5).unwrap();
        for _ in 0..100 {
            let q = digest.quantile(0.5).unwrap();
            assert!(
                (q - first).abs() < f64::EPSILON,
                "Repeated quantile(0.5) calls gave different results: {} vs {}",
                first,
                q
            );
        }
    }

    #[test]
    fn merge_with_unflushed_buffers() {
        let mut d1 = TDigest::new(100.0);
        let mut d2 = TDigest::new(100.0);

        for i in 1..=50 {
            d1.add(i as f64);
        }
        for i in 51..=100 {
            d2.add(i as f64);
        }

        d1.merge(&d2).unwrap();

        assert_eq!(d1.count(), 100);
        assert_eq!(d1.min(), Some(1.0));
        assert_eq!(d1.max(), Some(100.0));

        let median = d1.median().unwrap();
        assert!(
            median > 40.0 && median < 60.0,
            "Median of 1..=100 after merge should be ~50.5, got {}",
            median
        );
    }

    #[test]
    fn rank_quantile_roundtrip() {
        let mut digest = TDigest::new(200.0);

        for i in 1..=10_000 {
            digest.add(i as f64);
        }
        digest.compress();

        for &v in &[100.0, 500.0, 1000.0, 5000.0, 9000.0, 9900.0] {
            let q = digest.rank(&v);
            let recovered = digest.quantile(q).unwrap();
            let error = (recovered - v).abs() / v;
            assert!(
                error < 0.05,
                "Round-trip failed for v={}: rank={:.4}, quantile(rank)={:.1}, error={:.2}%",
                v,
                q,
                recovered,
                error * 100.0
            );
        }
    }

    #[test]
    fn quantile_is_monotonically_nondecreasing() {
        let mut digest = TDigest::new(100.0);

        for i in 1..=1000 {
            digest.add(i as f64);
        }
        digest.compress();

        let mut prev = f64::NEG_INFINITY;
        for i in 0..=100 {
            let q = i as f64 / 100.0;
            let val = digest.quantile(q).unwrap();
            assert!(
                val >= prev,
                "Quantile not monotonic: quantile({:.2}) = {} < quantile({:.2}) = {}",
                q,
                val,
                (i - 1) as f64 / 100.0,
                prev
            );
            prev = val;
        }
    }

    #[test]
    fn quantile_extremes_are_exact() {
        let mut digest = TDigest::new(100.0);

        for i in 1..=1000 {
            digest.add(i as f64);
        }

        assert_eq!(digest.quantile(0.0), Some(1.0));
        assert_eq!(digest.quantile(1.0), Some(1000.0));
    }

    #[test]
    fn quantile_empty_returns_none() {
        let digest = TDigest::new(100.0);

        assert!(digest.quantile(0.5).is_none());
        assert!(digest.min().is_none());
        assert!(digest.max().is_none());
    }

    #[test]
    fn merge_preserves_min_max() {
        let mut d1 = TDigest::new(100.0);
        let mut d2 = TDigest::new(100.0);

        for i in 100..=500 {
            d1.add(i as f64);
        }
        for i in 1..=99 {
            d2.add(i as f64);
        }

        d1.merge(&d2).unwrap();

        assert_eq!(d1.min(), Some(1.0));
        assert_eq!(d1.max(), Some(500.0));
        assert_eq!(d1.count(), 500);
    }

    #[test]
    fn merge_with_empty_is_identity() {
        let mut d1 = TDigest::new(100.0);
        let d2 = TDigest::new(100.0);

        for i in 1..=100 {
            d1.add(i as f64);
        }
        d1.compress();

        let median_before = d1.median().unwrap();

        d1.merge(&d2).unwrap();

        assert_eq!(d1.count(), 100);
        let median_after = d1.median().unwrap();
        assert!(
            (median_before - median_after).abs() < 1.0,
            "Merge with empty changed median: {} -> {}",
            median_before,
            median_after
        );
    }

    /// Performance: quantile queries should not clone the entire digest on
    /// every call. With a large buffer, 200 calls should complete quickly.
    #[test]
    fn quantile_query_performance_with_buffer() {
        let mut digest = TDigest::new(5000.0); // buffer_capacity = 10_000

        for i in 0..9_999 {
            digest.add(i as f64);
        }

        let start = std::time::Instant::now();
        for _ in 0..200 {
            let _ = digest.quantile(0.5);
        }
        let elapsed = start.elapsed();

        assert!(
            elapsed.as_secs_f64() < 2.0,
            "200 quantile queries took {:.2}s — possible clone-per-query regression",
            elapsed.as_secs_f64()
        );
    }
}

// ============================================================================
// HyperLogLog
// ============================================================================

mod hyperloglog {
    use super::*;

    #[test]
    fn duplicates_do_not_inflate_estimate() {
        let mut hll = HyperLogLog::new(14);

        for _ in 0..1_000_000 {
            hll.insert("same_item");
        }

        let estimate = hll.estimate();
        assert!(
            estimate >= 0.5 && estimate <= 2.0,
            "1M inserts of same item should estimate ~1, got {}",
            estimate
        );
    }

    #[test]
    fn error_within_theoretical_bounds() {
        let precision = 12u8;
        let true_cardinality = 100_000usize;
        let trials = 20;
        let mut total_relative_error = 0.0;

        for trial in 0..trials {
            let mut hll = HyperLogLog::new(precision);

            for i in 0..true_cardinality {
                hll.insert(&format!("t{}_item_{}", trial, i));
            }

            let estimate = hll.estimate();
            let relative_error =
                (estimate - true_cardinality as f64).abs() / true_cardinality as f64;
            total_relative_error += relative_error;
        }

        let avg_error = total_relative_error / trials as f64;
        assert!(
            avg_error < 0.05,
            "Average relative error over {} trials at p={} is {:.2}%, expected < 5%",
            trials,
            precision,
            avg_error * 100.0
        );
    }

    #[test]
    fn merge_disjoint_sets_estimate_sum() {
        let mut hll1 = HyperLogLog::new(14);
        let mut hll2 = HyperLogLog::new(14);

        for i in 0..50_000 {
            hll1.insert(&format!("a_{}", i));
        }
        for i in 0..50_000 {
            hll2.insert(&format!("b_{}", i));
        }

        hll1.merge(&hll2).unwrap();

        let estimate = hll1.estimate();
        assert!(
            estimate > 90_000.0 && estimate < 110_000.0,
            "Merge of two disjoint 50K sets should estimate ~100K, got {}",
            estimate
        );
    }

    #[test]
    fn merge_overlapping_sets_does_not_double_count() {
        let mut hll1 = HyperLogLog::new(14);
        let mut hll2 = HyperLogLog::new(14);

        for i in 0..10_000 {
            hll1.insert(&format!("item_{}", i));
            hll2.insert(&format!("item_{}", i));
        }

        hll1.merge(&hll2).unwrap();

        let estimate = hll1.estimate();
        assert!(
            estimate > 9_000.0 && estimate < 11_000.0,
            "Merge of identical sets should estimate ~10K, got {}",
            estimate
        );
    }

    #[test]
    fn merge_with_empty_is_identity() {
        let mut hll1 = HyperLogLog::new(12);
        let hll2 = HyperLogLog::new(12);

        for i in 0..10_000 {
            hll1.insert(&format!("item_{}", i));
        }

        let est_before = hll1.estimate();
        hll1.merge(&hll2).unwrap();
        let est_after = hll1.estimate();

        assert!(
            (est_before - est_after).abs() < 1.0,
            "Merge with empty changed estimate: {} -> {}",
            est_before,
            est_after
        );
    }

    #[test]
    fn estimate_nonnegative() {
        let hll = HyperLogLog::new(10);
        assert!(hll.estimate() >= 0.0);

        let mut hll2 = HyperLogLog::new(10);
        hll2.insert("x");
        assert!(hll2.estimate() >= 0.0);
    }

    #[test]
    fn clear_resets_completely() {
        let mut hll = HyperLogLog::new(12);

        for i in 0..10_000 {
            hll.insert(&format!("item_{}", i));
        }

        hll.clear();

        assert_eq!(hll.estimate(), 0.0);
        assert_eq!(hll.count(), 0);
    }
}

// ============================================================================
// Bloom Filter
// ============================================================================

mod bloom {
    use super::*;

    /// The absolute invariant: no false negatives, ever.
    #[test]
    fn zero_false_negatives() {
        let mut bloom = BloomFilter::new(10_000, 0.01);

        let items: Vec<String> = (0..10_000).map(|i| format!("item_{}", i)).collect();

        for item in &items {
            bloom.insert(item.as_bytes());
        }

        for item in &items {
            assert!(
                bloom.contains(item.as_bytes()),
                "FALSE NEGATIVE: '{}' was inserted but contains() returned false",
                item
            );
        }
    }

    /// Merge must preserve the zero-false-negatives invariant.
    #[test]
    fn merge_preserves_zero_false_negatives() {
        let mut bloom1 = BloomFilter::new(10_000, 0.01);
        let mut bloom2 = BloomFilter::new(10_000, 0.01);

        let items1: Vec<String> = (0..5_000).map(|i| format!("a_{}", i)).collect();
        let items2: Vec<String> = (0..5_000).map(|i| format!("b_{}", i)).collect();

        for item in &items1 {
            bloom1.insert(item.as_bytes());
        }
        for item in &items2 {
            bloom2.insert(item.as_bytes());
        }

        bloom1.merge(&bloom2).unwrap();

        for item in items1.iter().chain(items2.iter()) {
            assert!(
                bloom1.contains(item.as_bytes()),
                "FALSE NEGATIVE after merge: '{}' missing",
                item
            );
        }
    }

    #[test]
    fn false_positive_rate_within_tolerance() {
        let expected_items = 10_000;
        let target_fpr = 0.01;
        let mut bloom = BloomFilter::new(expected_items, target_fpr);

        for i in 0..expected_items {
            bloom.insert(format!("item_{}", i).as_bytes());
        }

        let mut false_positives = 0;
        let test_count = 100_000;
        for i in 0..test_count {
            if bloom.contains(format!("other_{}", i).as_bytes()) {
                false_positives += 1;
            }
        }

        let actual_fpr = false_positives as f64 / test_count as f64;
        assert!(
            actual_fpr < target_fpr * 3.0,
            "FP rate {:.4} exceeds 3x target {:.4}",
            actual_fpr,
            target_fpr
        );
    }

    #[test]
    fn merge_with_empty_is_identity() {
        let mut bloom1 = BloomFilter::new(1000, 0.01);
        let bloom2 = BloomFilter::new(1000, 0.01);

        bloom1.insert(b"hello");
        let bits_before = bloom1.bits_set();

        bloom1.merge(&bloom2).unwrap();

        assert_eq!(bloom1.bits_set(), bits_before);
        assert!(bloom1.contains(b"hello"));
    }

    #[test]
    fn clear_resets_completely() {
        let mut bloom = BloomFilter::new(1000, 0.01);
        bloom.insert(b"hello");

        bloom.clear();

        assert!(!bloom.contains(b"hello"));
        assert_eq!(bloom.count(), 0);
        assert_eq!(bloom.bits_set(), 0);
    }
}

// ============================================================================
// Count-Min Sketch
// ============================================================================

mod count_min_sketch {
    use super::*;

    /// CMS never underestimates (point query guarantee).
    #[test]
    fn estimate_never_underestimates() {
        let mut cms = CountMinSketch::new(0.01, 0.01);

        cms.add(b"apple", 100);
        cms.add(b"banana", 50);
        cms.add(b"cherry", 1);

        assert!(cms.estimate(b"apple") >= 100);
        assert!(cms.estimate(b"banana") >= 50);
        assert!(cms.estimate(b"cherry") >= 1);
    }

    /// Merge preserves the never-underestimate invariant.
    #[test]
    fn merge_preserves_lower_bound() {
        let mut cms1 = CountMinSketch::with_dimensions(1000, 5);
        let mut cms2 = CountMinSketch::with_dimensions(1000, 5);

        cms1.add(b"apple", 30);
        cms2.add(b"apple", 70);
        cms1.add(b"banana", 50);

        cms1.merge(&cms2).unwrap();

        assert!(
            cms1.estimate(b"apple") >= 100,
            "After merge, apple estimate {} < true count 100",
            cms1.estimate(b"apple")
        );
        assert!(
            cms1.estimate(b"banana") >= 50,
            "After merge, banana estimate {} < true count 50",
            cms1.estimate(b"banana")
        );
    }

    #[test]
    fn merge_total_count_is_sum() {
        let mut cms1 = CountMinSketch::with_dimensions(1000, 5);
        let mut cms2 = CountMinSketch::with_dimensions(1000, 5);

        cms1.add(b"a", 10);
        cms2.add(b"b", 20);

        cms1.merge(&cms2).unwrap();
        assert_eq!(cms1.total_count(), 30);
    }

    #[test]
    fn unseen_items_estimate_zero() {
        let cms = CountMinSketch::new(0.01, 0.01);
        assert_eq!(cms.estimate(b"never_added"), 0);
    }

    #[test]
    fn clear_resets_completely() {
        let mut cms = CountMinSketch::new(0.01, 0.01);
        cms.add(b"item", 100);

        cms.clear();

        assert_eq!(cms.estimate(b"item"), 0);
        assert_eq!(cms.total_count(), 0);
    }
}

// ============================================================================
// Space-Saving
// ============================================================================

mod space_saving {
    use super::*;
    use flowstats::frequency::SpaceSaving;
    use flowstats::traits::HeavyHitters;

    /// Any item with frequency > n/k is guaranteed to be tracked.
    #[test]
    fn heavy_hitter_guarantee() {
        let k = 10;
        let mut ss = SpaceSaving::<String>::new(k);

        for _ in 0..1000 {
            ss.add("hot".to_string());
        }
        for i in 0..100 {
            ss.add(format!("cold_{}", i));
        }

        assert!(
            ss.contains(&"hot".to_string()),
            "Item 'hot' with frequency 1000 > n/k=110 must be tracked"
        );

        let top = ss.top_k(1);
        assert_eq!(top[0].0, "hot");
    }

    #[test]
    fn merge_returns_error() {
        let mut ss1 = SpaceSaving::<String>::new(10);
        let ss2 = SpaceSaving::<String>::new(10);

        assert!(ss1.merge(&ss2).is_err());
    }

    #[test]
    fn clear_resets_completely() {
        let mut ss = SpaceSaving::<String>::new(10);
        ss.add("apple".to_string());

        ss.clear();

        assert_eq!(ss.num_tracked(), 0);
        assert_eq!(ss.total_count(), 0);
        assert!(!ss.contains(&"apple".to_string()));
    }
}

// ============================================================================
// Running Stats
// ============================================================================

mod running_stats {
    use super::*;

    #[test]
    fn merge_is_commutative() {
        let mut a = RunningStats::new();
        let mut b = RunningStats::new();

        for v in [1.0, 3.0, 5.0, 7.0, 9.0] {
            a.add(v);
        }
        for v in [2.0, 4.0, 6.0, 8.0, 10.0] {
            b.add(v);
        }

        let mut ab = a.clone();
        ab.merge(&b).unwrap();

        let mut ba = b.clone();
        ba.merge(&a).unwrap();

        assert_eq!(ab.len(), ba.len());
        assert!(
            (ab.mean() - ba.mean()).abs() < 1e-10,
            "mean: {} vs {}",
            ab.mean(),
            ba.mean()
        );
        assert!(
            (ab.variance() - ba.variance()).abs() < 1e-10,
            "variance: {} vs {}",
            ab.variance(),
            ba.variance()
        );
        assert_eq!(ab.min(), ba.min());
        assert_eq!(ab.max(), ba.max());
    }

    #[test]
    fn merge_is_associative() {
        let mut a = RunningStats::new();
        let mut b = RunningStats::new();
        let mut c = RunningStats::new();

        for v in [1.0, 2.0, 3.0] {
            a.add(v);
        }
        for v in [4.0, 5.0, 6.0] {
            b.add(v);
        }
        for v in [7.0, 8.0, 9.0] {
            c.add(v);
        }

        // (A merge B) merge C
        let mut ab_c = a.clone();
        ab_c.merge(&b).unwrap();
        ab_c.merge(&c).unwrap();

        // A merge (B merge C)
        let mut bc = b.clone();
        bc.merge(&c).unwrap();
        let mut a_bc = a.clone();
        a_bc.merge(&bc).unwrap();

        assert_eq!(ab_c.len(), a_bc.len());
        assert!(
            (ab_c.mean() - a_bc.mean()).abs() < 1e-10,
            "mean: {} vs {}",
            ab_c.mean(),
            a_bc.mean()
        );
        assert!(
            (ab_c.variance() - a_bc.variance()).abs() < 1e-10,
            "variance: {} vs {}",
            ab_c.variance(),
            a_bc.variance()
        );
    }

    #[test]
    fn merge_equivalent_to_sequential_add() {
        let data_a = [1.5, 3.7, 2.1, 8.9, 4.3];
        let data_b = [6.2, 7.4, 0.5, 9.1, 5.6];

        // Sequential
        let mut sequential = RunningStats::new();
        for &v in data_a.iter().chain(data_b.iter()) {
            sequential.add(v);
        }

        // Merged
        let mut sa = RunningStats::new();
        let mut sb = RunningStats::new();
        for &v in &data_a {
            sa.add(v);
        }
        for &v in &data_b {
            sb.add(v);
        }
        sa.merge(&sb).unwrap();

        assert_eq!(sa.len(), sequential.len());
        assert!(
            (sa.mean() - sequential.mean()).abs() < 1e-10,
            "mean: {} vs {}",
            sa.mean(),
            sequential.mean()
        );
        assert!(
            (sa.variance() - sequential.variance()).abs() < 1e-10,
            "variance: {} vs {}",
            sa.variance(),
            sequential.variance()
        );
        assert_eq!(sa.min(), sequential.min());
        assert_eq!(sa.max(), sequential.max());
    }

    #[test]
    fn merge_into_empty() {
        let mut empty = RunningStats::new();
        let mut populated = RunningStats::new();

        for v in [1.0, 2.0, 3.0, 4.0, 5.0] {
            populated.add(v);
        }

        empty.merge(&populated).unwrap();

        assert_eq!(empty.len(), 5);
        assert!((empty.mean() - 3.0).abs() < 0.001);
        assert_eq!(empty.min(), Some(1.0));
        assert_eq!(empty.max(), Some(5.0));
        assert!((empty.sum() - 15.0).abs() < 0.001);
        assert!((empty.variance() - 2.0).abs() < 0.001);
    }

    #[test]
    fn merge_empty_into_empty() {
        let mut a = RunningStats::new();
        let b = RunningStats::new();

        a.merge(&b).unwrap();

        assert!(a.is_empty());
        assert_eq!(a.mean(), 0.0);
        assert_eq!(a.min(), None);
    }

    #[test]
    fn nan_values_are_ignored() {
        let mut stats = RunningStats::new();

        stats.add(1.0);
        stats.add(f64::NAN);
        stats.add(2.0);
        stats.add(f64::NAN);
        stats.add(3.0);

        assert_eq!(stats.len(), 3);
        assert!((stats.mean() - 2.0).abs() < 0.001);
        assert!(!stats.mean().is_nan());
        assert!(!stats.variance().is_nan());
    }

    #[test]
    fn clear_resets_completely() {
        let mut stats = RunningStats::new();
        for v in [1.0, 2.0, 3.0] {
            stats.add(v);
        }

        stats.clear();

        assert!(stats.is_empty());
        assert_eq!(stats.len(), 0);
        assert_eq!(stats.min(), None);
        assert_eq!(stats.max(), None);
    }
}
