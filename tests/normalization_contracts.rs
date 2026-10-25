use ahuvista_nn::data::{FeatureStats, NormalizationMethod, Normalizer};

#[test]
fn feature_stats_round_trip_values() {
    let stats = FeatureStats::new("age", &[20.0, 30.0, 40.0]);

    assert_eq!(stats.count, 3);
    assert_eq!(stats.min, 20.0);
    assert_eq!(stats.max, 40.0);

    let normalized = stats.normalize(30.0);

    assert!(normalized.abs() < 1e-6);
    assert!((stats.denormalize(normalized) - 30.0).abs() < 1e-5);
}

#[test]
fn normalizer_applies_selected_strategy_and_leaves_unknown_values_unchanged() {
    let mut normalizer = Normalizer::new();

    normalizer.method = NormalizationMethod::MinMax;
    normalizer.feature_order.push("age".into());
    normalizer
        .feature_stats
        .insert("age".into(), FeatureStats::new("age", &[20.0, 40.0]));

    assert!((normalizer.normalize("age", 30.0) - 0.5).abs() < 1e-6);
    assert_eq!(normalizer.normalize("missing", 3.0), 3.0);
}
