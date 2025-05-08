use super::*;

#[test]
fn test_black() {
    let result = lrgb_to_hsl([0.0, 0.0, 0.0]);
    assert!((result[0] - 0.0).abs() < f32::EPSILON); // H
    assert!((result[1] - 0.0).abs() < f32::EPSILON); // S
    assert!((result[2] - 0.0).abs() < f32::EPSILON); // L
}

#[test]
fn test_white() {
    let result = lrgb_to_hsl([1.0, 1.0, 1.0]);
    assert!((result[0] - 0.0).abs() < f32::EPSILON); // H
    assert!((result[1] - 0.0).abs() < f32::EPSILON); // S
    assert!((result[2] - 1.0).abs() < f32::EPSILON); // L
}

#[test]
fn test_red() {
    let result = lrgb_to_hsl([1.0, 0.0, 0.0]);
    assert!((result[0] - 0.0).abs() < f32::EPSILON); // H
    assert!((result[1] - 1.0).abs() < f32::EPSILON); // S
    assert!((result[2] - 0.5).abs() < f32::EPSILON); // L
}

#[test]
fn test_green() {
    let result = lrgb_to_hsl([0.0, 1.0, 0.0]);
    assert!((result[0] - 120.0).abs() < f32::EPSILON); // H
    assert!((result[1] - 1.0).abs() < f32::EPSILON); // S
    assert!((result[2] - 0.5).abs() < f32::EPSILON); // L
}

#[test]
fn test_blue() {
    let result = lrgb_to_hsl([0.0, 0.0, 1.0]);
    assert!((result[0] - 240.0).abs() < f32::EPSILON); // H
    assert!((result[1] - 1.0).abs() < f32::EPSILON); // S
    assert!((result[2] - 0.5).abs() < f32::EPSILON); // L
}

#[test]
fn test_gray() {
    let result = lrgb_to_hsl([0.5, 0.5, 0.5]);
    assert!((result[0] - 0.0).abs() < f32::EPSILON); // H
    assert!((result[1] - 0.0).abs() < f32::EPSILON); // S
    assert!((result[2] - 0.5).abs() < f32::EPSILON); // L
}
