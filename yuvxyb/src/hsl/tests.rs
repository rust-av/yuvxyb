use super::*;

#[test]
fn hsl_new_zero_res() {
    assert!(matches!(
        Hsl::new(vec![], 0, 1),
        Err(CreationError::ZeroResolution)
    ));

    assert!(matches!(
        Hsl::new(vec![], 1, 0),
        Err(CreationError::ZeroResolution)
    ));
}

#[test]
fn hsl_new_res_mismatch() {
    assert!(matches!(
        Hsl::new(vec![], 320, 240),
        Err(CreationError::ResolutionMismatch)
    ));
}

#[test]
fn hsl_new_ok() {
    let data = vec![[1., 1., 1.]; 4];
    assert!(Hsl::new(data, 2, 2).is_ok())
}

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

#[test]
fn test_gray_through_from_trait() {
    let data = vec![[0.5, 0.5, 0.5]; 16];
    let linear_rgb = LinearRgb::new(data, 4, 4).unwrap();

    let hsl = Hsl::from(linear_rgb);
    for [h, s, l] in hsl.data() {
        assert!((h - 0.0).abs() < f32::EPSILON);
        assert!((s - 0.0).abs() < f32::EPSILON);
        assert!((l - 0.5).abs() < f32::EPSILON);
    }
}
