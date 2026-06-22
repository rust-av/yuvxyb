use super::*;

const TOLERANCE: f32 = 1e-5;

fn assert_close(actual: [f32; 3], expected: [f32; 3], description: &str) {
    for i in 0..3 {
        assert!(
            (actual[i] - expected[i]).abs() < TOLERANCE,
            "{}: Component {} differs. Expected: {:.6}, Got: {:.6}",
            description,
            i,
            expected[i],
            actual[i]
        );
    }
}

#[test]
fn test_hsl_to_linear_rgb_black() {
    // HSL: Hue=0°, Saturation=0%, Lightness=0% -> Linear RGB: (0, 0, 0)
    let hsl_data = vec![[0.0, 0.0, 0.0]];
    let hsl = Hsl::new(hsl_data, 1, 1).unwrap();
    let linear_rgb = LinearRgb::from(hsl);

    let result = linear_rgb.data()[0];
    let expected = [0.0, 0.0, 0.0];
    assert_close(result, expected, "Black color conversion");
}

#[test]
fn test_hsl_to_linear_rgb_white() {
    // HSL: Hue=0°, Saturation=0%, Lightness=100% -> Linear RGB: (1, 1, 1)
    let hsl_data = vec![[0.0, 0.0, 1.0]];
    let hsl = Hsl::new(hsl_data, 1, 1).unwrap();
    let linear_rgb = LinearRgb::from(hsl);

    let result = linear_rgb.data()[0];
    let expected = [1.0, 1.0, 1.0];
    assert_close(result, expected, "White color conversion");
}

#[test]
fn test_hsl_to_linear_rgb_red() {
    // HSL: Hue=0°, Saturation=100%, Lightness=50% -> Linear RGB: (1, 0, 0)
    let hsl_data = vec![[0.0, 1.0, 0.5]];
    let hsl = Hsl::new(hsl_data, 1, 1).unwrap();
    let linear_rgb = LinearRgb::from(hsl);

    let result = linear_rgb.data()[0];
    let expected = [1.0, 0.0, 0.0];
    assert_close(result, expected, "Pure red color conversion");
}

#[test]
fn test_hsl_to_linear_rgb_green() {
    // HSL: Hue=120°, Saturation=100%, Lightness=50% -> Linear RGB: (0, 1, 0)
    let hsl_data = vec![[120.0, 1.0, 0.5]];
    let hsl = Hsl::new(hsl_data, 1, 1).unwrap();
    let linear_rgb = LinearRgb::from(hsl);

    let result = linear_rgb.data()[0];
    let expected = [0.0, 1.0, 0.0];
    assert_close(result, expected, "Pure green color conversion");
}

#[test]
fn test_hsl_to_linear_rgb_blue() {
    // HSL: Hue=240°, Saturation=100%, Lightness=50% -> Linear RGB: (0, 0, 1)
    let hsl_data = vec![[240.0, 1.0, 0.5]];
    let hsl = Hsl::new(hsl_data, 1, 1).unwrap();
    let linear_rgb = LinearRgb::from(hsl);

    let result = linear_rgb.data()[0];
    let expected = [0.0, 0.0, 1.0];
    assert_close(result, expected, "Pure blue color conversion");
}

#[test]
fn test_hsl_to_linear_rgb_cyan() {
    // HSL: Hue=180°, Saturation=100%, Lightness=50% -> Linear RGB: (0, 1, 1)
    let hsl_data = vec![[180.0, 1.0, 0.5]];
    let hsl = Hsl::new(hsl_data, 1, 1).unwrap();
    let linear_rgb = LinearRgb::from(hsl);

    let result = linear_rgb.data()[0];
    let expected = [0.0, 1.0, 1.0];
    assert_close(result, expected, "Cyan color conversion");
}

#[test]
fn test_hsl_to_linear_rgb_magenta() {
    // HSL: Hue=300°, Saturation=100%, Lightness=50% -> Linear RGB: (1, 0, 1)
    let hsl_data = vec![[300.0, 1.0, 0.5]];
    let hsl = Hsl::new(hsl_data, 1, 1).unwrap();
    let linear_rgb = LinearRgb::from(hsl);

    let result = linear_rgb.data()[0];
    let expected = [1.0, 0.0, 1.0];
    assert_close(result, expected, "Magenta color conversion");
}

#[test]
fn test_hsl_to_linear_rgb_yellow() {
    // HSL: Hue=60°, Saturation=100%, Lightness=50% -> Linear RGB: (1, 1, 0)
    let hsl_data = vec![[60.0, 1.0, 0.5]];
    let hsl = Hsl::new(hsl_data, 1, 1).unwrap();
    let linear_rgb = LinearRgb::from(hsl);

    let result = linear_rgb.data()[0];
    let expected = [1.0, 1.0, 0.0];
    assert_close(result, expected, "Yellow color conversion");
}

#[test]
fn test_hsl_to_linear_rgb_gray() {
    // HSL: Hue=any, Saturation=0%, Lightness=50% -> Linear RGB: (0.5, 0.5, 0.5)
    let hsl_data = vec![[180.0, 0.0, 0.5]];
    let hsl = Hsl::new(hsl_data, 1, 1).unwrap();
    let linear_rgb = LinearRgb::from(hsl);

    let result = linear_rgb.data()[0];
    let expected = [0.5, 0.5, 0.5];
    assert_close(result, expected, "Gray color conversion");
}

#[test]
fn test_hsl_to_linear_rgb_partial_saturation() {
    // HSL: Hue=120°, Saturation=50%, Lightness=50% -> Linear RGB: (0.25, 0.75, 0.25)
    let hsl_data = vec![[120.0, 0.5, 0.5]];
    let hsl = Hsl::new(hsl_data, 1, 1).unwrap();
    let linear_rgb = LinearRgb::from(hsl);

    let result = linear_rgb.data()[0];
    let expected = [0.25, 0.75, 0.25];
    assert_close(result, expected, "Partial saturation color conversion");
}

#[test]
fn test_hsl_to_linear_rgb_low_lightness() {
    // HSL: Hue=0°, Saturation=100%, Lightness=25% -> Linear RGB: (0.5, 0, 0)
    let hsl_data = vec![[0.0, 1.0, 0.25]];
    let hsl = Hsl::new(hsl_data, 1, 1).unwrap();
    let linear_rgb = LinearRgb::from(hsl);

    let result = linear_rgb.data()[0];
    let expected = [0.5, 0.0, 0.0];
    assert_close(result, expected, "Low lightness color conversion");
}

#[test]
fn test_hsl_to_linear_rgb_high_lightness() {
    // HSL: Hue=0°, Saturation=100%, Lightness=75% -> Linear RGB: (1, 0.5, 0.5)
    let hsl_data = vec![[0.0, 1.0, 0.75]];
    let hsl = Hsl::new(hsl_data, 1, 1).unwrap();
    let linear_rgb = LinearRgb::from(hsl);

    let result = linear_rgb.data()[0];
    let expected = [1.0, 0.5, 0.5];
    assert_close(result, expected, "High lightness color conversion");
}

#[test]
fn test_hsl_to_linear_rgb_edge_hue_values() {
    // Test edge values around the 6 sectors of the hue wheel
    let test_cases = vec![
        // Hue at sector boundaries
        (
            [359.0, 1.0, 0.5],
            [1.0, 0.0, 1.0 / 60.0],
            "Hue 359° (near red)",
        ),
        ([1.0, 1.0, 0.5], [1.0, 1.0 / 60.0, 0.0], "Hue 1° (near red)"),
        (
            [59.0, 1.0, 0.5],
            [1.0, 59.0 / 60.0, 0.0],
            "Hue 59° (near yellow)",
        ),
        (
            [61.0, 1.0, 0.5],
            [59.0 / 60.0, 1.0, 0.0],
            "Hue 61° (near yellow)",
        ),
        (
            [119.0, 1.0, 0.5],
            [1.0 / 60.0, 1.0, 0.0],
            "Hue 119° (near green)",
        ),
        (
            [121.0, 1.0, 0.5],
            [0.0, 1.0, 1.0 / 60.0],
            "Hue 121° (near green)",
        ),
        (
            [179.0, 1.0, 0.5],
            [0.0, 1.0, 59.0 / 60.0],
            "Hue 179° (near cyan)",
        ),
        (
            [181.0, 1.0, 0.5],
            [0.0, 59.0 / 60.0, 1.0],
            "Hue 181° (near cyan)",
        ),
        (
            [239.0, 1.0, 0.5],
            [0.0, 1.0 / 60.0, 1.0],
            "Hue 239° (near blue)",
        ),
        (
            [241.0, 1.0, 0.5],
            [1.0 / 60.0, 0.0, 1.0],
            "Hue 241° (near blue)",
        ),
        (
            [299.0, 1.0, 0.5],
            [59.0 / 60.0, 0.0, 1.0],
            "Hue 299° (near magenta)",
        ),
        (
            [301.0, 1.0, 0.5],
            [1.0, 0.0, 59.0 / 60.0],
            "Hue 301° (near magenta)",
        ),
    ];

    for (hsl_values, expected_rgb, description) in test_cases {
        let hsl_data = vec![hsl_values];
        let hsl = Hsl::new(hsl_data, 1, 1).unwrap();
        let linear_rgb = LinearRgb::from(hsl);

        let result = linear_rgb.data()[0];
        assert_close(result, expected_rgb, description);
    }
}

#[test]
fn test_hsl_to_linear_rgb_multiple_pixels() {
    // Test multiple pixels in a single image
    let hsl_data = vec![
        [0.0, 0.0, 0.0],   // Black
        [0.0, 0.0, 1.0],   // White
        [0.0, 1.0, 0.5],   // Red
        [120.0, 1.0, 0.5], // Green
        [240.0, 1.0, 0.5], // Blue
    ];
    let hsl = Hsl::new(hsl_data, 5, 1).unwrap();
    let linear_rgb = LinearRgb::from(hsl);

    let expected_results = [
        [0.0, 0.0, 0.0], // Black
        [1.0, 1.0, 1.0], // White
        [1.0, 0.0, 0.0], // Red
        [0.0, 1.0, 0.0], // Green
        [0.0, 0.0, 1.0], // Blue
    ];

    for (i, expected) in expected_results.iter().enumerate() {
        let result = linear_rgb.data()[i];
        assert_close(result, *expected, &format!("Multi-pixel test, pixel {}", i));
    }

    // Verify dimensions are preserved
    assert_eq!(linear_rgb.width(), 5);
    assert_eq!(linear_rgb.height(), 1);
}

#[test]
fn test_hsl_to_linear_rgb_zero_saturation() {
    // When saturation is 0, all colors should be grayscale regardless of hue
    let test_cases = vec![
        ([0.0, 0.0, 0.3], [0.3, 0.3, 0.3]),
        ([90.0, 0.0, 0.3], [0.3, 0.3, 0.3]),
        ([180.0, 0.0, 0.3], [0.3, 0.3, 0.3]),
        ([270.0, 0.0, 0.3], [0.3, 0.3, 0.3]),
    ];

    for (hsl_values, expected_rgb) in test_cases {
        let hsl_data = vec![hsl_values];
        let hsl = Hsl::new(hsl_data, 1, 1).unwrap();
        let linear_rgb = LinearRgb::from(hsl);

        let result = linear_rgb.data()[0];
        assert_close(
            result,
            expected_rgb,
            &format!("Zero saturation test for hue {}", hsl_values[0]),
        );
    }
}

#[test]
fn test_hsl_to_linear_rgb_boundary_lightness() {
    // Test boundary lightness values
    let test_cases = vec![
        // Lightness = 0 should always produce black
        ([45.0, 0.8, 0.0], [0.0, 0.0, 0.0]),
        ([180.0, 1.0, 0.0], [0.0, 0.0, 0.0]),
        ([300.0, 0.5, 0.0], [0.0, 0.0, 0.0]),
        // Lightness = 1 should always produce white
        ([45.0, 0.8, 1.0], [1.0, 1.0, 1.0]),
        ([180.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
        ([300.0, 0.5, 1.0], [1.0, 1.0, 1.0]),
    ];

    for (hsl_values, expected_rgb) in test_cases {
        let hsl_data = vec![hsl_values];
        let hsl = Hsl::new(hsl_data, 1, 1).unwrap();
        let linear_rgb = LinearRgb::from(hsl);

        let result = linear_rgb.data()[0];
        assert_close(
            result,
            expected_rgb,
            &format!("Boundary lightness test: L={}", hsl_values[2]),
        );
    }
}
