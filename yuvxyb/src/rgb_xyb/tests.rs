use std::{
    fs,
    path::{Path, PathBuf},
};

use av_data::pixel::{ColorPrimaries, TransferCharacteristic};

use crate::{Rgb, Xyb};

fn parse_xyb_txt(path: &Path) -> Vec<[f32; 3]> {
    let input = fs::read_to_string(path).unwrap();
    input
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .map(|l| {
            let (x, l) = l.split_once(' ').unwrap();
            let (y, b) = l.split_once(' ').unwrap();
            [x.parse().unwrap(), y.parse().unwrap(), b.parse().unwrap()]
        })
        .collect()
}

#[test]
fn rgb_to_xyb_correct() {
    let source_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join("tank_srgb.png");
    let expected_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join("tank_xyb.txt");
    let source = image::open(source_path).unwrap();
    let source_data = source
        .to_rgb32f()
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect::<Vec<_>>();
    let source = Rgb::new(
        source_data,
        1448,
        1080,
        TransferCharacteristic::SRGB,
        ColorPrimaries::BT709,
    )
    .unwrap();
    let expected_data = parse_xyb_txt(&expected_path);

    let result = Xyb::try_from(source).unwrap();
    for (exp, res) in expected_data.into_iter().zip(result.data()) {
        assert!(
            (exp[0] - res[0]).abs() < 0.0005,
            "Difference in X channel: Expected {:.4}, got {:.4}",
            exp[0],
            res[0]
        );
        assert!(
            (exp[1] - res[1]).abs() < 0.0005,
            "Difference in Y channel: Expected {:.4}, got {:.4}",
            exp[1],
            res[1]
        );
        assert!(
            (exp[2] - res[2]).abs() < 0.0005,
            "Difference in B channel: Expected {:.4}, got {:.4}",
            exp[2],
            res[2]
        );
    }
}

#[test]
fn xyb_to_rgb_correct() {
    let source_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join("tank_xyb.txt");
    let expected_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test_data")
        .join("tank_srgb.png");
    let source_data = parse_xyb_txt(&source_path);
    let source = Xyb::new(source_data, 1448, 1080).unwrap();
    let expected = image::open(expected_path).unwrap();

    // Fixing this would result in worse code, and this
    // needless collect() doesn't really matter in tests
    #[allow(clippy::needless_collect)]
    let expected_data = expected
        .to_rgb32f()
        .chunks_exact(3)
        .map(|chunk| [chunk[0], chunk[1], chunk[2]])
        .collect::<Vec<_>>();

    let result =
        Rgb::try_from((source, TransferCharacteristic::SRGB, ColorPrimaries::BT709)).unwrap();
    for (exp, res) in expected_data.into_iter().zip(result.data()) {
        assert!(
            (exp[0] - res[0]).abs() < 0.0005,
            "Difference in R channel: Expected {:.4}, got {:.4}",
            exp[0],
            res[0]
        );
        assert!(
            (exp[1] - res[1]).abs() < 0.0005,
            "Difference in G channel: Expected {:.4}, got {:.4}",
            exp[1],
            res[1]
        );
        assert!(
            (exp[2] - res[2]).abs() < 0.0005,
            "Difference in B channel: Expected {:.4}, got {:.4}",
            exp[2],
            res[2]
        );
    }
}
