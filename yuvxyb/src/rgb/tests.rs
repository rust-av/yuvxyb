use super::*;

use ColorPrimaries as CP;
use TransferCharacteristic as TC;

#[test]
fn rgb_new_zero_res() {
    assert!(matches!(
        Rgb::new(vec![], 0, 1, TC::SRGB, CP::BT709),
        Err(CreationError::ZeroResolution)
    ));

    assert!(matches!(
        Rgb::new(vec![], 1, 0, TC::SRGB, CP::BT709),
        Err(CreationError::ZeroResolution)
    ));
}

#[test]
fn rgb_new_res_mismatch() {
    assert!(matches!(
        Rgb::new(vec![], 320, 240, TC::SRGB, CP::BT709),
        Err(CreationError::ResolutionMismatch)
    ));
}

#[test]
fn rgb_new_ok() {
    let data = vec![[1., 1., 1.]; 4];
    assert!(Rgb::new(data, 2, 2, TC::BT1886, CP::BT709).is_ok());
}

#[test]
fn rgb_new_fix_unspecified() {
    let data = vec![[1., 1., 1.]; 4];
    let rgb = Rgb::new(data, 2, 2, TC::Unspecified, CP::Unspecified).unwrap();

    assert_ne!(rgb.transfer, TC::Unspecified);
    assert_ne!(rgb.primaries, CP::Unspecified);
}
