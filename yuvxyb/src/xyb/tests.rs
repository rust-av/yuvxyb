use super::*;

#[test]
fn xyb_new_zero_res() {
    assert!(matches!(
        Xyb::new(vec![], 0, 1),
        Err(CreationError::ZeroResolution)
    ));

    assert!(matches!(
        Xyb::new(vec![], 1, 0),
        Err(CreationError::ZeroResolution)
    ));
}

#[test]
fn xyb_new_res_mismatch() {
    assert!(matches!(
        Xyb::new(vec![], 320, 240),
        Err(CreationError::ResolutionMismatch)
    ));
}

#[test]
fn xyb_new_ok() {
    let data = vec![[1., 1., 1.]; 4];
    assert!(Xyb::new(data, 2, 2).is_ok())
}
