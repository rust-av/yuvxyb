/// Computes (a * b) + c, leveraging FMA if available
#[inline]
#[allow(clippy::suboptimal_flops)]
#[must_use]
pub fn multiply_add(a: f32, b: f32, c: f32) -> f32 {
    if cfg!(target_feature = "fma") {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}
