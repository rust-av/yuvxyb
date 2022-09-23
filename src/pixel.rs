use std::fmt::Debug;

use num_traits::{AsPrimitive, FromPrimitive, Num, PrimInt, ToPrimitive, Zero};

pub trait YuvPixel:
    Num + PrimInt + Zero + FromPrimitive + ToPrimitive + AsPrimitive<f32> + Clone + Copy + Debug
{
}

impl YuvPixel for u8 {}
impl YuvPixel for u16 {}
