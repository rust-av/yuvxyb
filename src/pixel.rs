use num_traits::{AsPrimitive, FromPrimitive, Num, PrimInt, ToPrimitive};

pub trait YuvPixel:
    Num + PrimInt + ToPrimitive + FromPrimitive + AsPrimitive<f32> + Clone + Copy
{
}

impl YuvPixel for u8 {}
impl YuvPixel for u16 {}
