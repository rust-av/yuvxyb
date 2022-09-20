use num_traits::{FromPrimitive, Num, ToPrimitive};

pub trait Pixel: Num + ToPrimitive + FromPrimitive + Clone + Copy {}

impl Pixel for u8 {}
impl Pixel for u16 {}
impl Pixel for f32 {}
