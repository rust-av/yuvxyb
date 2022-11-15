/* origin: FreeBSD /usr/src/lib/msun/src/s_cbrtf.c */
/*
 * Conversion to float by Ian Lance Taylor, Cygnus Support, ian@cygnus.com.
 * Debugged and optimized by Bruce D. Evans.
 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */
/* cbrtf(x)
 * Return cube root of x
 */

use core::f32;

const B1: u32 = 709_958_130; /* B1 = (127-127.0/3-0.03306235651)*2**23 */

/// Cube root (f32)
///
/// Computes the cube root of the argument.
/// The argument must be normal (not NaN, +/-INF or subnormal).
/// This is required for optimization purposes.
pub fn cbrtf(x: f32) -> f32 {
    let mut r: f64;
    let mut t: f64;
    let mut ui: u32 = x.to_bits();
    let mut hx: u32 = ui & 0x7FFF_FFFF;

    hx = hx / 3 + B1;
    ui &= 0x8000_0000;
    ui |= hx;

    /*
     * First step Newton iteration (solving t*t-x/t == 0) to 16 bits.  In
     * double precision so that its terms can be arranged for efficiency
     * without causing overflow or underflow.
     */
    t = f64::from(f32::from_bits(ui));
    r = t * t * t;
    t = t * (f64::from(x) + f64::from(x) + r) / (f64::from(x) + r + r);

    /*
     * Second step Newton iteration to 47 bits.  In double precision for
     * efficiency and accuracy.
     */
    r = t * t * t;
    t = t * (f64::from(x) + f64::from(x) + r) / (f64::from(x) + r + r);

    /* rounding to 24 bits is perfect in round-to-nearest mode */
    t as f32
}
