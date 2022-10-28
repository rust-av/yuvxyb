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
use std::mem::transmute;

use wide::{f32x4, i32x4};

const B1: u32 = 709_958_130; /* B1 = (127-127.0/3-0.03306235651)*2**23 */
const B2: u32 = 642_849_266; /* B2 = (127-127.0/3-24/3-0.03306235651)*2**23 */

/// Cube root (f32)
///
/// Computes the cube root of the argument.
pub fn cbrtf(x: f32) -> f32 {
    let x1p24 = f32::from_bits(0x4b80_0000); // 0x1p24f === 2 ^ 24

    let mut r: f64;
    let mut t: f64;
    let mut ui: u32 = x.to_bits();
    let mut hx: u32 = ui & 0x7fff_ffff;

    if hx >= 0x7f80_0000 {
        /* cbrt(NaN,INF) is itself */
        return x + x;
    }

    /* rough cbrt to 5 bits */
    if hx < 0x0080_0000 {
        /* zero or subnormal? */
        if hx == 0 {
            return x; /* cbrt(+-0) is itself */
        }
        ui = (x * x1p24).to_bits();
        hx = ui & 0x7fff_ffff;
        hx = hx / 3 + B2;
    } else {
        hx = hx / 3 + B1;
    }
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

pub fn powf_wide(x: f32x4, pow: f32) -> f32x4 {
    exp2_wide(log2_wide(x) * pow)
}

fn exp2_wide(x: f32x4) -> f32x4 {
    // Uses a polynomial degree of 3 to fast approximate exp2
    let x = x
        .fast_min(f32x4::from(129.00000f32))
        .fast_max(f32x4::from(-126.99999f32));
    let ipart = (x - 0.5f32).to_i32x4_round_fast();
    let fpart = x - ipart.to_f32x4();
    // SAFETY: Types are the same size
    let expipart: f32x4 = unsafe { transmute((ipart + i32x4::from(127i32)) << 23i32) };
    let expfpart = poly3(
        fpart,
        9.999_252e-1_f32,
        6.958_335_6e-1_f32,
        2.260_671_6e-1_f32,
        7.802_452e-2_f32,
    );

    expipart * expfpart
}

#[allow(clippy::many_single_char_names)]
fn log2_wide(x: f32x4) -> f32x4 {
    // Uses a polynomial degree of 5 to fast approximate log2
    let exp = i32x4::from(0x7F80_0000_i32);
    let mant = i32x4::from(0x007F_FFFF_i32);
    // SAFETY: Types are the same size
    let i: i32x4 = unsafe { transmute(x) };
    let e = (((i & exp) >> 23i32) - 127i32).to_f32x4();
    // SAFETY: Types are the same size
    let m_temp: f32x4 = unsafe { transmute(i & mant) };
    let m = m_temp | f32x4::ONE;
    let p = poly5(
        m,
        3.115_79_f32,
        -3.324_199_f32,
        2.598_845_2_f32,
        -1.231_530_3_f32,
        3.182_133_7e-1_f32,
        -3.443_600_6e-2_f32,
    );
    let p = p * (m - f32x4::ONE);
    p + e
}

#[inline(always)]
fn poly0(_x: f32x4, c0: f32) -> f32x4 {
    f32x4::from(c0)
}

#[inline(always)]
fn poly1(x: f32x4, c0: f32, c1: f32) -> f32x4 {
    poly0(x, c1) * x + c0
}

#[inline(always)]
fn poly2(x: f32x4, c0: f32, c1: f32, c2: f32) -> f32x4 {
    poly1(x, c1, c2) * x + c0
}

#[inline(always)]
fn poly3(x: f32x4, c0: f32, c1: f32, c2: f32, c3: f32) -> f32x4 {
    poly2(x, c1, c2, c3) * x + c0
}

#[inline(always)]
fn poly4(x: f32x4, c0: f32, c1: f32, c2: f32, c3: f32, c4: f32) -> f32x4 {
    poly3(x, c1, c2, c3, c4) * x + c0
}

#[inline(always)]
fn poly5(x: f32x4, c0: f32, c1: f32, c2: f32, c3: f32, c4: f32, c5: f32) -> f32x4 {
    poly4(x, c1, c2, c3, c4, c5) * x + c0
}
