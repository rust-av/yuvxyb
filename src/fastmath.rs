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

/// Cube root (f32)
///
/// Computes the cube root of the argument.
/// The argument must be normal (not NaN, +/-INF or subnormal).
/// This is required for optimization purposes.
pub fn cbrtf(x: f32) -> f32 {
    debug_assert!(x.is_normal());
    let x64 = f64::from(x);

    let x_bits = x.to_bits();
    let sign = x_bits & 0x8000_0000;

    // rough cbrt to 5 bits
    let hx = sign | ((x_bits ^ sign) / 3 + B1);

    // First step Newton iteration (solving t*t-x/t == 0) to 16 bits.
    // In double precision so that its terms can be arranged for
    // efficiency without causing overflow or underflow.
    let t = f64::from(f32::from_bits(hx));
    let r = t * t * t;
    let t = t * (x64 + x64 + r) / (x64 + r + r);

    // Second step Newton iteration to 47 bits.
    // In double precision for efficiency and accuracy.
    let r = t * t * t;
    let t = t * (x64 + x64 + r) / (x64 + r + r);

    // rounding to 24 bits is perfect in round-to-nearest mode
    t as f32
}

// Credit to https://jrfonseca.blogspot.com/2008/09/fast-sse2-pow-tables-or-polynomials.html
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
    poly0(x, c1).mul_add(x, f32x4::from(c0))
}

#[inline(always)]
fn poly2(x: f32x4, c0: f32, c1: f32, c2: f32) -> f32x4 {
    poly1(x, c1, c2).mul_add(x, f32x4::from(c0))
}

#[inline(always)]
fn poly3(x: f32x4, c0: f32, c1: f32, c2: f32, c3: f32) -> f32x4 {
    poly2(x, c1, c2, c3).mul_add(x, f32x4::from(c0))
}

#[inline(always)]
fn poly4(x: f32x4, c0: f32, c1: f32, c2: f32, c3: f32, c4: f32) -> f32x4 {
    poly3(x, c1, c2, c3, c4).mul_add(x, f32x4::from(c0))
}

#[inline(always)]
fn poly5(x: f32x4, c0: f32, c1: f32, c2: f32, c3: f32, c4: f32, c5: f32) -> f32x4 {
    poly4(x, c1, c2, c3, c4, c5).mul_add(x, f32x4::from(c0))
}
