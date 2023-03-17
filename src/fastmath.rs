#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]

use core::f32;

// The following cbrtf implementation is a port of FreeBSDs cbrtf function
// found in <root>/lib/msun/src/s_cbrtf.c, modified to remove some edge case
// handling if the argument x is a non-normal float.
//
// The description and copyright notice found below apply only to the function
// cbrtf directly below it.

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

/// Computes the cube root of x.
///
/// The argument must be normal (not NaN, +/-INF or subnormal).
/// This is required for optimization purposes.
#[cfg(feature = "fastmath")]
pub fn cbrtf(x: f32) -> f32 {
    // B1 = (127-127.0/3-0.03306235651)*2**23
    const B1: u32 = 709_958_130;

    let mut r: f64;
    let mut t: f64;
    let mut ui: u32 = x.to_bits();
    let mut hx: u32 = ui & 0x7FFF_FFFF;

    hx = hx / 3 + B1;
    ui &= 0x8000_0000;
    ui |= hx;

    // First step Newton iteration (solving t*t-x/t == 0) to 16 bits.  In
    // double precision so that its terms can be arranged for efficiency
    // without causing overflow or underflow.
    t = f64::from(f32::from_bits(ui));
    r = t * t * t;
    t = t * (f64::from(x) + f64::from(x) + r) / (f64::from(x) + r + r);

    // Second step Newton iteration to 47 bits.  In double precision for
    // efficiency and accuracy.
    r = t * t * t;
    t = t * (f64::from(x) + f64::from(x) + r) / (f64::from(x) + r + r);

    // rounding to 24 bits is perfect in round-to-nearest mode
    t as f32
}

// The following implementation of powf is based on José Fonseca's
// polynomial-based implementation, ported to Rust as scalar code
// so that the compiler can auto-vectorize and otherwise optimize.
// Original: https://jrfonseca.blogspot.com/2008/09/fast-sse2-pow-tables-or-polynomials.html

/// Computes x raised to the power of y.
///
/// This implementation benefits a lot from FMA instructions being
/// available on the target platform. Make sure to enable the relevant
/// CPU feature during compilation.
#[cfg(feature = "fastmath")]
pub fn powf(x: f32, y: f32) -> f32 {
    exp2(log2(x) * y)
}

#[cfg(feature = "fastmath")]
fn exp2(x: f32) -> f32 {
    let x = x.clamp(-126.99999, 129.0);

    // SAFETY: Value is clamped
    let ipart: i32 = unsafe { (x - 0.5).to_int_unchecked() };
    let fpart = x - ipart as f32;

    let expi = f32::from_bits(((ipart + 127_i32) << 23_i32) as u32);
    let expf = poly5(
        fpart,
        9.999_999_4E-1,
        6.931_531E-1,
        2.401_536_1E-1,
        5.582_631_8E-2,
        8.989_34E-3,
        1.877_576_7E-3,
    );

    expi * expf
}

#[cfg(feature = "fastmath")]
fn log2(x: f32) -> f32 {
    let expmask = 0x7F80_0000_i32;
    let mantmask = 0x007F_FFFF_i32;

    let one_bits = 1_f32.to_bits() as i32;
    let x_bits = x.to_bits() as i32;
    let exp = (((x_bits & expmask) >> 23_i32) - 127_i32) as f32;

    let mant = f32::from_bits(((x_bits & mantmask) | one_bits) as u32);

    let polynomial = poly5(
        mant,
        3.115_79,
        -3.324_199,
        2.598_845_2,
        -1.231_530_3,
        3.182_133_7E-1,
        -3.443_600_6E-2,
    );
    let polynomial = polynomial * (mant - 1.0);

    polynomial + exp
}

#[cfg(feature = "fastmath")]
#[inline(always)]
fn poly5(x: f32, c0: f32, c1: f32, c2: f32, c3: f32, c4: f32, c5: f32) -> f32 {
    x.mul_add(poly4(x, c1, c2, c3, c4, c5), c0)
}

#[cfg(feature = "fastmath")]
#[inline(always)]
fn poly4(x: f32, c0: f32, c1: f32, c2: f32, c3: f32, c4: f32) -> f32 {
    x.mul_add(poly3(x, c1, c2, c3, c4), c0)
}

#[cfg(feature = "fastmath")]
#[inline(always)]
fn poly3(x: f32, c0: f32, c1: f32, c2: f32, c3: f32) -> f32 {
    x.mul_add(poly2(x, c1, c2, c3), c0)
}

#[cfg(feature = "fastmath")]
#[inline(always)]
fn poly2(x: f32, c0: f32, c1: f32, c2: f32) -> f32 {
    x.mul_add(poly1(x, c1, c2), c0)
}

#[cfg(feature = "fastmath")]
#[inline(always)]
fn poly1(x: f32, c0: f32, c1: f32) -> f32 {
    x.mul_add(poly0(x, c1), c0)
}

#[cfg(feature = "fastmath")]
#[inline(always)]
const fn poly0(_x: f32, c0: f32) -> f32 {
    c0
}

// Based on a C implementation from stackoverflow:
// https://stackoverflow.com/a/10792321/9727602

/// Computes e raised to the power of x.
#[cfg(feature = "fastmath")]
pub fn expf(x: f32) -> f32 {
    let t = 1.442_695_041 * x; // log2(e) * x
    let ft = t.floor();
    let f = t - ft;
    exp2(ft) * exp2(f)
}

/// Computes the cube root of x.
#[cfg(not(feature = "fastmath"))]
#[inline(always)]
pub fn cbrtf(x: f32) -> f32 {
    x.cbrt()
}

/// Computes x raised to the power of y.
#[cfg(not(feature = "fastmath"))]
#[inline(always)]
pub fn powf(x: f32, y: f32) -> f32 {
    x.powf(y)
}

/// Computes e raised to the power of x.
#[cfg(not(feature = "fastmath"))]
#[inline(always)]
pub fn expf(x: f32) -> f32 {
    x.exp()
}
