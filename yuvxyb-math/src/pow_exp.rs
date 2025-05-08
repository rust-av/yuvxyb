#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]

use core::f32::consts::LOG2_E;

use crate::multiply_add;

// The following implementation of powf is based on José Fonseca's
// polynomial-based implementation, ported to Rust as scalar code
// so that the compiler can auto-vectorize and otherwise optimize.
// Original: https://jrfonseca.blogspot.com/2008/09/fast-sse2-pow-tables-or-polynomials.html

/// Computes x raised to the power of y.
///
/// This implementation benefits a lot from FMA instructions being
/// available on the target platform. Make sure to enable the relevant
/// CPU feature during compilation.
#[must_use]
pub fn powf(x: f32, y: f32) -> f32 {
    if cfg!(feature = "fastmath") {
        exp2(log2(x) * y)
    } else {
        x.powf(y)
    }
}

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

fn poly5(x: f32, c0: f32, c1: f32, c2: f32, c3: f32, c4: f32, c5: f32) -> f32 {
    multiply_add(x, poly4(x, c1, c2, c3, c4, c5), c0)
}

fn poly4(x: f32, c0: f32, c1: f32, c2: f32, c3: f32, c4: f32) -> f32 {
    multiply_add(x, poly3(x, c1, c2, c3, c4), c0)
}

fn poly3(x: f32, c0: f32, c1: f32, c2: f32, c3: f32) -> f32 {
    multiply_add(x, poly2(x, c1, c2, c3), c0)
}

fn poly2(x: f32, c0: f32, c1: f32, c2: f32) -> f32 {
    multiply_add(x, poly1(x, c1, c2), c0)
}

fn poly1(x: f32, c0: f32, c1: f32) -> f32 {
    multiply_add(x, poly0(x, c1), c0)
}

#[inline]
const fn poly0(_x: f32, c0: f32) -> f32 {
    c0
}

// Based on a C implementation from stackoverflow:
// https://stackoverflow.com/a/10792321/9727602

/// Computes e raised to the power of x.
#[must_use]
pub fn expf(x: f32) -> f32 {
    if cfg!(feature = "fastmath") {
        let t = LOG2_E * x;
        let ft = t.floor();
        let f = t - ft;
        exp2(ft) * exp2(f)
    } else {
        x.exp()
    }
}
