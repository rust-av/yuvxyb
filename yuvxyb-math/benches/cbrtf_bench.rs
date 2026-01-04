use criterion::{black_box, criterion_group, criterion_main, Criterion};
use wide::{f32x16, f32x4, f32x8};
use yuvxyb_math::{cbrtf, cbrtf_x16, cbrtf_x4, cbrtf_x8};

const ITERATIONS: usize = 100;

fn bench_cbrtf_scalar(c: &mut Criterion) {
    let values: Vec<f32> = (0..1000).map(|i| (i as f32 + 1.0) * 0.001).collect();
    c.bench_function("cbrtf scalar x1000", |b| {
        b.iter(|| {
            for _ in 0..ITERATIONS {
                for &v in &values {
                    black_box(cbrtf(v));
                }
            }
        })
    });
}

fn bench_cbrtf_x4(c: &mut Criterion) {
    let values: Vec<f32x4> = (0..250)
        .map(|i| {
            let base = (i * 4) as f32 * 0.001;
            f32x4::new([base + 0.001, base + 0.002, base + 0.003, base + 0.004])
        })
        .collect();
    c.bench_function("cbrtf_x4 x1000", |b| {
        b.iter(|| {
            for _ in 0..ITERATIONS {
                for v in &values {
                    black_box(cbrtf_x4(*v));
                }
            }
        })
    });
}

// Call cbrtf_x4 four times to process 16 elements - compare to x16
fn bench_cbrtf_x4_x4(c: &mut Criterion) {
    let values: Vec<[f32x4; 4]> = (0..63)
        .map(|i| {
            let base = (i * 16) as f32 * 0.001;
            [
                f32x4::new([base + 0.001, base + 0.002, base + 0.003, base + 0.004]),
                f32x4::new([base + 0.005, base + 0.006, base + 0.007, base + 0.008]),
                f32x4::new([base + 0.009, base + 0.010, base + 0.011, base + 0.012]),
                f32x4::new([base + 0.013, base + 0.014, base + 0.015, base + 0.016]),
            ]
        })
        .collect();
    c.bench_function("cbrtf_x4 x4 (16 elems) x1008", |b| {
        b.iter(|| {
            for _ in 0..ITERATIONS {
                for [a, b, c, d] in &values {
                    black_box(cbrtf_x4(*a));
                    black_box(cbrtf_x4(*b));
                    black_box(cbrtf_x4(*c));
                    black_box(cbrtf_x4(*d));
                }
            }
        })
    });
}

// Call cbrtf_x8 twice to process 16 elements - compare to x16
fn bench_cbrtf_x8_x2(c: &mut Criterion) {
    let values: Vec<[f32x8; 2]> = (0..63)
        .map(|i| {
            let base = (i * 16) as f32 * 0.001;
            [
                f32x8::new([
                    base + 0.001,
                    base + 0.002,
                    base + 0.003,
                    base + 0.004,
                    base + 0.005,
                    base + 0.006,
                    base + 0.007,
                    base + 0.008,
                ]),
                f32x8::new([
                    base + 0.009,
                    base + 0.010,
                    base + 0.011,
                    base + 0.012,
                    base + 0.013,
                    base + 0.014,
                    base + 0.015,
                    base + 0.016,
                ]),
            ]
        })
        .collect();
    c.bench_function("cbrtf_x8 x2 (16 elems) x1008", |b| {
        b.iter(|| {
            for _ in 0..ITERATIONS {
                for [a, b] in &values {
                    black_box(cbrtf_x8(*a));
                    black_box(cbrtf_x8(*b));
                }
            }
        })
    });
}

fn bench_cbrtf_x8(c: &mut Criterion) {
    let values: Vec<f32x8> = (0..125)
        .map(|i| {
            let base = (i * 8) as f32 * 0.001;
            f32x8::new([
                base + 0.001,
                base + 0.002,
                base + 0.003,
                base + 0.004,
                base + 0.005,
                base + 0.006,
                base + 0.007,
                base + 0.008,
            ])
        })
        .collect();
    c.bench_function("cbrtf_x8 x1000", |b| {
        b.iter(|| {
            for _ in 0..ITERATIONS {
                for v in &values {
                    black_box(cbrtf_x8(*v));
                }
            }
        })
    });
}

fn bench_cbrtf_x16(c: &mut Criterion) {
    let values: Vec<f32x16> = (0..63)
        .map(|i| {
            let base = (i * 16) as f32 * 0.001;
            f32x16::new([
                base + 0.001,
                base + 0.002,
                base + 0.003,
                base + 0.004,
                base + 0.005,
                base + 0.006,
                base + 0.007,
                base + 0.008,
                base + 0.009,
                base + 0.010,
                base + 0.011,
                base + 0.012,
                base + 0.013,
                base + 0.014,
                base + 0.015,
                base + 0.016,
            ])
        })
        .collect();
    c.bench_function("cbrtf_x16 x1008", |b| {
        b.iter(|| {
            for _ in 0..ITERATIONS {
                for v in &values {
                    black_box(cbrtf_x16(*v));
                }
            }
        })
    });
}

criterion_group!(
    benches,
    bench_cbrtf_scalar,
    bench_cbrtf_x4,
    bench_cbrtf_x4_x4,
    bench_cbrtf_x8,
    bench_cbrtf_x8_x2,
    bench_cbrtf_x16
);
criterion_main!(benches);
