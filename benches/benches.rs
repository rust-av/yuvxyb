use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use yuvxyb::*;

fn make_yuv_8b(
    ss: (u8, u8),
    full_range: bool,
    mc: MatrixCoefficients,
    tc: TransferCharacteristic,
    cp: ColorPrimaries,
) -> Yuv<u8> {
    let y_dims = (320usize, 240usize);
    let uv_dims = (y_dims.0 >> ss.0, y_dims.1 >> ss.1);
    let mut data: Frame<u8> = Frame {
        planes: [
            Plane::new(y_dims.0, y_dims.1, 0, 0, 0, 0),
            Plane::new(
                uv_dims.0,
                uv_dims.1,
                usize::from(ss.0),
                usize::from(ss.1),
                0,
                0,
            ),
            Plane::new(
                uv_dims.0,
                uv_dims.1,
                usize::from(ss.0),
                usize::from(ss.1),
                0,
                0,
            ),
        ],
    };
    let mut rng = rand::thread_rng();
    for (i, plane) in data.planes.iter_mut().enumerate() {
        for val in plane.data_origin_mut().iter_mut() {
            *val = rng.gen_range(if full_range {
                0..=255
            } else if i == 0 {
                16..=235
            } else {
                16..=240
            });
        }
    }
    Yuv::new(
        data,
        YuvConfig {
            bit_depth: 8,
            subsampling_x: ss.0,
            subsampling_y: ss.1,
            full_range,
            matrix_coefficients: mc,
            transfer_characteristics: tc,
            color_primaries: cp,
        },
    )
    .unwrap()
}

fn make_yuv_10b(
    ss: (u8, u8),
    full_range: bool,
    mc: MatrixCoefficients,
    tc: TransferCharacteristic,
    cp: ColorPrimaries,
) -> Yuv<u16> {
    let y_dims = (320usize, 240usize);
    let uv_dims = (y_dims.0 >> ss.0, y_dims.1 >> ss.1);
    let mut data: Frame<u16> = Frame {
        planes: [
            Plane::new(y_dims.0, y_dims.1, 0, 0, 0, 0),
            Plane::new(
                uv_dims.0,
                uv_dims.1,
                usize::from(ss.0),
                usize::from(ss.1),
                0,
                0,
            ),
            Plane::new(
                uv_dims.0,
                uv_dims.1,
                usize::from(ss.0),
                usize::from(ss.1),
                0,
                0,
            ),
        ],
    };
    let mut rng = rand::thread_rng();
    for (i, plane) in data.planes.iter_mut().enumerate() {
        for val in plane.data_origin_mut().iter_mut() {
            *val = rng.gen_range(if full_range {
                0..=1023
            } else if i == 0 {
                64..=940
            } else {
                64..=960
            });
        }
    }
    Yuv::new(
        data,
        YuvConfig {
            bit_depth: 10,
            subsampling_x: ss.0,
            subsampling_y: ss.1,
            full_range,
            matrix_coefficients: mc,
            transfer_characteristics: tc,
            color_primaries: cp,
        },
    )
    .unwrap()
}

fn bench_yuv_8b_444_full_to_xyb(c: &mut Criterion) {
    c.bench_function("yuv 8-bit 4:4:4 full to xyb", |b| {
        let input = make_yuv_8b(
            (0, 0),
            true,
            MatrixCoefficients::BT709,
            TransferCharacteristic::BT1886,
            ColorPrimaries::BT709,
        );
        b.iter(|| Xyb::try_from(black_box(&input)).unwrap())
    });
}

fn bench_yuv_8b_420_full_to_xyb(c: &mut Criterion) {
    c.bench_function("yuv 8-bit 4:2:0 full to xyb", |b| {
        let input = make_yuv_8b(
            (1, 1),
            true,
            MatrixCoefficients::BT709,
            TransferCharacteristic::BT1886,
            ColorPrimaries::BT709,
        );
        b.iter(|| Xyb::try_from(black_box(&input)).unwrap())
    });
}

fn bench_yuv_8b_420_limited_to_xyb(c: &mut Criterion) {
    c.bench_function("yuv 8-bit 4:2:0 limited to xyb", |b| {
        let input = make_yuv_8b(
            (1, 1),
            false,
            MatrixCoefficients::BT709,
            TransferCharacteristic::BT1886,
            ColorPrimaries::BT709,
        );
        b.iter(|| Xyb::try_from(black_box(&input)).unwrap())
    });
}

fn bench_yuv_10b_444_full_to_xyb(c: &mut Criterion) {
    c.bench_function("yuv 10-bit 4:4:4 full to xyb", |b| {
        let input = make_yuv_10b(
            (0, 0),
            true,
            MatrixCoefficients::BT2020NonConstantLuminance,
            TransferCharacteristic::PerceptualQuantizer,
            ColorPrimaries::BT2020,
        );
        b.iter(|| Xyb::try_from(black_box(&input)).unwrap())
    });
}

fn bench_yuv_10b_420_full_to_xyb(c: &mut Criterion) {
    c.bench_function("yuv 10-bit 4:2:0 full to xyb", |b| {
        let input = make_yuv_10b(
            (1, 1),
            true,
            MatrixCoefficients::BT2020NonConstantLuminance,
            TransferCharacteristic::PerceptualQuantizer,
            ColorPrimaries::BT2020,
        );
        b.iter(|| Xyb::try_from(black_box(&input)).unwrap())
    });
}

fn bench_yuv_10b_420_limited_to_xyb(c: &mut Criterion) {
    c.bench_function("yuv 10-bit 4:2:0 limited to xyb", |b| {
        let input = make_yuv_10b(
            (1, 1),
            false,
            MatrixCoefficients::BT2020NonConstantLuminance,
            TransferCharacteristic::PerceptualQuantizer,
            ColorPrimaries::BT2020,
        );
        b.iter(|| Xyb::try_from(black_box(&input)).unwrap())
    });
}

fn bench_hybrid_log_gamma(c: &mut Criterion) {
    c.bench_function("rgb to lrgb via hybrid log-gamma system", |b| {
        let input = {
            let yuv = make_yuv_10b(
                (1, 1),
                false,
                MatrixCoefficients::BT2020NonConstantLuminance,
                TransferCharacteristic::HybridLogGamma,
                ColorPrimaries::BT2020,
            );
            Rgb::try_from(&yuv).unwrap()
        };

        b.iter(|| LinearRgb::try_from(black_box(input.clone())).unwrap())
    });
}

criterion_group!(
    benches,
    bench_yuv_8b_444_full_to_xyb,
    bench_yuv_8b_420_full_to_xyb,
    bench_yuv_8b_420_limited_to_xyb,
    bench_yuv_10b_444_full_to_xyb,
    bench_yuv_10b_420_full_to_xyb,
    bench_yuv_10b_420_limited_to_xyb,
    bench_hybrid_log_gamma,
);
criterion_main!(benches);
