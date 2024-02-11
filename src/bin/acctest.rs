use std::path::PathBuf;
use std::ops::{Range, Neg};

use anyhow::Result;
use yuvxyb::{LinearRgb, TransferCharacteristic, ColorPrimaries, Rgb};

fn main() {
    compare("tank_srgb.png", "tank_rgb.png", "tank_comparison.png");
    compare("chimera_srgb.png", "chimera_rgb.png", "chimera_comparison.png");
    compare("chimera_srgb.png", "tank_rgb.png", "useless_comparison.png");
}

fn compare(source_path: &str, expected_path: &str, graph_path: &str) {
    let source = struct_from_file(source_path, |d, w, h| Rgb::new(d, w, h, TransferCharacteristic::SRGB, ColorPrimaries::BT709)).unwrap();
    let expected = struct_from_file(expected_path, LinearRgb::new).unwrap();

    let actual = LinearRgb::try_from(source).unwrap();

    let deltas = compare_data(expected.data(), actual.data());
    draw_graph(deltas, ["R".to_string(), "G".to_string(), "B".to_string()], graph_path);
}

fn struct_from_file<T>(path: &str, create_fn: fn(Vec<[f32; 3]>, usize, usize) -> Result<T>) -> Result<T> {
    let base_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test_data");
    let img = image::open(base_path.join(path)).unwrap();

    let w = img.width() as usize;
    let h = img.height() as usize;
    let data: Vec<[f32; 3]> = img
        .into_rgb32f()
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();

    create_fn(data, w, h)
}

// return the delta for each channel/plane
fn compare_data(expected: &[[f32; 3]], actual: &[[f32; 3]]) -> [Vec<f32>; 3] {
    let mut r = vec![0f32; expected.len()];
    let mut g = vec![0f32; expected.len()];
    let mut b = vec![0f32; expected.len()];

    for (i, (pix_e, pix_a)) in expected.iter().zip(actual).enumerate() {
        r[i] = pix_a[0] - pix_e[0];
        g[i] = pix_a[1] - pix_e[1];
        b[i] = pix_a[2] - pix_e[2];
    }

    [r, g, b]
}

fn draw_graph(deltas: [Vec<f32>; 3], plane_labels: [String; 3], path: &str) {
    use plotters::prelude::*;
    use plotters::data::fitting_range;

    let root = BitMapBackend::new(path, (640, 360)).into_drawing_area();
    root.fill(&WHITE).unwrap();
    let root = root.margin(5, 5, 5, 20); // leave some space on the right

    let dataset: Vec<(String, Quartiles)> = deltas.iter().zip(&plane_labels).map(|(v, l)| (l.clone(), Quartiles::new(&v))).collect();
    let range = fitting_range(deltas.iter().flatten());
    let range = refit_range(range);

    let mut cc = ChartBuilder::on(&root)
        .x_label_area_size(50)
        .y_label_area_size(25)
        .caption("Conversion errors per color plane", ("sans-serif", 20))
        .build_cartesian_2d(
            // -5e-3f32..5e-3f32,
            range,
            plane_labels.into_segmented(),
        )
        .unwrap();

    cc.configure_mesh()
        .x_desc("Error")
        .x_label_style(("sans-serif", 20))
        .y_labels(plane_labels.len())
        .y_label_style(("sans-serif", 20))
        .y_label_formatter(&|v| match v {
            SegmentValue::Exact(l) => l.to_string(),
            SegmentValue::CenterOf(l) => l.to_string(),
            SegmentValue::Last => String::new(),
        })
        .light_line_style(&WHITE)
        .draw()
        .unwrap();

    cc.draw_series(dataset.iter().map(|(l, q)| {
            Boxplot::new_horizontal(SegmentValue::CenterOf(l), q)
                .width(25)
                .whisker_width(0.5)
        }))
        .unwrap();

    root.present().unwrap();
}

// make "symmetric"/centered around zero
fn refit_range<T: Neg<Output = T> + PartialOrd + Copy>(r: Range<T>) -> Range<T> {
    if r.end > -r.start {
        -r.end..r.end
    } else {
        r.start..-r.start
    }
}
