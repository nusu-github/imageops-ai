//! Performance benchmarks for imageops-ai
//!
//! This benchmark suite measures the performance of all major operations
//! to ensure they meet performance expectations and to track regressions.

use criterion::*;
use image::{Luma, Rgb, Rgba};
use imageops_ai::{
    ApplyAlphaMaskExt, ForegroundEstimationExt, Image, InterAreaResizeExt, PaddingExt, Position,
    PremultiplyAlphaAndDropExt, PremultiplyAlphaAndKeepExt,
};
use itertools::iproduct;
use std::hint::black_box;

/// Helper function to create a test RGB image with specific dimensions
fn create_rgb_image(width: u32, height: u32) -> Image<Rgb<u8>> {
    let mut image: Image<Rgb<u8>> = Image::new(width, height);

    // Fill with realistic pattern (gradient + content)
    iproduct!(0..height, 0..width).for_each(|(y, x)| {
        let r = ((x * 255) / width) as u8;
        let g = ((y * 255) / height) as u8;
        let b = ((x + y) * 255 / (width + height)) as u8;
        image.put_pixel(x, y, Rgb([r, g, b]));
    });

    image
}

/// Helper function to create a test RGBA image with specific dimensions
fn create_rgba_image(width: u32, height: u32) -> Image<Rgba<u8>> {
    let mut image: Image<Rgba<u8>> = Image::new(width, height);

    // Fill with semi-transparent pattern
    iproduct!(0..height, 0..width).for_each(|(y, x)| {
        let r = ((x * 255) / width) as u8;
        let g = ((y * 255) / height) as u8;
        let b = ((x + y) * 255 / (width + height)) as u8;
        let a = if (x + y) % 3 == 0 { 128 } else { 255 }; // Varying alpha
        image.put_pixel(x, y, Rgba([r, g, b, a]));
    });

    image
}

/// Helper function to create an alpha mask with realistic patterns
fn create_alpha_mask(width: u32, height: u32) -> Image<Luma<u8>> {
    let mut mask: Image<Luma<u8>> = Image::new(width, height);

    let center_x = width as f32 / 2.0;
    let center_y = height as f32 / 2.0;
    let max_radius = (width.min(height) as f32) / 2.0;

    // Create circular gradient mask
    iproduct!(0..height, 0..width).for_each(|(y, x)| {
        let distance = (x as f32 - center_x).hypot(y as f32 - center_y);
        let alpha = if distance <= max_radius {
            (255.0 * (1.0 - distance / max_radius)) as u8
        } else {
            0
        };
        mask.put_pixel(x, y, Luma([alpha]));
    });

    mask
}

/// Benchmark alpha premultiplication across different image sizes
fn bench_alpha_premultiply(c: &mut Criterion) {
    let sizes = vec![
        (100, 100),   // Small
        (500, 500),   // Medium
        (1000, 1000), // Large
        (1920, 1080), // HD
    ];

    let mut group = c.benchmark_group("alpha_premultiply");
    group.sample_size(10);

    for (width, height) in sizes {
        let pixels = width * height;
        group.throughput(Throughput::Elements(pixels as u64));

        let image = create_rgba_image(width, height);

        group.bench_with_input(
            BenchmarkId::new("premultiply_alpha", format!("{}x{}", width, height)),
            &image,
            |b, img| b.iter(|| black_box(img.clone().premultiply_alpha_and_keep().unwrap())),
        );
    }

    group.finish();
}

/// Benchmark alpha mask application across different image sizes
fn bench_alpha_mask_application(c: &mut Criterion) {
    let sizes = vec![
        (100, 100),   // Small
        (500, 500),   // Medium
        (1000, 1000), // Large
        (1920, 1080), // HD
    ];

    let mut group = c.benchmark_group("alpha_mask_application");
    group.sample_size(10);

    for (width, height) in sizes {
        let pixels = width * height;
        group.throughput(Throughput::Elements(pixels as u64));

        let image = create_rgb_image(width, height);
        let mask = create_alpha_mask(width, height);

        group.bench_with_input(
            BenchmarkId::new("apply_alpha_mask", format!("{}x{}", width, height)),
            &(image, mask),
            |b, (img, alpha_mask)| {
                b.iter(|| black_box(img.clone().apply_alpha_mask(alpha_mask).unwrap()))
            },
        );
    }

    group.finish();
}

/// Benchmark foreground estimation (Blur-Fusion) across different parameters
fn bench_foreground_estimation(c: &mut Criterion) {
    let sizes = vec![
        (200, 200), // Small for detailed measurement
        (500, 500), // Medium
        (800, 800), // Large
    ];

    let radii = vec![7, 31, 91]; // Different blur radii

    let mut group = c.benchmark_group("foreground_estimation");
    group.sample_size(10); // Fewer samples for expensive operations

    for (width, height) in sizes {
        for radius in &radii {
            let pixels = width * height;
            group.throughput(Throughput::Elements(pixels as u64));

            let image = create_rgb_image(width, height);
            let mask = create_alpha_mask(width, height);

            group.bench_with_input(
                BenchmarkId::new(
                    "estimate_foreground",
                    format!("{}x{}_r{}", width, height, radius),
                ),
                &(image, mask, *radius),
                |b, (img, alpha_mask, r)| {
                    b.iter(|| {
                        black_box(
                            img.clone()
                                .estimate_foreground_colors(alpha_mask, *r)
                                .unwrap(),
                        )
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark padding operations across different sizes and positions
fn bench_padding_operations(c: &mut Criterion) {
    let base_sizes = vec![(100, 100), (500, 300), (800, 600)];

    let positions = vec![Position::Center, Position::TopLeft, Position::BottomRight];

    let mut group = c.benchmark_group("padding_operations");
    group.sample_size(10);

    for (width, height) in base_sizes {
        for position in &positions {
            let pad_width = width + 200;
            let pad_height = height + 200;
            let pixels = pad_width * pad_height;
            group.throughput(Throughput::Elements(pixels as u64));

            let image = create_rgb_image(width, height);

            group.bench_with_input(
                BenchmarkId::new(
                    "add_padding",
                    format!(
                        "{}x{}_to_{}x{}_{:?}",
                        width, height, pad_width, pad_height, position
                    ),
                ),
                &(image, pad_width, pad_height, *position),
                |b, (img, pw, ph, pos)| {
                    b.iter(|| {
                        black_box(
                            img.clone()
                                .add_padding((*pw, *ph), *pos, Rgb([255, 255, 255]))
                                .unwrap(),
                        )
                    })
                },
            );
        }
    }

    group.finish();
}

/// Benchmark square padding operations
fn bench_square_padding(c: &mut Criterion) {
    let rectangular_sizes = vec![
        (100, 50),    // 2:1 ratio
        (300, 200),   // 3:2 ratio
        (800, 600),   // 4:3 ratio
        (1920, 1080), // 16:9 ratio
    ];

    let mut group = c.benchmark_group("square_padding");
    group.sample_size(10);

    for (width, height) in rectangular_sizes {
        let max_dim = width.max(height);
        let pixels = max_dim * max_dim;
        group.throughput(Throughput::Elements(pixels as u64));

        let image = create_rgb_image(width, height);

        group.bench_with_input(
            BenchmarkId::new("add_padding_square", format!("{}x{}", width, height)),
            &image,
            |b, img| b.iter(|| black_box(img.clone().to_square(Rgb([0, 0, 0])).unwrap().0)),
        );
    }

    group.finish();
}

/// Benchmark complex workflows that combine multiple operations
fn bench_complex_workflows(c: &mut Criterion) {
    let sizes = vec![(300, 200), (800, 600)];

    let mut group = c.benchmark_group("complex_workflows");
    group.sample_size(10); // Fewer samples for complex operations

    for (width, height) in sizes {
        let pixels = width * height;
        group.throughput(Throughput::Elements(pixels as u64));

        let image = create_rgb_image(width, height);
        let mask = create_alpha_mask(width, height);

        // Workflow: Foreground estimation → Alpha mask → Premultiplication → Square padding
        group.bench_with_input(
            BenchmarkId::new("full_workflow", format!("{}x{}", width, height)),
            &(image, mask),
            |b, (img, alpha_mask)| {
                b.iter(|| {
                    let foreground = img
                        .clone()
                        .estimate_foreground_colors(alpha_mask, 31)
                        .unwrap();

                    let with_alpha = foreground.apply_alpha_mask(alpha_mask).unwrap();

                    let premultiplied = with_alpha.premultiply_alpha_and_drop().unwrap();

                    let (final_result, _) = premultiplied.to_square(Rgb([255, 255, 255])).unwrap();

                    black_box(final_result)
                })
            },
        );
    }

    group.finish();
}

/// Benchmark memory efficiency by testing with large images
fn bench_memory_efficiency(c: &mut Criterion) {
    let large_sizes = vec![
        (2000, 2000), // 4MP
        (3000, 2000), // 6MP
    ];

    let mut group = c.benchmark_group("memory_efficiency");
    group.sample_size(5); // Very few samples for memory-intensive tests

    for (width, height) in large_sizes {
        let pixels = width * height;
        group.throughput(Throughput::Elements(pixels as u64));

        // Test the most memory-efficient operations
        let image = create_rgb_image(width, height);
        let mask = create_alpha_mask(width, height);

        group.bench_with_input(
            BenchmarkId::new("large_alpha_mask", format!("{}x{}", width, height)),
            &(image, mask),
            |b, (img, alpha_mask)| {
                b.iter(|| black_box(img.clone().apply_alpha_mask(alpha_mask).unwrap()))
            },
        );
    }

    group.finish();
}

/// Benchmark INTER_AREA resizing with integer scale factors
fn bench_inter_area_integer_scale(c: &mut Criterion) {
    let source_sizes = vec![
        (400, 400),   // Small
        (800, 800),   // Medium
        (1600, 1600), // Large
        (2000, 1000), // Rectangular
    ];

    // Integer scale factors for optimal performance path
    let scale_factors = vec![2, 3, 4, 8];

    let mut group = c.benchmark_group("inter_area_integer_scale");
    group.sample_size(10);

    for (src_width, src_height) in source_sizes {
        for scale_factor in &scale_factors {
            let dst_width = src_width / scale_factor;
            let dst_height = src_height / scale_factor;

            if dst_width == 0 || dst_height == 0 {
                continue;
            }

            let pixels = src_width * src_height;
            group.throughput(Throughput::Elements(pixels as u64));

            let image = create_rgb_image(src_width, src_height);

            group.bench_with_input(
                BenchmarkId::new(
                    "resize_area_integer",
                    format!(
                        "{}x{}_to_{}x{}_scale{}",
                        src_width, src_height, dst_width, dst_height, scale_factor
                    ),
                ),
                &(image, dst_width, dst_height),
                |b, (img, w, h)| b.iter(|| black_box(img.clone().resize_area(*w, *h).unwrap())),
            );
        }
    }

    group.finish();
}

criterion_group!(
    benches,
    // Individual benchmarks
    bench_alpha_premultiply,
    bench_alpha_mask_application,
    bench_foreground_estimation,
    bench_padding_operations,
    bench_square_padding,
    bench_inter_area_integer_scale,
    // Complex workflows and memory efficiency
    bench_complex_workflows,
    bench_memory_efficiency,
);
criterion_main!(benches);
