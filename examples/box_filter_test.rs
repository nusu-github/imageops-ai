use std::env;

use image::DynamicImage;
use imageops_ai::BoxFilterExt;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    if args.len() != 4 {
        eprintln!("Usage: {} <input_image> <radius> <output_image>", args[0]);
        eprintln!("Example: {} input.png 5 output.png", args[0]);
        std::process::exit(1);
    }

    let input_path = &args[1];
    let radius: u32 = args[2].parse().map_err(|_| "Invalid radius")?;
    let output_path = &args[3];

    // 入力画像を読み込み
    let img = image::open(input_path)?;

    // 画像のタイプに応じて処理
    let processed_img = match img {
        DynamicImage::ImageRgb8(rgb_img) => {
            println!(
                "Processing RGB8 image: {}x{}",
                rgb_img.width(),
                rgb_img.height()
            );
            let result = rgb_img.box_filter(radius)?;
            DynamicImage::ImageRgb8(result)
        }
        DynamicImage::ImageRgba8(rgba_img) => {
            println!(
                "Processing RGBA8 image: {}x{}",
                rgba_img.width(),
                rgba_img.height()
            );
            let result = rgba_img.box_filter(radius)?;
            DynamicImage::ImageRgba8(result)
        }
        DynamicImage::ImageLuma8(luma_img) => {
            println!(
                "Processing Luma8 image: {}x{}",
                luma_img.width(),
                luma_img.height()
            );
            let result = luma_img.box_filter(radius)?;
            DynamicImage::ImageLuma8(result)
        }
        DynamicImage::ImageLumaA8(luma_a_img) => {
            println!(
                "Processing LumaA8 image: {}x{}",
                luma_a_img.width(),
                luma_a_img.height()
            );
            let result = luma_a_img.box_filter(radius)?;
            DynamicImage::ImageLumaA8(result)
        }
        DynamicImage::ImageRgb16(rgb_img) => {
            println!(
                "Processing RGB16 image: {}x{}",
                rgb_img.width(),
                rgb_img.height()
            );
            let result = rgb_img.box_filter(radius)?;
            DynamicImage::ImageRgb16(result)
        }
        DynamicImage::ImageRgba16(rgba_img) => {
            println!(
                "Processing RGBA16 image: {}x{}",
                rgba_img.width(),
                rgba_img.height()
            );
            let result = rgba_img.box_filter(radius)?;
            DynamicImage::ImageRgba16(result)
        }
        DynamicImage::ImageLuma16(luma_img) => {
            println!(
                "Processing Luma16 image: {}x{}",
                luma_img.width(),
                luma_img.height()
            );
            let result = luma_img.box_filter(radius)?;
            DynamicImage::ImageLuma16(result)
        }
        DynamicImage::ImageLumaA16(luma_a_img) => {
            println!(
                "Processing LumaA16 image: {}x{}",
                luma_a_img.width(),
                luma_a_img.height()
            );
            let result = luma_a_img.box_filter(radius)?;
            DynamicImage::ImageLumaA16(result)
        }
        DynamicImage::ImageRgb32F(rgb_img) => {
            println!(
                "Processing RGB32F image: {}x{}",
                rgb_img.width(),
                rgb_img.height()
            );
            let result = rgb_img.box_filter(radius)?;
            DynamicImage::ImageRgb32F(result)
        }
        DynamicImage::ImageRgba32F(rgba_img) => {
            println!(
                "Processing RGBA32F image: {}x{}",
                rgba_img.width(),
                rgba_img.height()
            );
            let result = rgba_img.box_filter(radius)?;
            DynamicImage::ImageRgba32F(result)
        }
        _ => {
            eprintln!("Unsupported image format");
            std::process::exit(1);
        }
    };

    // 結果を保存
    processed_img.save(output_path)?;

    println!("Box filter (radius={}) applied successfully!", radius);
    println!("Output saved to: {}", output_path);

    Ok(())
}
