use image::flat::SampleLayout;
use image::{ImageBuffer, Pixel, Primitive};
use imageproc::definitions::Image;
use ndarray::{Array3, ArrayView3, ShapeBuilder};

/// Converts an `ImageBuffer` to a 3D ndarray (height, width, channels).
pub fn image_to_array3<P, S>(image: &Image<P>) -> Array3<S>
where
    P: Pixel<Subpixel = S>,
    S: Primitive,
{
    let SampleLayout {
        channels,
        channel_stride,
        height,
        height_stride,
        width,
        width_stride,
    } = image.sample_layout();
    let shape = (channels as usize, height as usize, width as usize);
    let strides = (channel_stride, height_stride, width_stride);
    let arr = ArrayView3::from_shape(shape.strides(strides), image)
        .expect("Failed to create ArrayView3 from ImageBuffer");

    arr.permuted_axes((1, 2, 0)).to_owned()
}

/// Converts a 3D ndarray (height, width, channels) to an `ImageBuffer`.
pub fn array3_to_image<P, S>(arr: &ArrayView3<S>) -> Image<P>
where
    P: Pixel<Subpixel = S>,
    S: Primitive,
{
    let (height, width, _channels) = arr.dim();
    let (raw_vec, _offset) = arr.into_owned().into_raw_vec_and_offset();
    ImageBuffer::from_raw(width as u32, height as u32, raw_vec)
        .expect("Failed to create ImageBuffer from raw data")
}
