use image::{
    imageops::{flip_horizontal, flip_vertical},
    Rgb, Rgb32FImage, RgbImage,
};
use imageproc::geometric_transformations::{Interpolation, Projection};
use palette::FromColor;
use rand::prelude::*;
use rand_distr::Normal;
use std::f32::consts::PI;

pub fn augment_training_image(image: &RgbImage, rng: &mut impl Rng) -> RgbImage {
    // Flip
    let image = match rng.gen::<f32>() {
        0.0..0.2 => flip_horizontal(image),
        0.2..0.4 => flip_vertical(image),
        0.4..0.6 => flip_horizontal(&flip_vertical(image)),
        _ => image.clone(),
    };

    // Remaining operations need to happen in a linear color space
    // to be correct.
    let mut image = make_linear(&image);

    // Transform
    if rng.gen_bool(0.75) {
        let scale = rng.gen_range(0.9..1.1);
        let tx = rng.gen_range(-20.0..20.0);
        let ty = rng.gen_range(-20.0..20.0);
        let rotation = rng.gen_range(-PI / 8.0..PI / 8.0);
        let projection = Projection::scale(scale, scale)
            .and_then(Projection::translate(tx, ty))
            .and_then(Projection::rotate(rotation));
        image = imageproc::geometric_transformations::warp(
            &image,
            &projection,
            Interpolation::Bicubic,
            Rgb::<f32>([0.0; 3]),
        );
    }

    // Guassian noise
    if rng.gen_bool(0.5) {
        let stdev = Normal::new(0.025f32, 0.005)
            .unwrap()
            .sample(rng)
            .max(0.0001);
        let noise_distr = Normal::new(0.0, stdev).unwrap();
        for val in image.iter_mut() {
            *val = (*val + noise_distr.sample(rng)).clamp(0.0, 1.0);
        }
    }

    make_srgb_eotf(&image)
}

fn make_linear(image: &RgbImage) -> Rgb32FImage {
    Rgb32FImage::from_vec(
        image.width(),
        image.height(),
        image
            .iter()
            .copied()
            .map(|x| {
                palette::SrgbLuma::<f32>::from_format(palette::SrgbLuma::new(x))
                    .into_linear()
                    .luma
            })
            .collect(),
    )
    .unwrap()
}

fn make_srgb_eotf(image: &Rgb32FImage) -> RgbImage {
    RgbImage::from_vec(
        image.width(),
        image.height(),
        image
            .iter()
            .copied()
            .map(|x| {
                palette::SrgbLuma::<u8>::from_format(palette::SrgbLuma::<f32>::from_color(
                    palette::LinLuma::new(x),
                ))
                .luma
            })
            .collect(),
    )
    .unwrap()
}
