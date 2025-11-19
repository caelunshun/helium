use image::{
    imageops::{flip_horizontal, flip_vertical},
    Rgb, Rgb32FImage, RgbImage,
};
use imageproc::geometric_transformations::{Interpolation, Projection};
use rand::prelude::*;
use rand_distr::Normal;
use std::f32::consts::PI;

#[profiling::function]
pub fn augment_training_image(image: &Rgb32FImage, rng: &mut impl Rng) -> Rgb32FImage {
    // Flip
    let mut image = match rng.gen::<f32>() {
        0.0..0.2 => flip_horizontal(image),
        0.2..0.4 => flip_vertical(image),
        0.4..0.6 => flip_horizontal(&flip_vertical(image)),
        _ => image.clone(),
    };

    // Gaussian noise
    if rng.gen_bool(0.4) {
        let stdev = Normal::new(0.025f32, 0.005)
            .unwrap()
            .sample(rng)
            .max(0.0001);
        let noise_distr = Normal::new(0.0, stdev).unwrap();
        for val in image.iter_mut() {
            *val = (*val + noise_distr.sample(rng)).clamp(0.0, 1.0);
        }
    }

    // Brightness / contrast adjustment
    let brightness = Normal::new(0.0f32, 0.2f32)
        .unwrap()
        .sample(rng)
        .clamp(-0.4, 0.4);
    let contrast = Normal::new(1.0f32, 0.2f32)
        .unwrap()
        .sample(rng)
        .clamp(0.6, 1.4);
    for val in image.iter_mut() {
        *val = ((*val * contrast) + brightness).clamp(0.0, 1.0);
    }

    // Transform
    if rng.gen_bool(0.9) {
        let scale_x = rng.gen_range(0.8f32..1.6);
        let scale_y = Normal::new(scale_x, 0.05)
            .unwrap()
            .sample(rng)
            .clamp(0.9, 1.1);

        let max_translate = image.width() as f32 / 8.0;

        let tx = rng.gen_range(-max_translate..=max_translate);
        let ty = rng.gen_range(-max_translate..=max_translate);
        let rotation = Normal::new(0.0, PI / 24.0).unwrap().sample(rng);
        let projection = Projection::translate(
            -(image.width() as f32) / 2.0,
            -(image.height() as f32) / 2.0,
        )
        .and_then(Projection::rotate(rotation))
        .and_then(Projection::translate(
            image.width() as f32 / 2.0,
            image.height() as f32 / 2.0,
        ))
        .and_then(Projection::scale(scale_x, scale_y))
        .and_then(Projection::translate(tx, ty));

        image = imageproc::geometric_transformations::warp(
            &image,
            &projection,
            Interpolation::Bicubic,
            Rgb::<f32>([0.0; 3]),
        );
    }

    image
}

pub fn make_srgb32f(image: &RgbImage) -> Rgb32FImage {
    Rgb32FImage::from_vec(
        image.width(),
        image.height(),
        image
            .iter()
            .copied()
            .map(|x| fast_srgb8::srgb8_to_f32(x))
            .collect(),
    )
    .unwrap()
}

pub fn make_srgb8(image: &Rgb32FImage) -> RgbImage {
    RgbImage::from_vec(
        image.width(),
        image.height(),
        image
            .as_raw()
            .chunks(4) // requires pixel count to be multiple of 4
            .flat_map(|chunk| fast_srgb8::f32x4_to_srgb8(chunk.try_into().unwrap()))
            .collect::<Vec<_>>(),
    )
    .unwrap()
}
