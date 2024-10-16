use crate::imageops_ai::Image;
use image::{GenericImage, GenericImageView, ImageBuffer, Luma, Rgb, Rgb32FImage};

pub trait BoxFilter {
    type Output;
    fn box_filter(&self, x_radius: u32, y_radius: u32) -> Self::Output;
}

impl BoxFilter for Image<Rgb<f32>> {
    type Output = Self;

    fn box_filter(&self, x_radius: u32, y_radius: u32) -> Self::Output {
        fn row_running_sum(image: &Rgb32FImage, row: u32, buffer: &mut [[f32; 3]], padding: u32) {
            let (width, _height) = image.dimensions();
            let (width, padding) = (width as usize, padding as usize);

            let row_data = &(**image)[width * row as usize * 3..][..width * 3];
            let first = [row_data[0], row_data[1], row_data[2]];
            let last = [
                row_data[width * 3 - 3],
                row_data[width * 3 - 2],
                row_data[width * 3 - 1],
            ];

            let mut sum = [0.0, 0.0, 0.0];

            for b in &mut buffer[..padding] {
                for i in 0..3 {
                    sum[i] += first[i];
                    b[i] = sum[i];
                }
            }
            for (b, chunk) in buffer[padding..].iter_mut().zip(row_data.chunks(3)) {
                for i in 0..3 {
                    sum[i] += chunk[i];
                    b[i] = sum[i];
                }
            }
            for b in &mut buffer[padding + width..] {
                for i in 0..3 {
                    sum[i] += last[i];
                    b[i] = sum[i];
                }
            }
        }

        fn column_running_sum(
            image: &Rgb32FImage,
            column: u32,
            buffer: &mut [[f32; 3]],
            padding: u32,
        ) {
            let (_width, height) = image.dimensions();

            let first = image.get_pixel(column, 0).0;
            let last = image.get_pixel(column, height - 1).0;

            let mut sum = [0.0, 0.0, 0.0];

            for b in &mut buffer[..padding as usize] {
                for i in 0..3 {
                    sum[i] += first[i];
                    b[i] = sum[i];
                }
            }
            unsafe {
                for y in 0..height {
                    let pixel = image.unsafe_get_pixel(column, y).0;
                    for i in 0..3 {
                        sum[i] += pixel[i];
                        buffer.get_unchecked_mut(y as usize + padding as usize)[i] = sum[i];
                    }
                }
            }
            for b in &mut buffer[padding as usize + height as usize..] {
                for i in 0..3 {
                    sum[i] += last[i];
                    b[i] = sum[i];
                }
            }
        }

        pub fn box_filter(image: &Rgb32FImage, x_radius: u32, y_radius: u32) -> Image<Rgb<f32>> {
            let (width, height) = image.dimensions();
            let mut out = ImageBuffer::new(width, height);
            if width == 0 || height == 0 {
                return out;
            }

            let kernel_width = (2 * x_radius + 1) as f32;
            let kernel_height = (2 * y_radius + 1) as f32;

            let mut row_buffer = vec![[0.0; 3]; (width + 2 * x_radius) as usize];
            for y in 0..height {
                row_running_sum(image, y, &mut row_buffer, x_radius);
                let val = row_buffer[(2 * x_radius) as usize].map(|v| v / kernel_width);
                unsafe {
                    debug_assert!(out.in_bounds(0, y));
                    out.unsafe_put_pixel(0, y, Rgb(val));
                };
                for x in 1..width {
                    let u = (x + 2 * x_radius) as usize;
                    let l = (x - 1) as usize;
                    let val =
                        [0, 1, 2].map(|i| (row_buffer[u][i] - row_buffer[l][i]) / kernel_width);
                    unsafe {
                        debug_assert!(out.in_bounds(x, y));
                        out.unsafe_put_pixel(x, y, Rgb(val));
                    }
                }
            }

            let mut col_buffer = vec![[0.0; 3]; (height + 2 * y_radius) as usize];
            for x in 0..width {
                column_running_sum(&out, x, &mut col_buffer, y_radius);
                let val = col_buffer[(2 * y_radius) as usize].map(|v| v / kernel_height);
                unsafe {
                    debug_assert!(out.in_bounds(x, 0));
                    out.unsafe_put_pixel(x, 0, Rgb(val));
                };
                for y in 1..height {
                    let u = (y + 2 * y_radius) as usize;
                    let l = (y - 1) as usize;
                    let val =
                        [0, 1, 2].map(|i| (col_buffer[u][i] - col_buffer[l][i]) / kernel_height);
                    unsafe {
                        debug_assert!(out.in_bounds(x, y));
                        out.unsafe_put_pixel(x, y, Rgb(val));
                    }
                }
            }

            out
        }

        box_filter(self, x_radius, y_radius)
    }
}

impl BoxFilter for Image<Luma<f32>> {
    type Output = Self;

    fn box_filter(&self, x_radius: u32, y_radius: u32) -> Self::Output {
        fn row_running_sum(image: &Image<Luma<f32>>, row: u32, buffer: &mut [f32], padding: u32) {
            let (width, _height) = image.dimensions();
            let (width, padding) = (width as usize, padding as usize);

            let row_data = &(**image)[width * row as usize..][..width];
            let first = row_data[0];
            let last = row_data[width - 1];

            let mut sum = 0.0;

            for b in &mut buffer[..padding] {
                sum += first;
                *b = sum;
            }
            for (b, p) in buffer[padding..].iter_mut().zip(row_data) {
                sum += *p;
                *b = sum;
            }
            for b in &mut buffer[padding + width..] {
                sum += last;
                *b = sum;
            }
        }

        fn column_running_sum(
            image: &Image<Luma<f32>>,
            column: u32,
            buffer: &mut [f32],
            padding: u32,
        ) {
            let (_width, height) = image.dimensions();

            let first = image.get_pixel(column, 0)[0];
            let last = image.get_pixel(column, height - 1)[0];

            let mut sum = 0.0;

            for b in &mut buffer[..padding as usize] {
                sum += first;
                *b = sum;
            }
            unsafe {
                for y in 0..height {
                    sum += image.unsafe_get_pixel(column, y)[0];
                    *buffer.get_unchecked_mut(y as usize + padding as usize) = sum;
                }
            }
            for b in &mut buffer[padding as usize + height as usize..] {
                sum += last;
                *b = sum;
            }
        }

        pub fn box_filter(
            image: &Image<Luma<f32>>,
            x_radius: u32,
            y_radius: u32,
        ) -> Image<Luma<f32>> {
            let (width, height) = image.dimensions();
            let mut out = ImageBuffer::new(width, height);
            if width == 0 || height == 0 {
                return out;
            }

            let kernel_width = (2 * x_radius + 1) as f32;
            let kernel_height = (2 * y_radius + 1) as f32;

            let mut row_buffer = vec![0.0; (width + 2 * x_radius) as usize];
            for y in 0..height {
                row_running_sum(image, y, &mut row_buffer, x_radius);
                let val = row_buffer[(2 * x_radius) as usize] / kernel_width;
                unsafe {
                    debug_assert!(out.in_bounds(0, y));
                    out.unsafe_put_pixel(0, y, Luma([val]));
                };
                for x in 1..width {
                    let u = (x + 2 * x_radius) as usize;
                    let l = (x - 1) as usize;
                    let val = (row_buffer[u] - row_buffer[l]) / kernel_width;
                    unsafe {
                        debug_assert!(out.in_bounds(x, y));
                        out.unsafe_put_pixel(x, y, Luma([val]));
                    }
                }
            }

            let mut col_buffer = vec![0.0; (height + 2 * y_radius) as usize];
            for x in 0..width {
                column_running_sum(&out, x, &mut col_buffer, y_radius);
                let val = col_buffer[(2 * y_radius) as usize] / kernel_height;
                unsafe {
                    debug_assert!(out.in_bounds(x, 0));
                    out.unsafe_put_pixel(x, 0, Luma([val]));
                };
                for y in 1..height {
                    let u = (y + 2 * y_radius) as usize;
                    let l = (y - 1) as usize;
                    let val = (col_buffer[u] - col_buffer[l]) / kernel_height;
                    unsafe {
                        debug_assert!(out.in_bounds(x, y));
                        out.unsafe_put_pixel(x, y, Luma([val]));
                    }
                }
            }

            out
        }

        box_filter(self, x_radius, y_radius)
    }
}
