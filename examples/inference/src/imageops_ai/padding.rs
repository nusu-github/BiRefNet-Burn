use anyhow::Result;
use image::{imageops, ImageBuffer, Pixel};
use num_traits::AsPrimitive;

#[allow(dead_code)]
pub enum Position {
    Top,
    Bottom,
    Left,
    Right,
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
    Center,
}

pub fn to_position(
    size: (u32, u32),
    pad_size: (u32, u32),
    position: &Position,
) -> Result<(i64, i64)> {
    let (width, height) = size;
    let (pad_width, pad_height) = pad_size;

    anyhow::ensure!(
        pad_width >= width,
        "padding width must be greater than image width"
    );
    anyhow::ensure!(
        pad_height >= height,
        "padding height must be greater than image height"
    );

    let (x, y) = match position {
        Position::Top => ((pad_width - width) / 2, 0),
        Position::Bottom => ((pad_width - width) / 2, pad_height - height),
        Position::Left => (0, (pad_height - height) / 2),
        Position::Right => (pad_width - width, (pad_height - height) / 2),
        Position::TopLeft => (0, 0),
        Position::TopRight => (pad_width - width, 0),
        Position::BottomLeft => (0, pad_height - height),
        Position::BottomRight => (pad_width - width, pad_height - height),
        Position::Center => ((pad_width - width) / 2, (pad_height - height) / 2),
    };

    Ok((x.as_(), y.as_()))
}

pub trait Padding<P: Pixel> {
    fn padding(
        self,
        pad_size: (u32, u32),
        position: &Position,
        color: P,
    ) -> (ImageBuffer<P, Vec<P::Subpixel>>, (u32, u32));

    fn padding_square(self, color: P) -> (ImageBuffer<P, Vec<P::Subpixel>>, (u32, u32));

    fn to_position(&self, pad_size: (u32, u32), position: &Position) -> Result<(i64, i64)>;

    fn to_position_square(&self) -> Result<((i64, i64), (u32, u32))>;
}

impl<P: Pixel> Padding<P> for ImageBuffer<P, Vec<P::Subpixel>> {
    fn padding(self, pad_size: (u32, u32), position: &Position, color: P) -> (Self, (u32, u32)) {
        self.to_position(pad_size, position).map_or_else(
            |_| (self.clone(), (0, 0)),
            |(x, y)| {
                let (pad_width, pad_height) = pad_size;
                let mut canvas = Self::from_pixel(pad_width, pad_height, color);
                imageops::overlay(&mut canvas, &self, x, y);
                (canvas, (x as u32, y as u32))
            },
        )
    }

    fn padding_square(self, color: P) -> (Self, (u32, u32)) {
        let (_, pad_size) = self.to_position_square().unwrap();
        self.padding(pad_size, &Position::Center, color)
    }

    fn to_position(&self, pad_size: (u32, u32), position: &Position) -> Result<(i64, i64)> {
        let (width, height) = self.dimensions();

        to_position((width, height), pad_size, position)
    }

    fn to_position_square(&self) -> Result<((i64, i64), (u32, u32))> {
        let (width, height) = self.dimensions();

        let pad_size = if width > height {
            (width, width)
        } else {
            (height, height)
        };

        self.to_position(pad_size, &Position::Center)
            .map(|(x, y)| ((x, y), pad_size))
    }
}
