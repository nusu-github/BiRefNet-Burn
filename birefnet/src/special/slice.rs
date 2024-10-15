use std::ops::Range;

pub struct Slice {
    start: Option<isize>,
    end: Option<isize>,
}

impl Slice {
    pub fn new(start: Option<isize>, end: Option<isize>) -> Self {
        Slice { start, end }
    }

    pub fn to_range(&self, len: usize) -> Range<usize> {
        let len_isize = len as isize;

        let start = match self.start {
            Some(s) if s < 0 => (len_isize + s).max(0),
            Some(s) => s.max(0).min(len_isize),
            None => 0,
        } as usize;

        let end = match self.end {
            Some(e) if e < 0 => (len_isize + e).max(0),
            Some(e) => e.max(0).min(len_isize),
            None => len_isize,
        } as usize;

        start..end
    }

    pub fn slice_length(&self, len: usize) -> usize {
        let range = self.to_range(len);
        if range.end > range.start {
            range.end - range.start
        } else {
            0
        }
    }
}
