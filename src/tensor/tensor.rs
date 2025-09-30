use super::traits::HasTensorCore;

#[derive(Clone, Debug)]
pub struct Tensor<T, const R: usize> {
    pub(crate) dims: [usize; R],
    pub(crate) strides: [usize; R],
    pub(crate) data: Vec<T>,
}

impl<T, const R: usize> Tensor<T, R> {
    pub fn new(dims: [usize; R], data: Vec<T>) -> Self {
        let expected: usize = dims.iter().product();
        assert_eq!(data.len(), expected, "Data length does not match shape");

        // row-major strides
        let mut strides = [0; R];
        let mut acc = 1;
        for (s, d) in strides.iter_mut().rev().zip(dims.iter().rev()) {
            *s = acc;
            acc *= *d;
        }


        Self { dims, strides, data }
    }
}


impl<T, const R: usize> HasTensorCore<T> for Tensor<T, R> {
    fn dims(&self) -> &[usize] { &self.dims }
    fn strides(&self) -> &[usize] { &self.strides }
    fn data(&self) -> &[T] { &self.data }
    fn data_mut(&mut self) -> &mut [T] { &mut self.data }
}
