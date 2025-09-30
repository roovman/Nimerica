use crate::traits::HasTensorCore;
use crate::utils::logging::{debug, trace, error};

/// Errors that can occur when constructing or using tensors.
#[derive(Debug)]
pub enum TensorError {
    ShapeMismatch { expected: usize, got: usize },
}

/// Dense, owned tensor with fixed rank `R`.
#[derive(Clone, Debug)]
pub struct Tensor<T, const R: usize> {
    pub(crate) dims: [usize; R],
    pub(crate) strides: [usize; R],
    pub(crate) data: Vec<T>,
}

impl<T, const R: usize> Tensor<T, R> {
    /// Try to create a new tensor with given dimensions and backing data.
    /// Returns `Err` if data length doesn’t match the product of dimensions.
    pub fn new(dims: [usize; R], data: Vec<T>) -> Result<Self, TensorError> {
        let expected: usize = dims.iter().product();
        let got = data.len();

        if got != expected {
            error!(
                "Tensor::new failed: shape {:?} expects {}, got {} elements",
                dims, expected, got
            );
            return Err(TensorError::ShapeMismatch { expected, got });
        }

        // Compute row-major strides
        let mut strides = [0; R];
        let mut acc = 1;
        for (s, d) in strides.iter_mut().rev().zip(dims.iter().rev()) {
            *s = acc;
            acc *= *d;
        }

        debug!("Tensor::new dims={:?}, strides={:?}, len={}", dims, strides, got);

        Ok(Self { dims, strides, data })
    }

    /// Construct a tensor filled with default values (like `zeros`).
    pub fn defaulted(dims: [usize; R]) -> Self
    where
        T: Default + Clone,
    {
        let total: usize = dims.iter().product();
        debug!("Tensor::defaulted dims={:?}, total elements={}", dims, total);
        let data = vec![T::default(); total];

        // Safe unwrap: data length always matches
        Self::new(dims, data).unwrap()
    }

    /// Return the underlying data buffer.
    pub fn into_inner(self) -> Vec<T> {
        trace!("Tensor::into_inner called, len={}", self.data.len());
        self.data
    }
}

impl<T, const R: usize> HasTensorCore<T> for Tensor<T, R> {
    fn dims(&self) -> &[usize] {
        trace!("Tensor::dims -> {:?}", self.dims);
        &self.dims
    }

    fn strides(&self) -> &[usize] {
        trace!("Tensor::strides -> {:?}", self.strides);
        &self.strides
    }

    fn data(&self) -> &[T] {
        trace!("Tensor::data -> &[T; {}]", self.data.len());
        &self.data
    }

    fn data_mut(&mut self) -> &mut [T] {
        trace!("Tensor::data_mut -> &mut [T; {}]", self.data.len());
        &mut self.data
    }
}
