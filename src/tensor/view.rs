use crate::traits::HasTensorCore;
use crate::utils::logging::{debug, trace, error};
use std::fmt;
use std::marker::PhantomData;

/// Immutable view into a tensor.
pub struct TensorView<'a, T, const R: usize> {
    pub(crate) len: usize,
    pub(crate) dims: [usize; R],
    pub(crate) strides: [usize; R],
    pub(crate) data: *const T, // raw pointer to data start
    _marker: PhantomData<&'a T>,
}

impl<'a, T, const R: usize> TensorView<'a, T, R> {
    pub fn new(dims: [usize; R], strides: [usize; R], data: &'a [T]) -> Self {
        let len: usize = dims.iter().product();
        debug!(
            "TensorView::new dims={:?}, strides={:?}, len={}",
            dims, strides, len
        );
        Self {
            len,
            dims,
            strides,
            data: data.as_ptr(),
            _marker: PhantomData,
        }
    }
}

impl<'a, T, const R: usize> HasTensorCore<T> for TensorView<'a, T, R> {
    fn dims(&self) -> &[usize] {
        trace!("TensorView::dims -> {:?}", self.dims);
        &self.dims
    }
    fn strides(&self) -> &[usize] {
        trace!("TensorView::strides -> {:?}", self.strides);
        &self.strides
    }
    fn data(&self) -> &[T] {
        trace!("TensorView::data -> &[T; {}]", self.len);
        // SAFETY: lifetime ties this slice to `'a`
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }
    fn data_mut(&mut self) -> &mut [T] {
        error!("TensorView::data_mut called on immutable view");
        panic!("TensorView is immutable; use TensorViewMut instead");
    }
}

impl<'a, T: fmt::Debug, const R: usize> fmt::Debug for TensorView<'a, T, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorView")
            .field("dims", &self.dims)
            .field("strides", &self.strides)
            .field("len", &self.len)
            .finish()
    }
}

/// Mutable view into a tensor.
pub struct TensorViewMut<'a, T, const R: usize> {
    pub(crate) len: usize,
    pub(crate) dims: [usize; R],
    pub(crate) strides: [usize; R],
    pub(crate) data: *mut T,
    _marker: PhantomData<&'a mut T>,
}

impl<'a, T, const R: usize> TensorViewMut<'a, T, R> {
    pub fn new(dims: [usize; R], strides: [usize; R], data: &'a mut [T]) -> Self {
        let len: usize = dims.iter().product();
        debug!(
            "TensorViewMut::new dims={:?}, strides={:?}, len={}",
            dims, strides, len
        );
        Self {
            len,
            dims,
            strides,
            data: data.as_mut_ptr(),
            _marker: PhantomData,
        }
    }
}

/// SAFETY: lifetime `'a` guarantees underlying slice lives long enough,
/// and dims*strides imply the data slice fits in bounds.
impl<'a, T, const R: usize> HasTensorCore<T> for TensorViewMut<'a, T, R> {
    fn dims(&self) -> &[usize] {
        trace!("TensorViewMut::dims -> {:?}", self.dims);
        &self.dims
    }
    fn strides(&self) -> &[usize] {
        trace!("TensorViewMut::strides -> {:?}", self.strides);
        &self.strides
    }
    fn data(&self) -> &[T] {
        trace!("TensorViewMut::data -> &[T; {}]", self.len);
        unsafe { std::slice::from_raw_parts(self.data, self.len) }
    }
    fn data_mut(&mut self) -> &mut [T] {
        trace!("TensorViewMut::data_mut -> &mut [T; {}]", self.len);
        unsafe { std::slice::from_raw_parts_mut(self.data, self.len) }
    }
}

impl<'a, T: fmt::Debug, const R: usize> fmt::Debug for TensorViewMut<'a, T, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TensorViewMut")
            .field("dims", &self.dims)
            .field("strides", &self.strides)
            .field("len", &self.len)
            .finish()
    }
}
