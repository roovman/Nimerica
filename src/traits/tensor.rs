use crate::utils::logging::{trace, error};


pub trait HasTensorCore<T> {
    fn dims(&self) -> &[usize];
    fn strides(&self) -> &[usize];
    fn data(&self) -> &[T];
    fn data_mut(&mut self) -> &mut [T];
}

/// High-level tensor behavior. Blanket-implemented for all HasTensorCore<T>.
pub trait TensorLike<T>: HasTensorCore<T> {
    fn rank(&self) -> usize {
        let r = self.dims().len();
        trace!("rank() -> {}", r);
        r
    }

    fn len(&self) -> usize {
        let l: usize = self.dims().iter().product();
        trace!("len() -> {}", l);
        l
    }

    fn is_empty(&self) -> bool {
        let e = self.len() == 0;
        trace!("is_empty() -> {}", e);
        e
    }

    fn offset(&self, idx: &[usize]) -> Option<usize> {
        let rank = self.rank();
        if idx.len() != rank {
            error!(
                "offset({:?}) failed: expected rank {}, got {}",
                idx,
                rank,
                idx.len()
            );
            return None;
        }
        let off: usize = idx
            .iter()
            .zip(self.strides().iter())
            .map(|(i, s)| i * s)
            .sum();
        trace!("offset({:?}) -> Some({})", idx, off);
        Some(off)
    }

    fn is_contiguous(&self) -> bool {
        let mut expected = 1;
        for (d, s) in self.dims().iter().rev().zip(self.strides().iter().rev()) {
            if *s != expected {
                trace!("is_contiguous() -> false");
                return false;
            }
            expected *= d;
        }
        trace!("is_contiguous() -> true");
        true
    }

    fn get(&self, idx: &[usize]) -> Option<&T> {
        match self.offset(idx) {
            Some(off) => match self.data().get(off) {
                Some(val) => {
                    trace!("get({:?}) -> &T @ offset {}", idx, off);
                    Some(val)
                }
                None => {
                    error!("get({:?}) out of bounds @ offset {}", idx, off);
                    None
                }
            },
            None => None,
        }
    }

    fn set(&mut self, idx: &[usize], val: T)
    where
        T: Clone,
    {
        if let Some(off) = self.offset(idx) {
            if let Some(slot) = self.data_mut().get_mut(off) {
                *slot = val;
                trace!("set({:?}) -> success @ offset {}", idx, off);
            } else {
                error!("set({:?}) failed: out of bounds @ offset {}", idx, off);
            }
        }
    }


    unsafe fn get_unchecked(&self, idx: &[usize]) -> &T {
        let off = self.offset(idx).unwrap();
        trace!("get_unchecked({:?}) -> &T @ offset {}", idx, off);
        unsafe { &*self.data().as_ptr().add(off) }
    }

    unsafe fn set_unchecked(&mut self, idx: &[usize], val: T) {
        let off = self.offset(idx).unwrap();
        trace!("set_unchecked({:?}) -> success @ offset {}", idx, off);
        let ptr = unsafe { self.data_mut().as_mut_ptr().add(off) };
        unsafe { *ptr = val };
    }
}

impl<T, U> TensorLike<T> for U where U: HasTensorCore<T> {}
