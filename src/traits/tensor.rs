use crate::utils::logging::{trace, error, TensorError};


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

    fn offset(&self, idx: &[usize]) -> Result<usize, TensorError> {
    let rank = self.rank();
    if idx.len() != rank {
        return Err(TensorError::OutOfBounds {
            idx: idx.to_vec(),
            dims: self.dims().to_vec(),
        });
    }

    for (i, d) in idx.iter().zip(self.dims()) {
        if *i >= *d {
            error!(
                "offset({:?}) failed: index {} out of bounds for dim size {}",
                idx, i, d
            );
            return Err(TensorError::OutOfBounds {
                idx: idx.to_vec(),
                dims: self.dims().to_vec(),
            });
        }
    }

    let off: usize = idx
        .iter()
        .zip(self.strides().iter())
        .map(|(i, s)| i * s)
        .sum();

    trace!("offset({:?}) -> {}", idx, off);
    Ok(off)
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

    fn get(&self, idx: &[usize]) -> Result<&T, TensorError> {
        let off = self.offset(idx)?;
        match self.data().get(off) {
            Some(val) => {
                trace!("get({:?}) -> &T @ offset {}", idx, off);
                Ok(val)
            }
            None => {
                error!("get({:?}) out of bounds @ offset {}", idx, off);
                Err(TensorError::OutOfBounds {
                    idx: idx.to_vec(),
                    dims: self.dims().to_vec(),
                })
            }
        }
    }

    fn set(&mut self, idx: &[usize], val: T) -> Result<(), TensorError>
    where
        T: Clone,
    {
        let off = self.offset(idx)?;
        if let Some(slot) = self.data_mut().get_mut(off) {
            *slot = val;
            trace!("set({:?}) -> success @ offset {}", idx, off);
            Ok(())
        } else {
            error!("set({:?}) failed: out of bounds @ offset {}", idx, off);
            Err(TensorError::OutOfBounds {
                idx: idx.to_vec(),
                dims: self.dims().to_vec(),
            })
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
