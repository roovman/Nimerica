use std::fmt;
use crate::tensor::traits::{HasTensorCore, TensorLike};

/// Blanket implementation: any HasTensorCore<T> is automatically TensorLike<T>.
impl<T, U> TensorLike<T> for U
where
    U: HasTensorCore<T>,
{
    
    fn rank(&self) -> usize {
        HasTensorCore::dims(self).len()
    }

    fn dims(&self) -> &[usize] {
        HasTensorCore::dims(self)
    }

    fn get(&self, idx: &[usize]) -> Option<&T> {
        if idx.len() != HasTensorCore::dims(self).len() {
            return None;
        }
        let flat: usize = idx
            .iter()
            .zip(HasTensorCore::strides(self).iter())
            .map(|(i, s)| i * s)
            .sum();
        HasTensorCore::data(self).get(flat)
    }

    fn set(&mut self, idx: &[usize], val: T) {
        if idx.len() != HasTensorCore::dims(self).len() {
            return;
        }
        let flat: usize = idx
            .iter()
            .zip(HasTensorCore::strides(self).iter())
            .map(|(i, s)| i * s)
            .sum();
        if let Some(slot) = HasTensorCore::data_mut(self).get_mut(flat) {
            *slot = val;
        }
    }
}

/// Recursive pretty-printer for tensor-like objects.
pub fn format_tensor<T, U>(tensor: &U, f: &mut fmt::Formatter<'_>) -> fmt::Result
where
    T: fmt::Display,
    U: TensorLike<T>,
{
    let dims = tensor.dims();
    if dims.is_empty() {
        // Rank-0 tensor (scalar)
        if let Some(val) = tensor.get(&[]) {
            write!(f, "{}", val)?;
        }
        return Ok(());
    }

    fn format_rec<T, U>(
        tensor: &U,
        f: &mut fmt::Formatter<'_>,
        prefix: &mut Vec<usize>,
    ) -> fmt::Result
    where
        T: fmt::Display,
        U: TensorLike<T>,
    {
        let dims = tensor.dims();
        if prefix.len() == dims.len() - 1 {
            write!(f, "[")?;
            for i in 0..dims[dims.len() - 1] {
                prefix.push(i);
                if let Some(val) = tensor.get(prefix) {
                    write!(f, "{}", val)?;
                }
                prefix.pop();
                if i + 1 < dims[dims.len() - 1] {
                    write!(f, " ")?;
                }
            }
            write!(f, "]")?;
        } else {
            write!(f, "[")?;
            for i in 0..dims[prefix.len()] {
                prefix.push(i);
                format_rec::<T, U>(tensor, f, prefix)?;
                prefix.pop();
                if i + 1 < dims[prefix.len()] {
                    write!(f, " ")?;
                }
            }
            write!(f, "]")?;
        }
        Ok(())
    }

    format_rec::<T, U>(tensor, f, &mut Vec::new())
}

/// A wrapper Display impl so any TensorLike<T> can be pretty-printed.
pub struct TensorDisplay<'a, T, U>(&'a U, std::marker::PhantomData<T>)
where
    U: TensorLike<T>;

impl<'a, T, U> fmt::Display for TensorDisplay<'a, T, U>
where
    T: fmt::Display,
    U: TensorLike<T>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        format_tensor::<T, U>(self.0, f)
    }
}

pub fn display_tensor<'a, T, U>(tensor: &'a U) -> TensorDisplay<'a, T, U>
where
    U: TensorLike<T>,
{
    TensorDisplay(tensor, std::marker::PhantomData)
}




