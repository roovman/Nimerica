
/// Low-level access to tensor internals.
/// Anything that exposes dims/strides/data can be lifted to TensorLike.
pub trait HasTensorCore<T> {
    fn dims(&self) -> &[usize];
    fn strides(&self) -> &[usize];
    fn data(&self) -> &[T];
    fn data_mut(&mut self) -> &mut [T];
}

/// High-level tensor behavior.
/// Blanket-implemented for all HasTensorCore types.
pub trait TensorLike<T> {
    fn rank(&self) -> usize;
    fn dims(&self) -> &[usize];
    fn get(&self, idx: &[usize]) -> Option<&T>;
    fn set(&mut self, idx: &[usize], val: T);
}

