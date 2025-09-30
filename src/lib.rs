pub mod tensor;
pub mod utils;
pub mod traits;
pub use traits::*;     // HasTensorCore, TensorLike, Ring, Field
pub use tensor::*;     // Tensor and friends

pub fn init() {
    utils::logging::init_logging();
}

#[cfg(test)]
mod tests {
    use super::*; // bring in Tensor, TensorLike, HasTensorCore
    use crate::init;
        #[test]
        fn tensor_test() {
            crate::utils::logging::init_logging(); // safe to call multiple times
            let t = Tensor::new([2,2], vec![1,2,3,4]).unwrap();
            assert_eq!(t.get(&[0,0]), Some(&1));
        }

    #[test]
    fn test_tensor_new_and_access() {
        init(); // set up logging

        // Create a 2x2 tensor
        let t = Tensor::new([2, 2], vec![1, 2, 3, 4]).unwrap();

        // Check metadata
        assert_eq!(t.dims(), &[2, 2]);
        assert_eq!(t.strides(), &[2, 1]);
        assert_eq!(t.len(), 4);
        assert!(t.is_contiguous());

        // Read values with TensorLike::get
        assert_eq!(t.get(&[0, 0]), Some(&1));
        assert_eq!(t.get(&[0, 1]), Some(&2));
        assert_eq!(t.get(&[1, 0]), Some(&3));
        assert_eq!(t.get(&[1, 1]), Some(&4));

        // Safe out-of-bounds read
        assert_eq!(t.get(&[2, 0]), None);
    }

    #[test]
    fn test_tensor_set_and_mutation() {
        init();

        let mut t = Tensor::new([2, 2], vec![0, 0, 0, 0]).unwrap();
        assert_eq!(t.get(&[1, 1]), Some(&0));

        // Mutate with set
        t.set(&[1, 1], 42);
        assert_eq!(t.get(&[1, 1]), Some(&42));

        // Out of bounds set (should log error, no panic)
        t.set(&[2, 0], 99);
        assert_eq!(t.get(&[2, 0]), None);
    }

    #[test]
    fn test_tensor_shape_mismatch() {
        init();

        // Wrong length: 3 elements but shape 2x2
        let result = Tensor::new([2, 2], vec![1, 2, 3]);
        assert!(result.is_err());

        if let Err(err) = result {
            match err {
                crate::tensor::data::TensorError::ShapeMismatch { expected, got } => {
                    assert_eq!(expected, 4);
                    assert_eq!(got, 3);
                }
            }
        }
    }
}
