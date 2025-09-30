pub mod tensor;
pub mod utils;
pub mod traits;
pub use traits::*;     // HasTensorCore, TensorLike, Ring, Field
pub use tensor::*;     // Tensor and friends

pub fn init() {
    utils::logging::init_logging();
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use super::*;

    
    fn init_logging_for_tests() {
        crate::init(); // calls utils::logging::init_logging()
    }

    #[test]
    fn tensor_new_and_access() {
         //init_logging_for_tests();
        let t = Tensor::new([2, 2], vec![1, 2, 3, 4]).unwrap();
        assert_eq!(t.get(&[0, 0]).unwrap(), &1);
        assert_eq!(t.get(&[0, 1]).unwrap(), &2);
        assert_eq!(t.get(&[1, 0]).unwrap(), &3);
        assert_eq!(t.get(&[1, 1]).unwrap(), &4);

        // Out of bounds
        assert!(t.get(&[2, 0]).is_err());
        assert!(t.get(&[0, 3]).is_err());
    }

    #[test]
    fn tensor_set_and_mutate() {
         //init_logging_for_tests();
        let mut t = Tensor::new([2, 2], vec![1, 2, 3, 4]).unwrap();
        t.set(&[1, 0], 99).unwrap();
        assert_eq!(t.get(&[1, 0]).unwrap(), &99);

        // Out of bounds write
        assert!(t.set(&[2, 0], 123).is_err());
    }

    #[test]
    fn tensor_view_read_only() {
         //init_logging_for_tests();
        let t = Tensor::new([2, 2], vec![10, 20, 30, 40]).unwrap();
        let view = t.view();

        assert_eq!(view.get(&[0, 0]).unwrap(), &10);
        assert_eq!(view.get(&[1, 1]).unwrap(), &40);

        // // Immutable view must panic on data_mut
        // let result = std::panic::catch_unwind(|| {
        //     let _ = view.data_mut();
        // });
        //assert!(result.is_err());
    }

    #[test]
    fn tensor_view_mut_write_back() {
         //init_logging_for_tests();
        let mut t = Tensor::new([2, 2], vec![10, 20, 30, 40]).unwrap();
        {
            let mut v = t.view_mut();
            v.set(&[0, 1], 123).unwrap();
            assert_eq!(v.get(&[0, 1]).unwrap(), &123);
        }
        // change reflected in parent tensor
        assert_eq!(t.get(&[0, 1]).unwrap(), &123);
    }

    #[test]
    fn tensor_contiguity() {
         //init_logging_for_tests();
        let t = Tensor::new([2, 3], vec![1, 2, 3, 4, 5, 6]).unwrap();
        assert!(t.is_contiguous());

        // Simulate slice with weird stride
        let mut v = t.view();
        v.strides = [10, 1];
        assert!(!v.is_contiguous());
    }

    #[test]
    fn tensor_defaulted_is_zeroed() {
         //init_logging_for_tests();
        let t = Tensor::<i32, 2>::defaulted([2, 3]);
        assert_eq!(t.len(), 6);
        assert!(t.data().iter().all(|&x| x == 0));
    }

    #[test]
    fn tensor_shape_mismatch_error() {
         //init_logging_for_tests();
        let bad = Tensor::new([2, 2], vec![1, 2, 3]);
        assert!(bad.is_err());
    }

        #[test]
    fn tensor_view_aliasing_behavior() {
        let mut t = Tensor::new([2, 2], vec![1, 2, 3, 4]).unwrap();
        
        {
            // Create a mutable view and change a value
            let mut v_mut = t.view_mut();
            v_mut.set(&[0, 1], 999).unwrap();
        }

        // Immutable view should see the updated value
        let v = t.view();
        assert_eq!(v.get(&[0, 1]).unwrap(), &999);

        // Parent tensor should also reflect the change
        assert_eq!(t.get(&[0, 1]).unwrap(), &999);
    }

}

