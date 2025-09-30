pub mod tensor;
pub mod utils;

pub fn init() {
    utils::logging::init_logging();
}

#[cfg(test)]
mod tests {
    use crate::tensor::*;

    #[test]
    fn test_create_and_get() {
        let t = Tensor::new([2, 2], vec![1, 2, 3, 4]);
        assert_eq!(t.get(&[0,0]), Some(&1));
        assert_eq!(t.get(&[1,1]), Some(&4));

    }


}