use cublas::api::{Context, Operation};
use cublas::API;

/// matrices m1 (m by n), m2 (n by k) are in column major
/// the returned matrix (as a continguous vector) is also in column major
fn cuda_matmul(m1: &mut [f32], m2: &mut [f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let context = Context::new().unwrap();
    let mut result = Vec::with_capacity(m * n);
    let mut alpha = [1.];
    let mut beta = [1.];
    unsafe { result.set_len(m * n) }
    API::gemm(&context, Operation::NoTrans, Operation::NoTrans, m as i32, n as i32, k as i32,
              alpha.as_mut_ptr(), m1.as_mut_ptr(), m as i32, m2.as_mut_ptr(), k as i32,
              beta.as_mut_ptr(), result.as_mut_ptr(), m as i32).unwrap();
    result
}

#[cfg(test)]
mod tests {
    use crate::matmul_cublas::cuda_matmul;

    #[test]
    fn test_matmul() {
        // column major
        let mut m1 = vec![1., -9., 8., 2., -4., 23., 2., -2., -30.];
        let mut m2 = vec![21., 9., 2., -2., 43., -3., 9., 8., 4.];
        let result = cuda_matmul(&mut m1, &mut m2, 3, 3, 3);
        assert_eq!(vec![43., -229., 315., 78., -148., 1063., 33., -121., 136.], result);
    }
}
