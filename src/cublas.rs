#[cfg(feature = "cuda")]
pub mod linalg {
    use co::backend::{Backend, BackendConfig, IBackend};
    use co::framework::IFramework;
    use co::frameworks::{Cuda, Native};
    use co::memory::MemoryType;
    use co::plugin::numeric_helpers::{cast, Float};
    use co::tensor::SharedTensor;
    use co_blas::plugin::*;
    use co_blas::transpose::Transpose;
    use ndarray::{Array, Ix2};
    use ndarray_rand::RandomExt;
    use rand::distributions::{Standard, Uniform};

    pub fn write_to_native_memory<T: ::std::marker::Copy>(mem: &mut MemoryType, data: &[T]) {
        match mem {
            MemoryType::Native(ref mut mem) => {
                let mut mem_buffer = mem.as_mut_slice::<T>();
                for (i, datum) in data.iter().enumerate() {
                    mem_buffer[i] = *datum;
                }
            }
            #[cfg(any(feature = "cuda", feature = "opencl"))]
            _ => assert!(false)
        }
    }

    pub fn get_shared_tensor_one<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> SharedTensor<T> {
        let mut one = match SharedTensor::<T>::new(backend.device(), &vec![1]) {
            Ok(one) => one,
            Err(why) => panic!("{}", why)
        };
        write_to_native_memory(one.get_mut(backend.device()).unwrap(), &[T::one()]);
        one
    }

    pub fn get_shared_tensor_zero<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> SharedTensor<T> {
        let mut zero = match SharedTensor::<T>::new(backend.device(), &vec![1]) {
            Ok(zero) => zero,
            Err(why) => panic!("{}", why)
        };
        write_to_native_memory(zero.get_mut(backend.device()).unwrap(), &[T::zero()]);
        zero
    }

    pub fn get_one_zero_shared_tensor<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedTensor<T>, SharedTensor<T>) {
        (get_shared_tensor_one::<T, B>(backend), get_shared_tensor_zero::<T, B>(backend))
    }

    pub fn get_native_backend() -> Backend<Native> {
        let framework = Native::new();
        let hardwares = framework.hardwares().to_vec();
        let backend_config = BackendConfig::new(framework, &hardwares);
        match Backend::new(backend_config) {
            Ok(backend) => backend,
            Err(why) => panic!("{}", why)
        }
    }

    pub fn get_cuda_backend() -> Backend<Cuda> {
        let framework = Cuda::new();
        let hardwares = framework.hardwares().to_vec();
        let backend_config = BackendConfig::new(framework, &hardwares);
        match Backend::new(backend_config) {
            Ok(backend) => backend,
            Err(why) => panic!("{}", why)
        }
    }

    pub fn matmul_f32(arr1: &[f32], arr2: &[f32], m: usize, n: usize, k: usize, a_t: Transpose, b_t: Transpose) -> Vec<f32> {
        let native_backend = get_native_backend();
        let cuda_backend = get_cuda_backend();

        let mut a = SharedTensor::<f32>::new(native_backend.device(), &vec![m, k]).unwrap();
        write_to_native_memory(a.get_mut(native_backend.device()).unwrap(), arr1);

        let mut b = SharedTensor::<f32>::new(native_backend.device(), &vec![k, n]).unwrap();
        write_to_native_memory(b.get_mut(native_backend.device()).unwrap(), arr2);

        let mut c = SharedTensor::<f32>::new(native_backend.device(), &vec![m, n]).unwrap();

        let mut one = get_shared_tensor_one::<f32, Native>(&native_backend);
        let mut another_one = get_shared_tensor_one::<f32, Native>(&native_backend);

        cuda_backend.gemm(&mut one, a_t, &mut a, b_t, &mut b, &mut another_one, &mut c).unwrap();
        cuda_backend.synchronize().unwrap();
        c.sync(native_backend.device()).unwrap();
        c.get(native_backend.device()).unwrap().as_native().unwrap().as_slice::<f32>().to_vec()
    }

    /// arr1 has shape m by k
    /// arr2 has shape k by n
    pub fn mul_xtxz_f32(arr1: &[f32], arr2: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
        let native_backend = get_native_backend();
        let cuda_backend = get_cuda_backend();

        let mut x_arr = SharedTensor::<f32>::new(native_backend.device(), &vec![m, k]).unwrap();
        write_to_native_memory(x_arr.get_mut(native_backend.device()).unwrap(), arr1);

        let mut z_arr = SharedTensor::<f32>::new(native_backend.device(), &vec![k, n]).unwrap();
        write_to_native_memory(z_arr.get_mut(native_backend.device()).unwrap(), arr2);

        let mut xz = SharedTensor::<f32>::new(native_backend.device(), &vec![m, n]).unwrap();
        let mut c = SharedTensor::<f32>::new(native_backend.device(), &vec![k, n]).unwrap();

        let mut one = get_shared_tensor_one::<f32, Native>(&native_backend);
        let mut another_one = get_shared_tensor_one::<f32, Native>(&native_backend);

        cuda_backend.gemm(&mut one, Transpose::NoTrans, &mut x_arr, Transpose::NoTrans, &mut z_arr, &mut another_one, &mut xz).unwrap();
        cuda_backend.gemm(&mut one, Transpose::Trans, &mut x_arr, Transpose::NoTrans, &mut xz, &mut another_one, &mut c).unwrap();

        cuda_backend.synchronize().unwrap();
        c.sync(native_backend.device()).unwrap();
        c.get(native_backend.device()).unwrap().as_native().unwrap().as_slice::<f32>().to_vec()
    }
}

#[cfg(feature = "cuda")]
#[cfg(test)]
mod tests {
    extern crate test;
    extern crate collenchyma_blas as co_blas;
    extern crate collenchyma as co;

    use test::Bencher;

    use co::backend::{Backend, BackendConfig, IBackend};
    use co::framework::IFramework;
    use co::frameworks::{Cuda, Native};
    use co::memory::MemoryType;
    use co::plugin::numeric_helpers::{cast, Float};
    use co::tensor::SharedTensor;
    use co_blas::plugin::*;
    use co_blas::transpose::Transpose;
    use ndarray::{Array, Ix2};
    use ndarray_rand::RandomExt;
    use rand::distributions::{Standard, Uniform};

    use super::linalg::{get_cuda_backend, get_native_backend, get_one_zero_shared_tensor, write_to_native_memory};

    fn get_gemm_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>, m: usize, n: usize, k: usize)
        -> (SharedTensor<T>, SharedTensor<T>, SharedTensor<T>) {
        let a_data = Array::random((m, k), Uniform::<f32>::new(-900., 1000.)).as_slice().unwrap().to_vec();
        let b_data = Array::random((k, n), Uniform::<f32>::new(-900., 1000.)).as_slice().unwrap().to_vec();

        let mut a = SharedTensor::<T>::new(backend.device(), &vec![m, k]).unwrap();
        write_to_native_memory(a.get_mut(backend.device()).unwrap(), a_data.as_slice());

        let mut b = SharedTensor::<T>::new(backend.device(), &vec![k, n]).unwrap();
        write_to_native_memory(b.get_mut(backend.device()).unwrap(), b_data.as_slice());

        let c = SharedTensor::<T>::new(backend.device(), &vec![m, n]).unwrap();

        (a, b, c)
    }

    #[test]
    fn test_gemm() {
        let (m, k, n) = (5, 5, 10);
        let backend = get_native_backend();
        let (mut a, mut b, mut c) = get_gemm_memory::<f32, Native>(&backend, m, n, k);
        let (mut alpha, mut beta) = get_one_zero_shared_tensor::<f32, Native>(&backend);

        let a_data = a.get(backend.device()).unwrap().as_native().unwrap();
        println!("a_data: {:?}", a_data.as_slice::<f32>());
        let b_data = b.get(backend.device()).unwrap().as_native().unwrap();
        println!("b_data: {:?}", b_data.as_slice::<f32>());

        let a_arr = Array::<f32, Ix2>::from_shape_vec((m, k),
                                                      a_data.as_slice::<f32>().to_vec()).unwrap();
        let b_arr = Array::<f32, Ix2>::from_shape_vec((k, n),
                                                      b_data.as_slice::<f32>().to_vec()).unwrap();
        let c_arr = a_arr.dot(&b_arr);

        println!("before {:?}", c.get(backend.device()).unwrap().as_native().unwrap().as_slice::<f32>());
        backend.gemm(&mut alpha, Transpose::NoTrans, &mut a, Transpose::NoTrans, &mut b, &mut beta, &mut c);
        let mem = c.get(backend.device()).unwrap().as_native().unwrap();
        for (i, j) in c_arr.as_slice().unwrap().iter().zip(mem.as_slice::<f32>()) {
            assert!((i - j).abs() < 1e-6);
        }
        println!("after {:?}", c.get(backend.device()).unwrap().as_native().unwrap().as_slice::<f32>());
    }

    #[test]
    fn gemm_cuda_f32() {
        let (m, n, k) = (50, 100, 50);
        let native_backend = get_native_backend();
        let cuda_backend = get_cuda_backend();
        let (mut a, mut b, mut c) = get_gemm_memory::<f32, Native>(&native_backend, m, n, k);
        let (mut one, _) = get_one_zero_shared_tensor::<f32, Native>(&native_backend);

        let a_data = a.get(native_backend.device()).unwrap().as_native().unwrap();
        println!("a_data: {:?}", a_data.as_slice::<f32>());
        let b_data = b.get(native_backend.device()).unwrap().as_native().unwrap();
        println!("b_data: {:?}", b_data.as_slice::<f32>());

        let a_arr = Array::<f32, Ix2>::from_shape_vec((m, k),
                                                      a_data.as_slice::<f32>().to_vec()).unwrap();
        let b_arr = Array::<f32, Ix2>::from_shape_vec((k, n),
                                                      b_data.as_slice::<f32>().to_vec()).unwrap();
        let c_arr = a_arr.dot(&b_arr);

        println!("before {:?}", c.get(native_backend.device()).unwrap().as_native().unwrap().as_slice::<f32>());

        cuda_backend.gemm(&mut one, Transpose::NoTrans, &mut a, Transpose::NoTrans, &mut b, &mut one, &mut c).unwrap();
        cuda_backend.synchronize().unwrap();
        c.sync(native_backend.device()).unwrap();
        let mem = c.get(native_backend.device()).unwrap().as_native().unwrap();

        for (i, j) in c_arr.as_slice().unwrap().iter().zip(mem.as_slice::<f32>()) {
            assert!((i - j).abs() < 1e-6);
        }

        println!("after: {:?}", mem);
    }
}
