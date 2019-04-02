#[cfg(test)]
mod tests {
    extern crate test;
    extern crate collenchyma_blas as co_blas;
    extern crate collenchyma as co;

    use co::backend::Backend;
    use co::{Native, BackendConfig};
    use co::framework::IFramework;
    use co_blas::plugin::*;
    use co_blas::transpose::Transpose;
    use co::memory::MemoryType;
    use co::tensor::SharedTensor;
    use co::plugin::numeric_helpers::{cast, Float};
    use test::Bencher;
    use rand::{Rng, thread_rng};
    use rand::distributions::{Standard, Uniform};
    use ndarray::{Array, Ix2};
    use ndarray_rand::RandomExt;

    pub fn write_to_memory<T: ::std::marker::Copy>(mem: &mut MemoryType, data: &[T]) {
        match mem {
            &mut MemoryType::Native(ref mut mem) => {
                let mut mem_buffer = mem.as_mut_slice::<T>();
                for (index, datum) in data.iter().enumerate() {
                    mem_buffer[index] = *datum;
                }
            }
            #[cfg(any(feature = "cuda", feature = "opencl"))]
            _ => assert!(false)
        }
    }

    fn get_scale_one_zero_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>) -> (SharedTensor<T>, SharedTensor<T>) {
        let mut alpha = SharedTensor::<T>::new(backend.device(), &vec![1]).unwrap();
        write_to_memory(alpha.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(1).unwrap()]);

        let mut beta = SharedTensor::<T>::new(backend.device(), &vec![1]).unwrap();
        write_to_memory(beta.get_mut(backend.device()).unwrap(), &[cast::<i32, T>(0).unwrap()]);

        (alpha, beta)
    }

    fn get_gemm_memory<T: Float, B: IFramework + Clone>(backend: &Backend<B>, m: usize, n: usize, k: usize)
        -> (SharedTensor<T>, SharedTensor<T>, SharedTensor<T>) {
        let a_data = Array::random((m, k), Uniform::<f32>::new(-900., 1000.)).as_slice().unwrap().to_vec();
        let b_data = Array::random((k, n), Uniform::<f32>::new(-900., 1000.)).as_slice().unwrap().to_vec();

        let mut a = SharedTensor::<T>::new(backend.device(), &vec![m, k]).unwrap();
        write_to_memory(a.get_mut(backend.device()).unwrap(), a_data.as_slice());

        let mut b = SharedTensor::<T>::new(backend.device(), &vec![k, n]).unwrap();
        write_to_memory(b.get_mut(backend.device()).unwrap(), b_data.as_slice());

        let c = SharedTensor::<T>::new(backend.device(), &vec![m, n]).unwrap();

        (a, b, c)
    }

    fn get_native_backend() -> Backend<Native> {
        let framework = Native::new();
        let hardwares = framework.hardwares().to_vec();
        let backend_config = BackendConfig::new(framework, &hardwares);
        Backend::new(backend_config).unwrap()
    }

    #[test]
    fn test_gemm() {
        let (m, k, n) = (5, 5, 10);
        let backend = get_native_backend();
        let (mut a, mut b, mut c) = get_gemm_memory::<f32, Native>(&backend, m, n, k);
        let (mut alpha, mut beta) = get_scale_one_zero_memory::<f32, Native>(&backend);

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

    #[cfg(feature = "cuda")]
    mod cuda {
        use co::backend::{IBackend, Backend, BackendConfig};
        use co::framework::IFramework;
        use co::frameworks::Native;
        use co::frameworks::Cuda;
        use co_blas::plugin::*;
        use co_blas::transpose::Transpose;
        use super::*;

        fn get_cuda_backend() -> Backend<Cuda> {
            let framework = Cuda::new();
            let hardwares = framework.hardwares().to_vec();
            let backend_config = BackendConfig::new(framework, &hardwares);
            Backend::new(backend_config).unwrap()
        }

        #[test]
        fn gemm_cuda_f32() {
            let (m, n, k) = (50, 100, 50);
            let native_backend = get_native_backend();
            let cuda_backend = get_cuda_backend();
            let (mut a, mut b, mut c) = get_gemm_memory::<f32, Native>(&native_backend, m, n, k);
            let (mut alpha, mut beta) = get_scale_one_zero_memory::<f32, Native>(&native_backend);

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

            cuda_backend.gemm(&mut alpha, Transpose::NoTrans, &mut a, Transpose::NoTrans, &mut b, &mut beta, &mut c).unwrap();
            cuda_backend.synchronize().unwrap();
            c.sync(native_backend.device()).unwrap();
            let mem = c.get(native_backend.device()).unwrap().as_native().unwrap();

            for (i, j) in c_arr.as_slice().unwrap().iter().zip(mem.as_slice::<f32>()) {
                assert!((i - j).abs() < 1e-6);
            }

            println!("after: {:?}", mem);
        }
    }
}
