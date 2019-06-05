extern crate test;

use ndarray::{Array, Slice, Ix2, Axis};
use num_traits::{pow, PrimInt};
use ndarray::prelude::aview1;

pub fn zero_one_two_matrix_to_indicator_vec<T: PrimInt>(matrix: &Array<T, Ix2>) -> Vec<usize> {
    let (m, n) = matrix.dim();
    assert_eq!(pow(3usize, m), n, "# columns n ({}) is not 3 to the power of m ({}), # rows", n, m);
    let mut indicator_vec = Vec::new();
    for col in matrix.gencolumns() {
        let mut index = 0usize;
        for (i, b) in col.iter().rev().enumerate() {
            index += 3usize.pow(i as u32) * (*b).to_usize().unwrap();
        }
        indicator_vec.push(index);
    }
    indicator_vec
}

/// A = UP
/// Ax = UPx = U(Px)
pub fn mailman_zero_one_two(indicator_vec: &Vec<usize>, vecs: &Array<f32, Ix2>) -> Result<Array<f32, Ix2>, String> {
    let (mut len, num_vecs) = vecs.dim();
    let axis0 = Axis(0);
    let mut px = Array::from_shape_vec((len, num_vecs), vec![0f32; len * num_vecs]).unwrap();
    for j in 0..indicator_vec.len() {
        let mut row = px.index_axis_mut(axis0, indicator_vec[j]);
        row += &vecs.index_axis(axis0, j);
    }

    if len < 3 {
        return Err(format!("the length ({}) of the vector is not a positive power of 3", vecs.len()));
    }
    let mut log_3_len = 0;
    while len != 1 {
        if len % 3 != 0 {
            return Err(format!("the length ({}) of the vector is not a positive power of 3", vecs.len()));
        }
        log_3_len += 1;
        len /= 3;
    }

    let mut result = Array::<f32, Ix2>::zeros((log_3_len, num_vecs));
    let mut seg_len = vecs.dim().0 / 3;
    let mut z_accumulator = px;
    for i in 0..log_3_len {
        let ones = Array::<f32, Ix2>::ones((1, seg_len));
        let z1 = z_accumulator.slice_axis(axis0, Slice::from(0..seg_len));
        let z2 = z_accumulator.slice_axis(axis0, Slice::from(seg_len..seg_len * 2));
        let z3 = z_accumulator.slice_axis(axis0, Slice::from(seg_len * 2..));
        let mut row = result.index_axis_mut(axis0, i);
        row += &aview1((ones.dot(&z2) + 2. * ones.dot(&z3)).as_slice().unwrap());
        z_accumulator = (z1.into_owned() + z2) + z3;
        seg_len /= 3;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::{mailman_zero_one_two, zero_one_two_matrix_to_indicator_vec};
    use ndarray_rand::RandomExt;
    use ndarray::{Array, Ix2};
    use rand::distributions::Uniform;
    use rand::distributions::uniform::SampleUniform;
    use super::test::Bencher;
    use num_traits::{NumCast, Float};

    #[test]
    fn test_zero_one_two_matrix_to_indicator_vec() {
        let a = array![[0, 1, 1, 2, 1, 1, 0, 0, 1], [2, 1, 2, 0, 0, 1, 0, 1, 0]];
        let indicator_vec = zero_one_two_matrix_to_indicator_vec(&a);
        assert_eq!(vec![2, 4, 5, 6, 3, 4, 0, 1, 3], indicator_vec);
    }

    #[test]
    fn test_mailman_zero_one_two() {
        let a = array![[0, 1, 1, 2, 1, 1, 0, 0, 1], [2, 1, 2, 0, 0, 1, 0, 1, 0]];
        let x = Array::from_shape_vec(
            (9, 1), vec![5., 1., -2., 3., 0., 4., 5., 2., 0.]).unwrap();
        let ax = a.mapv(|e| e as f32).dot(&x);
        let indicator_vec = zero_one_two_matrix_to_indicator_vec(&a);
        let result = mailman_zero_one_two(&indicator_vec, &x).unwrap();
        assert_eq!(ax, result);
    }

    fn generate_mailman_mat_vec<A, B>(num_vectors: usize) -> (Array<A, Ix2>, Array<B, Ix2>) where A: NumCast, B: Float + SampleUniform {
        let num_rows = 5usize;
        let num_cols = 3usize.pow(num_rows as u32);
        let matrix = Array::random(
            (num_rows, num_cols), Uniform::<u8>::new(0, 3)).mapv(|e| A::from(e).unwrap());
        let vecs = Array::random(
            (num_cols, num_vectors), Uniform::<B>::new(B::from(-100.).unwrap(), B::from(100.).unwrap()));
        (matrix, vecs)
    }

    #[bench]
    fn bench_mailman_zero_one_two(b: &mut Bencher) {
        let (matrix, vecs) = generate_mailman_mat_vec::<u8, f32>(100);
        let indicator_vec = zero_one_two_matrix_to_indicator_vec(&matrix);
        b.iter(|| {
            mailman_zero_one_two(&indicator_vec, &vecs);
        })
    }

    #[bench]
    fn bench_regular_matvec(b: &mut Bencher) {
        let (matrix, vecs) = generate_mailman_mat_vec::<f32, f32>(100);
        b.iter(|| matrix.dot(&vecs))
    }
}
