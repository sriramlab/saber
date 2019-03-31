use ndarray::{Array, Ix1, Ix2};
use num_traits::{pow, PrimInt};

use crate::stats_util::sum;

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
pub fn mailman_zero_one_two(indicator_vec: Vec<usize>, vector: &Vec<f32>) -> Result<Array<f32, Ix1>, String> {
    let mut px = Array::from_vec(vec![0f32; vector.len()]);
    for j in 0..indicator_vec.len() {
        px[indicator_vec[j]] += vector[j];
    }
    let mut len = vector.len();
    if len < 3 {
        return Err(format!("the length ({}) of the vector is not a positive power of 3", vector.len()));
    }
    let mut log_3_len = 0;
    while len != 1 {
        if len % 3 != 0 {
            return Err(format!("the length ({}) of the vector is not a positive power of 3", vector.len()));
        }
        log_3_len += 1;
        len /= 3;
    }

    let mut result = vec![0f32; log_3_len];
    let mut seg_len = vector.len() / 3;
    let mut z_accumulator = px;
    for i in 0..log_3_len {
        let z1 = z_accumulator.slice(s![0..seg_len]);
        let z2 = z_accumulator.slice(s![seg_len..seg_len * 2]);
        let z3 = z_accumulator.slice(s![seg_len * 2 .. seg_len * 3]);
        result[i] = (sum(z2.iter()) + 2. * sum(z3.iter())) as f32;
        z_accumulator = (z1.into_owned() + z2) + z3;
        seg_len /= 3;
    }
    Ok(Array::from_vec(result))
}

#[cfg(test)]
mod tests {
    use super::{mailman_zero_one_two, zero_one_two_matrix_to_indicator_vec};

    #[test]
    fn test_zero_one_two_matrix_to_indicator_vec() {
        let a = array![[0, 1, 1, 2, 1, 1, 0, 0, 1], [2, 1, 2, 0, 0, 1, 0, 1, 0]];
        let indicator_vec = zero_one_two_matrix_to_indicator_vec(&a);
        assert_eq!(vec![2, 4, 5, 6, 3, 4, 0, 1, 3], indicator_vec);
    }

    #[test]
    fn test_mailman_zero_one_two() {
        let a = array![[0, 1, 1, 2, 1, 1, 0, 0, 1], [2, 1, 2, 0, 0, 1, 0, 1, 0]];
        let x = array![5., 1., -2., 3., 0., 4., 5., 2., 0.];
        let ax = a.mapv(|e| e as f32).dot(&x);
        let indicator_vec = zero_one_two_matrix_to_indicator_vec(&a);
        let result = mailman_zero_one_two(indicator_vec, &x.to_vec()).unwrap();
        assert_eq!(ax, result);
    }
}
