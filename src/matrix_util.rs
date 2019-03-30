use ndarray::{Array, Ix2, ScalarOperand};
use ndarray_rand::RandomExt;
use num_traits::{Float, FromPrimitive, NumAssign, ToPrimitive};
use rand::distributions::Bernoulli;

use crate::stats_util::sum_of_squares;

pub fn generate_plus_minus_one_bernoulli_matrix(num_rows: usize, num_cols: usize) -> Array<f32, Ix2> {
    Array::random((num_rows, num_cols), Bernoulli::new(0.5)).mapv(|e| (e as i32 * 2 - 1) as f32)
}

/// `ddof`: delta degrees of freedom, where the denominator will be `N - ddof`,
/// where `N` is the number of elements per row
pub fn normalize_matrix_row_wise<A>(mut matrix: Array<A, Ix2>, ddof: usize) -> Array<A, Ix2>
    where A: ToPrimitive + FromPrimitive + NumAssign + Float + ScalarOperand {
    let (num_rows, num_cols) = matrix.dim();
    let ones = Array::from_shape_vec((num_cols, 1), vec![A::one(); num_cols]).unwrap();

    // mean center
    let mean_vec = matrix.dot(&ones) / A::from(num_cols).unwrap();
    matrix -= &mean_vec;

    // now that each row has zero mean, so we can just use the sum of squares
    let denominator = A::from(num_cols - ddof).unwrap();
    let mut std_vec = vec![A::zero(); num_rows];
    let mut i = 0;
    for row in matrix.genrows() {
        std_vec[i] = (A::from(sum_of_squares(row.iter())).unwrap() / denominator).sqrt();
        i += 1;
    };

    let std_arr = Array::from_shape_vec((num_rows, 1), std_vec).unwrap();
    matrix /= &std_arr;
    matrix
}

#[cfg(test)]
mod tests {
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use rand::distributions::Uniform;

    use crate::stats_util::{mean, std};

    use super::normalize_matrix_row_wise;

    #[test]
    fn test_normalize_matrix_row_wise() {
        let ddof = 1;
        let (num_rows, num_cols) = (50, 100);
        let mut matrix = Array::random((num_rows, num_cols), Uniform::new(-10f32, 50f32));
        matrix = normalize_matrix_row_wise(matrix, ddof);

        // check that the means are close to 0 and the standard deviations are close to 1
        for row in matrix.genrows() {
            assert!(mean(row.iter()).abs() < 1e-6);
            assert!((std(row.iter(), ddof) - 1.).abs() < 1e-6);
        }
    }
}