use ndarray::{Array, Ix1, Ix2, ScalarOperand, ShapeError};
use ndarray_rand::RandomExt;
use num_traits::{Float, FromPrimitive, NumAssign, ToPrimitive};
use rand::distributions::{Bernoulli, StandardNormal};

use crate::stats_util::{sum_of_squares, mean};
use bio_file_reader::plink_bed::MatrixIR;

pub fn matrix_ir_to_ndarray<T>(matrix_ir: MatrixIR<T>) -> Result<Array<T, Ix2>, ShapeError> {
    Array::from_shape_vec((matrix_ir.num_rows, matrix_ir.num_columns), matrix_ir.data)
}

pub fn generate_plus_minus_one_bernoulli_matrix(num_rows: usize, num_cols: usize) -> Array<f32, Ix2> {
    Array::random((num_rows, num_cols), Bernoulli::new(0.5)).mapv(|e| (e as i32 * 2 - 1) as f32)
}

pub fn generate_standard_normal_matrix(num_rows: usize, num_cols: usize) -> Array<f32, Ix2> {
    Array::random((num_rows, num_cols), StandardNormal).mapv(|e| e as f32)
}

/// `ddof`: delta degrees of freedom, where the denominator will be `N - ddof`,
/// where `N` is the number of elements per row
pub fn normalize_matrix_row_wise<A>(mut matrix: Array<A, Ix2>, ddof: usize) -> Array<A, Ix2>
    where A: ToPrimitive + FromPrimitive + NumAssign + Float + ScalarOperand {
    let (_num_rows, num_cols) = matrix.dim();
    let ones = Array::from_shape_vec((num_cols, 1), vec![A::one(); num_cols]).unwrap();

    // mean center
    let mean_vec = matrix.dot(&ones) / A::from(num_cols).unwrap();
    matrix -= &mean_vec;

    // now that each row has zero mean, so we can just use the sum of squares
    let denominator = A::from(num_cols - ddof).unwrap();
    for mut row in matrix.genrows_mut() {
        row /= (A::from(sum_of_squares(row.iter())).unwrap() / denominator).sqrt();
    };
    matrix
}

pub fn mean_center_vector<A>(mut vector: Array<A, Ix1>) -> Array<A, Ix1>
    where A: ToPrimitive + FromPrimitive + NumAssign + Float + ScalarOperand {
    vector -= A::from(mean(vector.iter())).unwrap();
    vector
}

#[cfg(test)]
mod tests {
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use rand::distributions::Uniform;

    use crate::stats_util::{mean, std};

    use super::{mean_center_vector, normalize_matrix_row_wise};

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

    #[test]
    fn test_mean_center_vector() {
        let size = 100;
        let mut vec = Array::random(size, Uniform::new(-10f32, 50f32));
        vec = mean_center_vector(vec);
        assert!(mean(vec.iter()).abs() < 1e-6);
    }
}
