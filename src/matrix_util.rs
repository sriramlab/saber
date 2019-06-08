extern crate ndarray_parallel;

use ndarray_parallel::prelude::*;

use ndarray::{Array, Axis, Ix1, Ix2, ScalarOperand};
use ndarray_rand::RandomExt;
use num_traits::{Float, FromPrimitive, NumAssign, ToPrimitive};
use rand::distributions::{Bernoulli, StandardNormal};

use crate::stats_util::{mean, std};
use bio_file_reader::plink_bed::MatrixIR;

pub fn matrix_ir_to_ndarray<T>(matrix_ir: MatrixIR<T>) -> Result<Array<T, Ix2>, String> {
    match Array::from_shape_vec((matrix_ir.num_rows, matrix_ir.num_columns), matrix_ir.data) {
        Err(why) => Err(format!("{}", why)),
        Ok(arr) => Ok(arr)
    }
}

pub fn generate_plus_minus_one_bernoulli_matrix(num_rows: usize, num_cols: usize) -> Array<f32, Ix2> {
    Array::random((num_rows, num_cols), Bernoulli::new(0.5)).mapv(|e| (e as i32 * 2 - 1) as f32)
}

pub fn generate_standard_normal_matrix(num_rows: usize, num_cols: usize) -> Array<f32, Ix2> {
    Array::random((num_rows, num_cols), StandardNormal).mapv(|e| e as f32)
}

/// `ddof`: delta degrees of freedom, where the denominator will be `N - ddof`,
/// where `N` is the number of elements per row
pub fn normalize_matrix_row_wise_inplace<A>(mut matrix: Array<A, Ix2>, ddof: usize) -> Array<A, Ix2>
    where A: ToPrimitive + FromPrimitive + NumAssign + Float + ScalarOperand {
    let (_num_rows, num_cols) = matrix.dim();
    let ones = Array::from_shape_vec((num_cols, 1), vec![A::one(); num_cols]).unwrap();

    // mean center
    let mean_vec = matrix.dot(&ones) / A::from(num_cols).unwrap();
    matrix -= &mean_vec;

    // now that each row has zero mean, so we can just use the sum of squares
    let denominator = A::from(num_cols - ddof).unwrap();
    for mut row in matrix.genrows_mut() {
        let std = (A::from((&row * &row).sum()).unwrap() / denominator).sqrt();
        if std > A::zero() {
            row /= std;
        }
    };
    matrix
}

/// `ddof`: delta degrees of freedom, where the denominator will be `N - ddof`,
/// where `N` is the number of elements per row
pub fn normalize_matrix_columns_inplace<A>(matrix: &mut Array<A, Ix2>, ddof: usize)
    where A: ToPrimitive + FromPrimitive + NumAssign + Float + ScalarOperand + Send + Sync {
    let (num_rows, _num_cols) = matrix.dim();
    let num_rows_denom = A::from(num_rows).unwrap();
    let denominator = A::from(num_rows - ddof).unwrap();
    let zero = A::zero();
    matrix.axis_iter_mut(Axis(1))
          .into_par_iter()
          .for_each(|mut col| {
              col -= col.sum() / num_rows_denom;
              let std = ((&col * &col).sum() / denominator).sqrt();
              if std > zero {
                  col /= std;
              }
          });
}

pub fn normalize_vector_inplace<A>(vec: &mut Array<A, Ix1>, ddof: usize)
    where A: ToPrimitive + FromPrimitive + NumAssign + Float + ScalarOperand + Send + Sync {
    *vec -= A::from(mean(vec.iter())).unwrap();
    *vec /= A::from(std(vec.iter(), ddof)).unwrap();
}

pub fn mean_center_vector<A>(vector: &mut Array<A, Ix1>)
    where A: ToPrimitive + FromPrimitive + NumAssign + Float + ScalarOperand {
    *vector -= A::from(mean(vector.iter())).unwrap();
}

pub fn row_mean_vec<A, T>(matrix: &Array<A, Ix2>) -> Array<T, Ix1>
    where A: Copy + ToPrimitive + NumAssign, T: Float + FromPrimitive {
    let mut mean_vec = Vec::new();
    for row in matrix.genrows() {
        mean_vec.push(T::from(mean(row.iter())).unwrap());
    }
    Array::from_vec(mean_vec)
}

pub fn row_std_vec<A, T>(matrix: &Array<A, Ix2>, ddof: usize) -> Array<T, Ix1>
    where A: Copy + ToPrimitive + NumAssign, T: Float + FromPrimitive {
    let mut std_vec = Vec::new();
    for row in matrix.genrows() {
        std_vec.push(T::from(std(row.iter(), ddof)).unwrap());
    }
    Array::from_vec(std_vec)
}

pub fn get_correlation<A>(arr1: &Array<A, Ix1>, arr2: &Array<A, Ix1>) -> f64
    where A: Copy + ToPrimitive + FromPrimitive + NumAssign + ScalarOperand {
    let mut a = arr1.clone() - A::from_f64(mean(arr1.iter())).unwrap();
    a /= A::from_f64(std(arr1.iter(), 0)).unwrap();

    let mut b = arr2.clone() - A::from_f64(mean(arr2.iter())).unwrap();
    b /= A::from_f64(std(arr2.iter(), 0)).unwrap();

    a.dot(&b).to_f64().unwrap() / arr1.dim() as f64
}

#[cfg(test)]
mod tests {
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use rand::distributions::Uniform;

    use crate::stats_util::{mean, std};

    use super::{mean_center_vector, normalize_matrix_row_wise_inplace, normalize_matrix_columns_inplace,
                normalize_vector_inplace, get_correlation};

    #[test]
    fn test_normalize_matrix_row_wise() {
        let ddof = 1;
        let (num_rows, num_cols) = (50, 100);
        let mut matrix = Array::random((num_rows, num_cols), Uniform::new(-10f32, 50f32));
        matrix = normalize_matrix_row_wise_inplace(matrix, ddof);

        // check that the means are close to 0 and the standard deviations are close to 1
        for row in matrix.genrows() {
            assert!(mean(row.iter()).abs() < 1e-6);
            assert!((std(row.iter(), ddof) - 1.).abs() < 1e-6);
        }
    }

    #[test]
    fn test_normalize_matrix_columns() {
        let ddof = 1;
        let (num_rows, num_cols) = (50, 100);
        let mut matrix = Array::random((num_rows, num_cols), Uniform::new(-10f32, 50f32));
        normalize_matrix_columns_inplace(&mut matrix, ddof);

        // check that the means are close to 0 and the standard deviations are close to 1
        for col in matrix.gencolumns() {
            assert!(mean(col.iter()).abs() < 1e-6);
            assert!((std(col.iter(), ddof) - 1.).abs() < 1e-6);
        }
    }

    #[test]
    fn test_normalize_vector_inplace() {
        let num_elements = 1000;
        let ddof = 0;
        let mut vec = Array::random(num_elements, Uniform::new(-10f32, 50f32));
        assert!(mean(vec.iter()).abs() > 1e-3, "the randomly generated vector should have a large non-zero mean");
        assert!((std(vec.iter(), ddof) - 1.).abs() > 2., "the randomly generated vector should have a large std");
        normalize_vector_inplace(&mut vec, ddof);
        assert!(mean(vec.iter()).abs() < 1e-6);
        assert!((std(vec.iter(), ddof) - 1.).abs() < 1e-6);
    }

    #[test]
    fn test_mean_center_vector() {
        let size = 100;
        let mut vec = Array::random(size, Uniform::new(-10f32, 50f32));
        assert!(mean(vec.iter()).abs() > 1e-3, "the randomly generated vector should have a large non-zero mean");
        mean_center_vector(&mut vec);
        assert!(mean(vec.iter()).abs() < 1e-6);
    }

    #[test]
    fn test_get_correlation() {
        let size = 500;
        let v1 = Array::random(size, Uniform::new(-10f32, 50f32));
        let v1_clone = v1.clone();
        assert!((get_correlation(&v1, &v1_clone) - 1.).abs() < 1e-6);
    }
    // TODO: test row_mean_vec and row_std_vec
}
