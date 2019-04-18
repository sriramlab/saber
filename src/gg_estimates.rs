use bio_file_reader::plink_bed::MatrixIR;
use ndarray::{Array, Ix1, Ix2};
use rand::distributions::{Bernoulli, StandardNormal};
use crate::matrix_util::generate_plus_minus_one_bernoulli_matrix;
use crate::stats_util::sum_of_squares;

pub fn estimate_gxg_trace(geno_arr: &Array<f32, Ix2>, num_random_vecs: usize) -> Result<f64, String> {
    let (num_rows, num_cols) = geno_arr.dim();
    let num_cols_gxg = num_cols * (num_cols - 1) / 2;
    let u_arr = generate_plus_minus_one_bernoulli_matrix(num_cols, num_random_vecs);
    let ones = Array::<f32, Ix1>::ones(num_cols);

    let geno_ssq = (geno_arr * geno_arr).dot(&ones);
    let squashed = geno_arr.dot(&u_arr);
    let squashed_squared = &squashed * &squashed;
    let mut sum = 0f64;
    for col in squashed_squared.gencolumns() {
        let uugg = (&col - &geno_ssq) / 2.;
        sum += sum_of_squares(uugg.iter());
    }
    let avg = sum / num_random_vecs as f64;
    Ok(avg)
}
