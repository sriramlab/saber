use bio_file_reader::plink_bed::MatrixIR;
use ndarray::{Array, Ix1, Ix2, Axis};
use crate::matrix_util::generate_plus_minus_one_bernoulli_matrix;
use crate::stats_util::sum_of_squares;
use ndarray::prelude::aview2;
use crate::timer::Timer;

pub fn estimate_tr_k_gxg_k(geno_arr: &Array<f32, Ix2>, independent_snps_arr: &Array<f32, Ix2>, num_random_vecs: usize) -> f64 {
    let u_arr = generate_plus_minus_one_bernoulli_matrix(independent_snps_arr.dim().1, num_random_vecs);
    let ones = Array::<f32, Ix1>::ones(independent_snps_arr.dim().1);
    let geno_ssq = (independent_snps_arr * independent_snps_arr).dot(&ones);
    let squashed = independent_snps_arr.dot(&u_arr);
    let squashed_squared = &squashed * &squashed;

    let v = Vec::from(geno_ssq.as_slice().unwrap());
    let x = Array::from_shape_vec((geno_arr.dim().0, 1), v).unwrap();
    let corrected = (squashed_squared - x) / 2.;
    sum_of_squares(geno_arr.t().dot(&corrected).iter()) / num_random_vecs as f64
}

pub fn estimate_gxg_gram_trace(geno_arr: &Array<f32, Ix2>, num_random_vecs: usize) -> Result<f64, String> {
    let (num_rows, num_cols) = geno_arr.dim();
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

pub fn estimate_kk_trace(geno_arr: &Array<f32, Ix2>, num_random_vecs: usize) -> Result<f64, String> {
    let (num_rows, num_cols) = geno_arr.dim();
    let u_arr = generate_plus_minus_one_bernoulli_matrix(num_cols, num_random_vecs);
    let ones = Array::<f32, Ix1>::ones(num_cols);

    let gg_sq = geno_arr * geno_arr;
    let geno_ssq = gg_sq.dot(&ones);
    let squashed = geno_arr.dot(&u_arr);
    let squashed_squared = &squashed * &squashed;
    let mut sum = 0f64;
    let num_rand_z_vecs = 100;
    println!("num_rand_z_vecs: {}\nnum_random_vecs: {}", num_rand_z_vecs, num_random_vecs);

    let mut i = 0;
    for col in squashed_squared.gencolumns() {
        if i % 100 == 0 {
            println!("{} / {}", i + 1, num_random_vecs);
        }
        i += 1;
        let uugg_sum = (&col - &geno_ssq) / 2.;
        let wg = &geno_arr.t() * &uugg_sum;
        let s = (&gg_sq.t() * &uugg_sum).sum();
        let rand_vecs = generate_plus_minus_one_bernoulli_matrix(num_cols, num_rand_z_vecs);
        let geno_arr_dot_rand_vecs = geno_arr.dot(&rand_vecs);
        let ggz = wg.dot(&geno_arr_dot_rand_vecs);
        sum += (((&ggz * &ggz).sum() / num_rand_z_vecs as f32 - s) / 2.) as f64;
    }
    let avg = sum / num_random_vecs as f64;
    Ok(avg)
}

pub fn estimate_gxg_dot_y_norm_sq(geno_arr: &Array<f32, Ix2>, y: &Array<f32, Ix1>, num_random_vecs: usize) -> f64 {
    let (num_rows, num_cols) = geno_arr.dim();
    println!("estimate_gxg_dot_y using {} random vectors", num_random_vecs);
    let wg = &geno_arr.t() * y;
    let gg_sq = geno_arr * geno_arr;
    let s = (&gg_sq.t() * y).sum();
    let rand_vecs = generate_plus_minus_one_bernoulli_matrix(num_cols, num_random_vecs);
    let geno_arr_dot_rand_vecs = geno_arr.dot(&rand_vecs);
    let ggz = wg.dot(&geno_arr_dot_rand_vecs);
    (((&ggz * &ggz).sum() / num_random_vecs as f32 - s) / 2.) as f64
}
