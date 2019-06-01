extern crate ndarray_parallel;

use ndarray_parallel::prelude::*;

use ndarray::{Array, Axis, Ix1, Ix2};
use crate::matrix_util::generate_plus_minus_one_bernoulli_matrix;
use crate::stats_util::{sum_of_squares, sum_of_squares_f32};

/// geno_arr has shape num_people x num_snps
pub fn estimate_tr_k(geno_arr: &Array<f32, Ix2>, num_random_vecs: usize) -> f64 {
    let (num_people, num_snps) = geno_arr.dim();
    let rand_mat = generate_plus_minus_one_bernoulli_matrix(num_people, num_random_vecs);
    let xz_arr = geno_arr.t().dot(&rand_mat);
    let xxz = geno_arr.dot(&xz_arr);

    let mut sums = Vec::new();
    xxz.axis_iter(Axis(1))
       .into_par_iter()
       .map(|col| sum_of_squares_f32(col.iter()))
       .collect_into_vec(&mut sums);

    (sums.into_iter().sum::<f32>() / (num_snps * num_snps * num_random_vecs) as f32) as f64
//    sum_of_squares(xxz.iter()) / (num_snps * num_snps * num_random_vecs) as f64
}

pub fn estimate_tr_k_gxg_k(geno_arr: &Array<f32, Ix2>, independent_snps_arr: &Array<f32, Ix2>, num_random_vecs: usize) -> f64 {
    let u_arr = generate_plus_minus_one_bernoulli_matrix(independent_snps_arr.dim().1, num_random_vecs);
    let mut sums = Vec::new();
    independent_snps_arr.axis_iter(Axis(0))
                        .into_par_iter()
                        .map(|row| sum_of_squares_f32(row.iter()))
                        .collect_into_vec(&mut sums);
    let geno_ssq = Array::from_shape_vec((independent_snps_arr.dim().0, 1), sums).unwrap();
    let mut squashed = independent_snps_arr.dot(&u_arr);
    squashed.par_iter_mut().for_each(|x| *x = (*x) * (*x));
    let corrected = (squashed - geno_ssq) / 2.;

    let gc = geno_arr.t().dot(&corrected);
    let mut sums = Vec::new();
    gc.axis_iter(Axis(1))
      .into_par_iter()
      .map(|col| sum_of_squares_f32(col.iter()))
      .collect_into_vec(&mut sums);
    (sums.into_iter().sum::<f32>() / num_random_vecs as f32) as f64
}

pub fn estimate_gxg_gram_trace(geno_arr: &Array<f32, Ix2>, num_random_vecs: usize) -> Result<f64, String> {
    let (_num_rows, num_cols) = geno_arr.dim();
    let u_arr = generate_plus_minus_one_bernoulli_matrix(num_cols, num_random_vecs);
    let ones = Array::<f32, Ix1>::ones(num_cols);

    let geno_ssq = (geno_arr * geno_arr).dot(&ones);
    let mut squashed = geno_arr.dot(&u_arr);
    squashed.par_iter_mut().for_each(|x| *x = (*x) * (*x));

    let mut sums = Vec::new();
    squashed.axis_iter(Axis(1))
            .into_par_iter()
            .map(|col| {
                let uugg = (&col - &geno_ssq) / 2.;
                sum_of_squares(uugg.iter())
            })
            .collect_into_vec(&mut sums);
    Ok(sums.into_iter().sum::<f64>() / num_random_vecs as f64)

//    let mut sum = 0f64;
//    for col in squashed_squared.gencolumns() {
//        let uugg = (&col - &geno_ssq) / 2.;
//        sum += sum_of_squares(uugg.iter());
//    }
//    let avg = sum / num_random_vecs as f64;
//    Ok(avg)
}

pub fn estimate_gxg_kk_trace(geno_arr: &Array<f32, Ix2>, num_random_vecs: usize) -> Result<f64, String> {
    let (_num_rows, num_cols) = geno_arr.dim();
    let u_arr = generate_plus_minus_one_bernoulli_matrix(num_cols, num_random_vecs);
    let ones = Array::<f32, Ix1>::ones(num_cols);

    let gg_sq = geno_arr * geno_arr;
    let geno_ssq = gg_sq.dot(&ones);
    let mut squashed = geno_arr.dot(&u_arr);
    squashed.par_iter_mut().for_each(|x| *x = (*x) * (*x));
    let num_rand_z_vecs = 100;
    println!("num_rand_z_vecs: {}\nnum_random_vecs: {}", num_rand_z_vecs, num_random_vecs);

    let mut sums = Vec::new();
    squashed.axis_iter(Axis(1))
            .into_par_iter()
            .map(|col| {
                let uugg_sum = (&col - &geno_ssq) / 2.;
                let wg = &geno_arr.t() * &uugg_sum;
                let s = (&gg_sq.t() * &uugg_sum).sum();
                let rand_vecs = generate_plus_minus_one_bernoulli_matrix(num_cols, num_rand_z_vecs);
                let geno_arr_dot_rand_vecs = geno_arr.dot(&rand_vecs);
                let ggz = wg.dot(&geno_arr_dot_rand_vecs);
                (((&ggz * &ggz).sum() / num_rand_z_vecs as f32 - s) / 2.) as f64
            })
            .collect_into_vec(&mut sums);
    Ok(sums.into_iter().sum::<f64>() / num_random_vecs as f64)

//    let mut sum = 0f64;
//    for col in squashed_squared.gencolumns() {
//        let uugg_sum = (&col - &geno_ssq) / 2.;
//        let wg = &geno_arr.t() * &uugg_sum;
//        let s = (&gg_sq.t() * &uugg_sum).sum();
//        let rand_vecs = generate_plus_minus_one_bernoulli_matrix(num_cols, num_rand_z_vecs);
//        let geno_arr_dot_rand_vecs = geno_arr.dot(&rand_vecs);
//        let ggz = wg.dot(&geno_arr_dot_rand_vecs);
//        sum += (((&ggz * &ggz).sum() / num_rand_z_vecs as f32 - s) / 2.) as f64;
//    }
//    let avg = sum / num_random_vecs as f64;
//    Ok(avg)
}

pub fn estimate_gxg_dot_y_norm_sq(gxg_basis_arr: &Array<f32, Ix2>, y: &Array<f32, Ix1>, num_random_vecs: usize) -> f64 {
    let (_num_rows, num_cols) = gxg_basis_arr.dim();
    println!("estimate_gxg_dot_y using {} random vectors", num_random_vecs);
    let wg = &gxg_basis_arr.t() * y;
    let gg_sq = gxg_basis_arr * gxg_basis_arr;
    let s = (&gg_sq.t() * y).sum();
    let rand_vecs = generate_plus_minus_one_bernoulli_matrix(num_cols, num_random_vecs);
    let geno_arr_dot_rand_vecs = gxg_basis_arr.dot(&rand_vecs);
    let mut ggz = wg.dot(&geno_arr_dot_rand_vecs);
    ggz.par_iter_mut().for_each(|x| *x = (*x) * (*x));
    ((ggz.sum() / num_random_vecs as f32 - s) / 2.) as f64
}

pub fn estimate_gxg1_k_gxg2_k_trace(geno_arr: &Array<f32, Ix2>, geno_arr2: &Array<f32, Ix2>,
    num_random_vecs: usize) -> Result<f64, String> {
    let (_num_rows, num_cols) = geno_arr.dim();
    let (_num_rows2, num_cols2) = geno_arr2.dim();
    let u_arr = generate_plus_minus_one_bernoulli_matrix(num_cols, num_random_vecs);
    let ones = Array::<f32, Ix1>::ones(num_cols);

    let gg_sq = geno_arr * geno_arr;
    let geno_ssq = gg_sq.dot(&ones);
    let squashed = geno_arr.dot(&u_arr);
    let squashed_squared = &squashed * &squashed;
    let mut sum = 0f64;
    let num_rand_z_vecs = 100;
    println!("num_rand_z_vecs: {}\nnum_random_vecs: {}", num_rand_z_vecs, num_random_vecs);

    let gg2_sq = geno_arr2 * geno_arr2;
    let mut i = 0;
    for col in squashed_squared.gencolumns() {
        if i % 100 == 0 {
            println!("{} / {}", i + 1, num_random_vecs);
        }
        i += 1;
        let uugg_sum = (&col - &geno_ssq) / 2.;
        let wg = &geno_arr2.t() * &uugg_sum;
        let s = (&gg2_sq.t() * &uugg_sum).sum();
        let rand_vecs = generate_plus_minus_one_bernoulli_matrix(num_cols2, num_rand_z_vecs);
        let geno_arr_dot_rand_vecs = geno_arr2.dot(&rand_vecs);
        let ggz = wg.dot(&geno_arr_dot_rand_vecs);
        sum += (((&ggz * &ggz).sum() / num_rand_z_vecs as f32 - s) / 2.) as f64;
    }
    let avg = sum / num_random_vecs as f64;
    Ok(avg)
}
