use ndarray::{Array, Axis, Ix1, Ix2};
use ndarray_parallel::prelude::*;

use crate::util::matrix_util::generate_plus_minus_one_bernoulli_matrix;
use crate::util::stats_util::{n_choose_2, sum_f32, sum_of_squares, sum_of_squares_f32};

/// geno_arr has shape num_people x num_snps
pub fn estimate_tr_kk(geno_arr: &Array<f32, Ix2>, num_random_vecs: usize) -> f64 {
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

pub fn estimate_tr_k_gxg_k(geno_arr: &Array<f32, Ix2>, le_snps_arr: &Array<f32, Ix2>, num_random_vecs: usize) -> f64 {
    let u_arr = generate_plus_minus_one_bernoulli_matrix(le_snps_arr.dim().1, num_random_vecs);
    let mut sums = Vec::new();
    le_snps_arr.axis_iter(Axis(0))
               .into_par_iter()
               .map(|row| sum_of_squares_f32(row.iter()))
               .collect_into_vec(&mut sums);
    let geno_ssq = Array::from_shape_vec((le_snps_arr.dim().0, 1), sums).unwrap();
    let mut squashed = le_snps_arr.dot(&u_arr);
    squashed.par_iter_mut().for_each(|x| *x = (*x) * (*x));
    let corrected = (squashed - geno_ssq) / 2.;

    let gc = geno_arr.t().dot(&corrected);
    let mut sums = Vec::new();
    gc.axis_iter(Axis(1))
      .into_par_iter()
      .map(|col| sum_of_squares_f32(col.iter()))
      .collect_into_vec(&mut sums);
    (sums.into_iter().sum::<f32>() / (geno_arr.dim().1 * n_choose_2(le_snps_arr.dim().1) * num_random_vecs) as f32) as f64
}

// TODO: test
pub fn estimate_tr_gxg_ki_gxg_kj(arr_i: &Array<f32, Ix2>, arr_j: &Array<f32, Ix2>, num_random_vecs: usize) -> f64 {
    let u_arr = generate_plus_minus_one_bernoulli_matrix(arr_i.dim().1, num_random_vecs);
    let mut arr_i_row_sq_sums = Vec::new();
    arr_i.axis_iter(Axis(0))
         .into_par_iter()
         .map(|row| sum_of_squares_f32(row.iter()))
         .collect_into_vec(&mut arr_i_row_sq_sums);
    let mut arr_i_squashed = arr_i.dot(&u_arr);
    arr_i_squashed.par_iter_mut().for_each(|x| *x = (*x) * (*x));
    let arr_i_uugg_sums = (arr_i_squashed - Array::from_shape_vec((arr_i.dim().0, 1), arr_i_row_sq_sums).unwrap()) / 2.;

    let arr_j_sq = arr_j * arr_j;
    let num_rand_z_vecs = 100;
    let mut sums = Vec::new();
    arr_i_uugg_sums.axis_iter(Axis(1))
                   .into_par_iter()
                   .map(|uugg_sum| {
                       let rand_vecs = generate_plus_minus_one_bernoulli_matrix(arr_j.dim().1, num_rand_z_vecs);
                       let arr_j_dot_rand_vecs = arr_j.dot(&rand_vecs);
                       let wg = &arr_j.t() * &uugg_sum;
                       let ggz = wg.dot(&arr_j_dot_rand_vecs);
                       let gg_sq_dot_y = arr_j_sq.t().dot(&uugg_sum);
                       let s = (&gg_sq_dot_y * &gg_sq_dot_y).sum();
                       (((&ggz * &ggz).sum() / num_rand_z_vecs as f32 - s) / 2.)
                   })
                   .collect_into_vec(&mut sums);
    (sums.into_iter().sum::<f32>() / (n_choose_2(arr_i.dim().1) * n_choose_2(arr_j.dim().1) * num_random_vecs) as f32) as f64
}

pub fn estimate_gxg_gram_trace(geno_arr: &Array<f32, Ix2>, num_random_vecs: usize) -> Result<f64, String> {
    let (_num_rows, num_cols) = geno_arr.dim();

    let mut row_sums = Vec::new();
    geno_arr.axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| sum_of_squares_f32(row.iter()))
            .collect_into_vec(&mut row_sums);
    let geno_ssq = Array::from_shape_vec((row_sums.len(), 1), row_sums).unwrap();

    let u_arr = generate_plus_minus_one_bernoulli_matrix(num_cols, num_random_vecs);
    let mut squashed = geno_arr.dot(&u_arr);
    squashed.par_iter_mut().for_each(|x| *x = (*x) * (*x));
    squashed = (squashed - &geno_ssq) / 2.;

    let mut sums = Vec::new();
    squashed.axis_iter(Axis(1))
            .into_par_iter()
            .map(|uugg| {
                sum_of_squares(uugg.iter())
            })
            .collect_into_vec(&mut sums);
    Ok(sums.into_iter().sum::<f64>() / num_random_vecs as f64)
}

pub fn estimate_gxg_kk_trace(gxg_basis: &Array<f32, Ix2>, num_random_vecs: usize) -> Result<f64, String> {
    let num_rand_z_vecs = 100;
    println!("estimate_gxg_kk_trace\nnum_random_vecs: {}\nnum_rand_z_vecs: {}", num_random_vecs, num_rand_z_vecs);
    let (_num_rows, num_le_snps) = gxg_basis.dim();
    let u_arr = generate_plus_minus_one_bernoulli_matrix(num_le_snps, num_random_vecs);

    let gxg_basis_sq = gxg_basis * gxg_basis;
    let mut row_sums = Vec::new();
    gxg_basis_sq.axis_iter(Axis(0))
                .into_par_iter()
                .map(|row| sum_f32(row.iter()))
                .collect_into_vec(&mut row_sums);
    let geno_ssq = Array::from_shape_vec((row_sums.len(), 1), row_sums).unwrap();

    let mut uugg_sum_matrix = gxg_basis.dot(&u_arr);
    uugg_sum_matrix.par_iter_mut().for_each(|x| *x = (*x) * (*x));
    uugg_sum_matrix = (uugg_sum_matrix - &geno_ssq) / 2.;

    let mut sums = Vec::new();
    uugg_sum_matrix.axis_iter(Axis(1))
                   .into_par_iter()
                   .map(|uugg_sum| {
                       let rand_vecs = generate_plus_minus_one_bernoulli_matrix(num_le_snps, num_rand_z_vecs);
                       let geno_arr_dot_rand_vecs = gxg_basis.dot(&rand_vecs);
                       let wg = &gxg_basis.t() * &uugg_sum;
                       let ggz = wg.dot(&geno_arr_dot_rand_vecs);
                       let gg_sq_dot_y = gxg_basis_sq.t().dot(&uugg_sum);
                       let s = (&gg_sq_dot_y * &gg_sq_dot_y).sum();
                       (((&ggz * &ggz).sum() / num_rand_z_vecs as f32 - s) / 2.) as f64
                   })
                   .collect_into_vec(&mut sums);
    let mm = n_choose_2(num_le_snps) as f64;
    Ok(sums.into_iter().sum::<f64>() / (num_random_vecs as f64 * mm * mm))

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
    let gg_sq_dot_y = (gxg_basis_arr * gxg_basis_arr).t().dot(y);
    let s = (&gg_sq_dot_y * &gg_sq_dot_y).sum();
    let rand_vecs = generate_plus_minus_one_bernoulli_matrix(num_cols, num_random_vecs);
    let geno_arr_dot_rand_vecs = gxg_basis_arr.dot(&rand_vecs);
    let wg = &gxg_basis_arr.t() * y;
    let mut ggz = wg.dot(&geno_arr_dot_rand_vecs);
    ggz.par_iter_mut().for_each(|x| *x = (*x) * (*x));
    ((ggz.sum() / num_random_vecs as f32 - s) / 2.) as f64
}
