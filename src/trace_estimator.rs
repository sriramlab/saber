use analytic::set::ordered_integer_set::OrderedIntegerSet;
use analytic::set::traits::Finite;
use analytic::stats::{n_choose_2, sum_f32, sum_of_squares, sum_of_squares_f32};
use biofile::plink_bed::{PlinkBed, PlinkSnpType};
use ndarray::{Array, Axis, Ix1, Ix2};
use ndarray_parallel::prelude::*;
use rayon::prelude::*;

use crate::matrix_ops::{DEFAULT_NUM_SNPS_PER_CHUNK, normalized_g_dot_matrix,
                        normalized_g_dot_rand, normalized_g_transpose_dot_matrix,
};
use crate::util::matrix_util::{
    generate_plus_minus_one_bernoulli_matrix, normalize_matrix_columns_inplace,
};

/// geno_bed has shape num_people x num_snps
pub fn estimate_tr_kk(
    geno_bed: &mut PlinkBed,
    snp_range: Option<OrderedIntegerSet<usize>>,
    num_random_vecs: usize,
    num_snps_per_chunk: Option<usize>,
) -> f64 {
    let chunk_size = num_snps_per_chunk.unwrap_or(DEFAULT_NUM_SNPS_PER_CHUNK);

    let num_people = geno_bed.num_people;
    let num_snps = match &snp_range {
        Some(range) => range.size(),
        None => geno_bed.total_num_snps(),
    };
    let rand_mat = generate_plus_minus_one_bernoulli_matrix(num_people, num_random_vecs);
    let xxz_arr: Vec<f32> = geno_bed
        .col_chunk_iter(chunk_size, snp_range, PlinkSnpType::Additive)
        .into_par_iter()
        .fold(|| vec![0f32; num_people * num_random_vecs], |mut acc, mut snp_chunk| {
            normalize_matrix_columns_inplace(&mut snp_chunk, 0);
            for (i, val) in snp_chunk.dot(&snp_chunk.t().dot(&rand_mat)).as_slice().unwrap().into_iter().enumerate() {
                acc[i] += val;
            }
            acc
        })
        .reduce(|| vec![0f32; num_people * num_random_vecs], |mut a, b| {
            for (i, val) in b.iter().enumerate() {
                a[i] += val;
            }
            a
        });

    sum_of_squares_f32(xxz_arr.iter()) as f64 / (num_snps * num_snps * num_random_vecs) as f64
}

pub fn estimate_tr_ki_kj(
    geno_bed: &mut PlinkBed,
    snp_range_i: Option<OrderedIntegerSet<usize>>,
    snp_range_j: Option<OrderedIntegerSet<usize>>,
    snp_mean_i: &Array<f32, Ix1>,
    snp_std_i: &Array<f32, Ix1>,
    snp_mean_j: &Array<f32, Ix1>,
    snp_std_j: &Array<f32, Ix1>,
    precomputed_normalized_g_j_dot_rand: Option<&Array<f32, Ix2>>,
    num_random_vecs: usize,
    num_snps_per_chunk: Option<usize>,
) -> f64 {
    let chunk_size = num_snps_per_chunk.unwrap_or(DEFAULT_NUM_SNPS_PER_CHUNK);

    let num_snps_i = match &snp_range_i {
        Some(range) => range.size(),
        None => geno_bed.total_num_snps(),
    };
    let num_snps_j = match &snp_range_j {
        Some(range) => range.size(),
        None => geno_bed.total_num_snps(),
    };

    let gj_z = match precomputed_normalized_g_j_dot_rand {
        Some(arr) => arr.to_owned(),
        None => normalized_g_dot_rand(geno_bed, snp_range_j, snp_mean_j, snp_std_j, num_random_vecs, Some(chunk_size)),
    };
    let gj_z_col_sum = {
        let mut col_sums = Vec::new();
        gj_z.axis_iter(Axis(1)).into_par_iter().map(|col| sum_f32(col.iter())).collect_into_vec(&mut col_sums);
        Array::from_shape_vec(num_random_vecs, col_sums).unwrap()
    };
    let ssq = geno_bed
        .col_chunk_iter(chunk_size, snp_range_i, PlinkSnpType::Additive)
        .into_par_iter()
        .enumerate()
        .fold_with(0f32, |mut acc, (chunk_index, snp_chunk)| {
            let arr = snp_chunk.t().dot(&gj_z).as_slice().unwrap().to_owned();
            for local_snp_index in 0..snp_chunk.dim().1 {
                let offset = local_snp_index * num_random_vecs;
                let m = snp_mean_i[chunk_index * chunk_size + local_snp_index];
                let s = snp_std_i[chunk_index * chunk_size + local_snp_index];
                for j in 0..num_random_vecs {
                    let x = (arr[offset + j] - m * gj_z_col_sum[j]) / s;
                    acc += x * x;
                }
            }
            acc
        })
        .sum::<f32>();
    ssq as f64 / (num_snps_i * num_snps_j * num_random_vecs) as f64
}

pub fn estimate_tr_k(
    geno_bed: &mut PlinkBed,
    snp_range: Option<OrderedIntegerSet<usize>>,
    num_random_vecs: usize,
    num_snps_per_chunk: Option<usize>,
) -> f64 {
    let chunk_size = num_snps_per_chunk.unwrap_or(DEFAULT_NUM_SNPS_PER_CHUNK);

    let num_people = geno_bed.num_people;
    let num_snps = match &snp_range {
        Some(range) => range.size(),
        None => geno_bed.total_num_snps(),
    };
    let rand_mat = generate_plus_minus_one_bernoulli_matrix(num_people, num_random_vecs);
    let sum_of_squares: f64 = geno_bed
        .col_chunk_iter(chunk_size, snp_range, PlinkSnpType::Additive)
        .into_par_iter()
        .fold_with(0f64, |mut acc, mut snp_chunk| {
            normalize_matrix_columns_inplace(&mut snp_chunk, 0);
            acc += sum_of_squares_f32(snp_chunk.t().dot(&rand_mat).as_slice().unwrap().into_iter()) as f64;
            acc
        }).sum();
    sum_of_squares / (num_snps * num_random_vecs) as f64
}

pub fn estimate_tr_k_gxg_k(
    geno_arr: &mut PlinkBed,
    le_snps_arr: &Array<f32, Ix2>,
    num_random_vecs: usize,
    num_snps_per_chunk: Option<usize>,
) -> f64 {
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

    let chunk_size = num_snps_per_chunk.unwrap_or(DEFAULT_NUM_SNPS_PER_CHUNK);
    let ssq = geno_arr
        .col_chunk_iter(chunk_size, None, PlinkSnpType::Additive)
        .into_par_iter()
        .fold_with(0f32, |mut acc, mut snp_chunk| {
            normalize_matrix_columns_inplace(&mut snp_chunk, 0);
            acc += sum_of_squares_f32(snp_chunk.t().dot(&corrected).as_slice().unwrap().iter());
            acc
        })
        .reduce(|| 0f32, |a, b| {
            a + b
        });
    (ssq / (geno_arr.total_num_snps() * n_choose_2(le_snps_arr.dim().1) * num_random_vecs) as f32) as f64

//    let gc = geno_arr.t().dot(&corrected);
//    let mut sums = Vec::new();
//    gc.axis_iter(Axis(1))
//      .into_par_iter()
//      .map(|col| sum_of_squares_f32(col.iter()))
//      .collect_into_vec(&mut sums);
//    (sums.into_iter().sum::<f32>() / (geno_arr.dim().1 * n_choose_2(le_snps_arr.dim().1) * num_random_vecs) as f32) as f64
}

// TODO: test
pub fn estimate_tr_gxg_ki_gxg_kj(
    arr_i: &Array<f32, Ix2>,
    arr_j: &Array<f32, Ix2>,
    num_random_vecs: usize,
) -> f64 {
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

pub fn estimate_gxg_gram_trace(
    geno_arr: &Array<f32, Ix2>,
    num_random_vecs: usize,
) -> Result<f64, String> {
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

pub fn estimate_gxg_kk_trace(
    gxg_basis: &Array<f32, Ix2>,
    num_random_vecs: usize,
) -> Result<f64, String> {
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

pub fn estimate_gxg_dot_y_norm_sq(
    gxg_basis_arr: &Array<f32, Ix2>,
    y: &Array<f32, Ix1>,
    num_random_vecs: usize,
) -> f64 {
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

pub fn estimate_gxg_dot_y_norm_sq_from_basis_bed(
    gxg_basis_bed: &PlinkBed,
    snp_range: Option<OrderedIntegerSet<usize>>,
    snp_mean: &Array<f32, Ix1>,
    snp_std: &Array<f32, Ix1>,
    y: &Array<f32, Ix1>,
    num_random_vecs: usize,
) -> f64 {
    let num_cols = match &snp_range {
        Some(range) => range.size(),
        None => gxg_basis_bed.total_num_snps(),
    };
    let ssq_of_hi_hi = gxg_basis_bed
        .col_chunk_iter(DEFAULT_NUM_SNPS_PER_CHUNK, snp_range.clone(), PlinkSnpType::Additive)
        .into_par_iter()
        .fold(|| 0f32, |acc, mut snp_chunk| {
            normalize_matrix_columns_inplace(&mut snp_chunk, 0);
            let gg_sq_dot_y = ((&snp_chunk) * (&snp_chunk)).t().dot(y);
            acc + sum_of_squares_f32(gg_sq_dot_y.iter())
        })
        .sum::<f32>();

    let y_scaled_basis_dot_rand_vecs = normalized_g_dot_matrix(
        gxg_basis_bed,
        snp_range.clone(),
        snp_mean,
        snp_std,
        &generate_plus_minus_one_bernoulli_matrix(num_cols, num_random_vecs),
        Some(y),
        None,
    );
    let mut hhz = normalized_g_transpose_dot_matrix(
        gxg_basis_bed,
        snp_range,
        snp_mean,
        snp_std,
        &y_scaled_basis_dot_rand_vecs,
        None,
        None,
    );
    hhz.par_iter_mut().for_each(|x| *x = (*x) * (*x));
    ((hhz.sum() / num_random_vecs as f32 - ssq_of_hi_hi) / 2.) as f64
}

pub fn get_gxg_dot_y_norm_sq_from_basis_bed(
    gxg_basis_bed: &PlinkBed,
    snp_range: Option<OrderedIntegerSet<usize>>,
    snp_mean: &Array<f32, Ix1>,
    snp_std: &Array<f32, Ix1>,
    y: &Array<f32, Ix1>,
) -> f64 {
    let ssq_of_hi_hi = gxg_basis_bed
        .col_chunk_iter(DEFAULT_NUM_SNPS_PER_CHUNK, snp_range.clone(), PlinkSnpType::Additive)
        .into_par_iter()
        .fold(|| 0f32, |acc, mut snp_chunk| {
            normalize_matrix_columns_inplace(&mut snp_chunk, 0);
            let gg_sq_dot_y = ((&snp_chunk) * (&snp_chunk)).t().dot(y);
            acc + sum_of_squares_f32(gg_sq_dot_y.iter())
        })
        .sum::<f32>();

    let mut rhs_matrix = gxg_basis_bed
        .get_genotype_matrix(snp_range.clone(), PlinkSnpType::Additive)
        .unwrap();
    normalize_matrix_columns_inplace(&mut rhs_matrix, 0);
    let hh = normalized_g_transpose_dot_matrix(
        &gxg_basis_bed,
        snp_range,
        &snp_mean,
        &snp_std,
        &rhs_matrix,
        Some(&y),
        None,
    );
    ((sum_of_squares_f32(hh.iter()) - ssq_of_hi_hi) / 2.) as f64
}

pub fn estimate_inter_gxg_dot_y_norm_sq_from_basis_bed(
    gxg_basis_bed: &PlinkBed,
    snp_range_1: Option<OrderedIntegerSet<usize>>,
    snp_range_2: Option<OrderedIntegerSet<usize>>,
    snp_mean_1: &Array<f32, Ix1>,
    snp_std_1: &Array<f32, Ix1>,
    snp_mean_2: &Array<f32, Ix1>,
    snp_std_2: &Array<f32, Ix1>,
    y: &Array<f32, Ix1>,
    num_random_vecs: usize,
) -> f64 {
    let num_snps_1 = match &snp_range_1 {
        Some(range) => range.size(),
        None => gxg_basis_bed.total_num_snps(),
    };
    let y_scaled_basis_dot_rand_vecs = normalized_g_dot_matrix(
        gxg_basis_bed,
        snp_range_1,
        snp_mean_1,
        snp_std_1,
        &generate_plus_minus_one_bernoulli_matrix(num_snps_1, num_random_vecs),
        Some(y),
        None,
    );
    let hhz = normalized_g_transpose_dot_matrix(
        gxg_basis_bed,
        snp_range_2,
        snp_mean_2,
        snp_std_2,
        &y_scaled_basis_dot_rand_vecs,
        None,
        None,
    );
    sum_of_squares_f32(hhz.iter()) as f64 / num_random_vecs as f64
}

/*
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
}
*/
