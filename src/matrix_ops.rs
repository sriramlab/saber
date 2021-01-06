use std::marker::Sync;

use biofile::plink_bed::PlinkBed;
use math::{
    set::{ordered_integer_set::OrderedIntegerSet, traits::Finite},
    stats::{
        mean, standard_deviation, sum_f32, sum_of_fourth_power_f32,
        sum_of_squares, sum_of_squares_f32,
    },
};
use ndarray::{iter, s, Array, Axis, Dim, Ix1, Ix2};
use ndarray_parallel::prelude::*;
use rayon::prelude::*;

use crate::util::matrix_util::{
    generate_plus_minus_one_bernoulli_matrix, normalize_matrix_columns_inplace,
};

pub const DEFAULT_NUM_SNPS_PER_CHUNK: usize = 25;

pub fn column_normalized_sum_of_row_wise_fourth_moment(
    bed: &PlinkBed,
    snp_range: Option<OrderedIntegerSet<usize>>,
    num_snps_per_chunk: Option<usize>,
) -> Array<f32, Ix1> {
    column_normalized_row_wise_sigma(
        bed,
        snp_range,
        |i| sum_of_fourth_power_f32(i),
        num_snps_per_chunk,
    )
}

pub fn column_normalized_row_ssq(
    bed: &PlinkBed,
    snp_range: Option<OrderedIntegerSet<usize>>,
    num_snps_per_chunk: Option<usize>,
) -> Array<f32, Ix1> {
    column_normalized_row_wise_sigma(
        bed,
        snp_range,
        |i| sum_of_squares_f32(i),
        num_snps_per_chunk,
    )
}

pub fn column_normalized_row_wise_sigma<F>(
    bed: &PlinkBed,
    snp_range: Option<OrderedIntegerSet<usize>>,
    op: F,
    num_snps_per_chunk: Option<usize>,
) -> Array<f32, Ix1>
where
    F: Fn(iter::Iter<'_, f32, Dim<[usize; 1]>>) -> f32 + Sync, {
    let chunk_size = num_snps_per_chunk.unwrap_or(DEFAULT_NUM_SNPS_PER_CHUNK);
    let num_people = bed.num_people;
    let sigma_vec = bed
        .col_chunk_iter(chunk_size, snp_range)
        .into_par_iter()
        .fold(
            || vec![0f32; num_people],
            |mut acc, mut snp_chunk| {
                normalize_matrix_columns_inplace(&mut snp_chunk, 0);
                snp_chunk
                    .axis_iter(Axis(0))
                    .enumerate()
                    .for_each(|(i, row)| acc[i] += op(row.iter()));
                acc
            },
        )
        .reduce(
            || vec![0f32; num_people],
            |mut acc, x| {
                acc.iter_mut().enumerate().for_each(|(i, a)| *a += x[i]);
                acc
            },
        );
    Array::from_shape_vec(num_people, sigma_vec).unwrap()
}

// TODO: unit test
pub fn get_column_mean_and_std(
    geno_bed: &PlinkBed,
    snp_range: &OrderedIntegerSet<usize>,
    snp_chunk_size: usize,
) -> (Array<f32, Ix1>, Array<f32, Ix1>) {
    let mut snp_means = Vec::new();
    let mut snp_stds = Vec::new();
    geno_bed
        .col_chunk_iter(snp_chunk_size, Some(snp_range.clone()))
        .into_par_iter()
        .flat_map(|snp_chunk| {
            let mut m_and_s = Vec::new();
            for col in snp_chunk.gencolumns() {
                m_and_s.push((
                    mean(col.iter()) as f32,
                    standard_deviation(col.iter(), 0) as f32,
                ));
            }
            m_and_s
        })
        .collect::<Vec<(f32, f32)>>()
        .into_iter()
        .for_each(|(m, s)| {
            snp_means.push(m);
            snp_stds.push(s);
        });
    (
        Array::from_shape_vec(snp_means.len(), snp_means).unwrap(),
        Array::from_shape_vec(snp_stds.len(), snp_stds).unwrap(),
    )
}

pub fn get_gxg_dot_semi_kronecker_z_from_gz_and_ssq(
    mut gz: Array<f32, Ix2>,
    ssq: &Array<f32, Ix1>,
) -> Array<f32, Ix2> {
    let (num_people, num_cols) = gz.dim();
    for i in 0..num_people {
        let s = ssq[i];
        for b in 0..num_cols {
            let val1 = gz[[i, b]];
            gz[[i, b]] = (val1 * val1 - s) / 2.;
        }
    }
    gz
}

#[inline]
pub fn normalized_g_dot_rand(
    geno_bed: &mut PlinkBed,
    snp_range: Option<OrderedIntegerSet<usize>>,
    snp_mean: &Array<f32, Ix1>,
    snp_std: &Array<f32, Ix1>,
    num_random_vecs: usize,
    num_snps_per_chunk: Option<usize>,
) -> Array<f32, Ix2> {
    let num_snps = match &snp_range {
        Some(range) => range.size(),
        None => geno_bed.total_num_snps(),
    };
    let rand_mat =
        generate_plus_minus_one_bernoulli_matrix(num_snps, num_random_vecs);
    normalized_g_dot_matrix(
        geno_bed,
        snp_range,
        snp_mean,
        snp_std,
        &rand_mat,
        None,
        num_snps_per_chunk,
    )
}

pub fn normalized_g_transpose_dot_matrix(
    geno_bed: &PlinkBed,
    snp_range: Option<OrderedIntegerSet<usize>>,
    snp_mean: &Array<f32, Ix1>,
    snp_std: &Array<f32, Ix1>,
    rhs_matrix: &Array<f32, Ix2>,
    rhs_matrix_row_scaling: Option<&Array<f32, Ix1>>,
    num_snps_per_chunk: Option<usize>,
) -> Array<f32, Ix2> {
    let chunk_size = num_snps_per_chunk.unwrap_or(DEFAULT_NUM_SNPS_PER_CHUNK);

    let num_snps = match &snp_range {
        Some(range) => range.size(),
        None => geno_bed.total_num_snps(),
    };
    let num_random_vecs = rhs_matrix.dim().1;

    let scaled_rhs_matrix = match rhs_matrix_row_scaling {
        Some(scales) => {
            assert_eq!(rhs_matrix.dim().0, scales.dim());
            rhs_matrix
                * &scales
                    .to_owned()
                    .into_shape((rhs_matrix.dim().0, 1))
                    .unwrap()
        }
        // never used
        None => Array::zeros((1, 1)),
    };
    let rhs_matrix = match rhs_matrix_row_scaling {
        Some(_) => &scaled_rhs_matrix,
        None => rhs_matrix,
    };

    let mut z_col_sum = Vec::<f32>::new();
    rhs_matrix
        .axis_iter(Axis(1))
        .into_par_iter()
        .map(|col| sum_f32(col.iter()))
        .collect_into_vec(&mut z_col_sum);

    let product_vec = geno_bed
        .col_chunk_iter(chunk_size, snp_range)
        .into_par_iter()
        .enumerate()
        .fold(
            || vec![0f32; num_snps * num_random_vecs],
            |mut acc, (chunk_index, snp_chunk)| {
                let chunk_product = snp_chunk
                    .t()
                    .dot(rhs_matrix)
                    .as_slice()
                    .unwrap()
                    .to_owned();
                for local_snp_index in 0..snp_chunk.dim().1 {
                    let global_snp_index =
                        chunk_index * chunk_size + local_snp_index;
                    let m = snp_mean[global_snp_index];
                    let s = snp_std[global_snp_index];
                    let offset = local_snp_index * num_random_vecs;
                    let global_offset = global_snp_index * num_random_vecs;
                    for j in 0..num_random_vecs {
                        acc[global_offset + j] =
                            (chunk_product[offset + j] - m * z_col_sum[j]) / s;
                    }
                }
                acc
            },
        )
        .reduce(
            || vec![0f32; num_snps * num_random_vecs],
            |mut acc, x| {
                acc.iter_mut().enumerate().for_each(|(i, a)| *a += x[i]);
                acc
            },
        );
    Array::from_shape_vec((num_snps, num_random_vecs), product_vec).unwrap()
}

pub fn normalized_g_dot_matrix(
    geno_bed: &PlinkBed,
    snp_range: Option<OrderedIntegerSet<usize>>,
    snp_mean: &Array<f32, Ix1>,
    snp_std: &Array<f32, Ix1>,
    rhs_matrix: &Array<f32, Ix2>,
    row_scaling: Option<&Array<f32, Ix1>>,
    num_snps_per_chunk: Option<usize>,
) -> Array<f32, Ix2> {
    let chunk_size = num_snps_per_chunk.unwrap_or(DEFAULT_NUM_SNPS_PER_CHUNK);

    let num_people = geno_bed.num_people;
    let num_cols = rhs_matrix.dim().1;
    let rhs_matrix = rhs_matrix
        / &snp_std.to_owned().into_shape((snp_std.dim(), 1)).unwrap();

    let mut product_vec = geno_bed
        .col_chunk_iter(chunk_size, snp_range)
        .into_par_iter()
        .enumerate()
        .fold(
            || vec![0f32; num_people * num_cols],
            |mut acc, (chunk_index, snp_chunk)| {
                let start = chunk_index * chunk_size;
                let chunk_product = snp_chunk
                    .dot(
                        &rhs_matrix
                            .slice(s![start..start + snp_chunk.dim().1, ..]),
                    )
                    .as_slice()
                    .unwrap()
                    .to_owned();
                acc.par_iter_mut()
                    .enumerate()
                    .for_each(|(i, a)| *a += chunk_product[i]);
                acc
            },
        )
        .reduce(
            || vec![0f32; num_people * num_cols],
            |mut acc, x| {
                acc.par_iter_mut().enumerate().for_each(|(i, a)| *a += x[i]);
                acc
            },
        );
    let mean_dot_rhs_matrix = snp_mean.t().dot(&rhs_matrix);
    product_vec
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, x)| *x -= mean_dot_rhs_matrix[i % num_cols]);
    if let Some(scales) = row_scaling {
        for (i, s) in scales.iter().enumerate() {
            for j in 0..num_cols {
                product_vec[i * num_cols + j] *= s;
            }
        }
    }
    Array::from_shape_vec((num_people, num_cols), product_vec).unwrap()
}

pub fn pheno_dot_geno(
    pheno_arr: &Array<f32, Ix1>,
    geno_bed: &PlinkBed,
    snp_range: &OrderedIntegerSet<usize>,
    chunk_size: usize,
) -> Vec<f32> {
    geno_bed
        .col_chunk_iter(chunk_size, Some(snp_range.clone()))
        .into_par_iter()
        .flat_map(|mut snp_chunk| {
            normalize_matrix_columns_inplace(&mut snp_chunk, 0);
            pheno_arr.dot(&snp_chunk).as_slice().unwrap().to_owned()
        })
        .collect()
}

pub fn pheno_k_pheno(
    pheno_arr: &Array<f32, Ix1>,
    snp_range: &OrderedIntegerSet<usize>,
    geno_bed: &PlinkBed,
    snp_means: &Array<f32, Ix1>,
    snp_stds: &Array<f32, Ix1>,
    chunk_size: usize,
) -> f64 {
    let pheno_sum = sum_f32(pheno_arr.iter());
    let yggy = geno_bed
        .col_chunk_iter(chunk_size, Some(snp_range.clone()))
        .into_par_iter()
        .enumerate()
        .fold(
            || 0f32,
            |acc, (chunk_index, snp_chunk)| {
                let mut arr =
                    pheno_arr.dot(&snp_chunk).as_slice().unwrap().to_owned();
                let offset = chunk_index * chunk_size;
                for (i, x) in arr.iter_mut().enumerate() {
                    *x = (*x - pheno_sum * snp_means[offset + i])
                        / snp_stds[offset + i];
                }
                acc + sum_of_squares_f32(arr.iter())
            },
        )
        .sum::<f32>();
    yggy as f64 / snp_range.size() as f64
}

pub fn pheno_g_pheno_from_pheno_matrix(
    pheno_matrix: &Array<f32, Ix2>,
    snp_range: &OrderedIntegerSet<usize>,
    geno_bed: &PlinkBed,
    snp_means: &Array<f32, Ix1>,
    snp_stds: &Array<f32, Ix1>,
    chunk_size: Option<usize>,
) -> Vec<f64> {
    let gy = normalized_g_transpose_dot_matrix(
        geno_bed,
        Some(snp_range.clone()),
        snp_means,
        snp_stds,
        pheno_matrix,
        None,
        chunk_size,
    );
    gy.gencolumns()
        .into_iter()
        .map(|col| sum_of_squares(col.iter()))
        .collect()
}

pub fn sum_of_column_wise_inner_product(
    arr1: &Array<f32, Ix2>,
    arr2: &Array<f32, Ix2>,
) -> f32 {
    arr1.axis_iter(Axis(1))
        .into_par_iter()
        .enumerate()
        .map(|(b, col)| col.t().dot(&arr2.slice(s![.., b])))
        .sum::<f32>()
}
