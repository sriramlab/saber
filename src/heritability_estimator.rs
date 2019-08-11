use colored::Colorize;
use ndarray::{Array, array, Axis, Ix1, Ix2, s};
use ndarray_linalg::Solve;
use rayon::prelude::*;

use bio_file_reader::error::Error as PlinkBedError;
use bio_file_reader::plink_bed::PlinkBed;
use bio_file_reader::plink_bim::{PartitionKeyType, PlinkBim};
use math::set::ordered_integer_set::OrderedIntegerSet;
use math::set::traits::{Finite, Set};
use std::{fmt, io};
use std::collections::{HashMap, HashSet};
use std::iter::FromIterator;

use crate::jackknife::{AdditiveJackknife, Jackknife, JackknifePartitions};
use crate::trace_estimator::{DEFAULT_NUM_SNPS_PER_CHUNK, estimate_gxg_dot_y_norm_sq, estimate_gxg_gram_trace,
                             estimate_gxg_kk_trace, estimate_tr_gxg_ki_gxg_kj, estimate_tr_k_gxg_k,
                             estimate_tr_kk, normalized_g_g_transpose_dot_matrix};
use crate::util::matrix_util::{generate_plus_minus_one_bernoulli_matrix, normalize_matrix_columns_inplace,
                               normalize_vector_inplace};
use crate::util::stats_util::{mean, n_choose_2, std, sum_f32, sum_of_squares, sum_of_squares_f32};

const DEFAULT_PARTITION_NAME: &str = "default_partition";

#[inline]
fn bold_print(msg: &String) {
    println!("{}", msg.bold());
}

pub enum Error {
    IO { why: String, io_error: io::Error },
    Generic(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::IO { why, .. } => write!(f, "IO error: {}", why),
            Error::Generic(why) => write!(f, "Generic Error: {}", why)
        }
    }
}

impl From<PlinkBedError> for Error {
    fn from(err: PlinkBedError) -> Error {
        match err {
            PlinkBedError::BadFormat(why) => Error::Generic(why),
            PlinkBedError::Generic(why) => Error::Generic(why),
            PlinkBedError::IO { why, io_error } => Error::IO { why, io_error },
        }
    }
}

impl From<io::Error> for Error {
    fn from(err: io::Error) -> Error {
        Error::IO { why: "IO Error: ".to_string(), io_error: err }
    }
}

impl From<String> for Error {
    fn from(err: String) -> Error {
        Error::Generic(err)
    }
}

pub fn pheno_dot_geno(pheno_arr: &Array<f32, Ix1>,
                      geno_bed: &PlinkBed, snp_range: &OrderedIntegerSet<usize>,
                      chunk_size: usize) -> Vec<f32> {
    geno_bed.col_chunk_iter(chunk_size, Some(snp_range.clone()))
            .into_par_iter()
            .flat_map(|mut snp_chunk| {
                normalize_matrix_columns_inplace(&mut snp_chunk, 0);
                pheno_arr.dot(&snp_chunk).as_slice().unwrap().to_owned()
            })
            .collect()
}

pub fn pheno_k_pheno(pheno_arr: &Array<f32, Ix1>, snp_range: &OrderedIntegerSet<usize>, geno_bed: &PlinkBed,
                     snp_means: &Array<f32, Ix1>, snp_stds: &Array<f32, Ix1>,
                     chunk_size: usize) -> f64 {
    let pheno_sum = sum_f32(pheno_arr.iter());
    let yggy = geno_bed.col_chunk_iter(chunk_size, Some(snp_range.clone()))
                       .into_par_iter()
                       .enumerate()
                       .fold(|| 0f32, |acc, (chunk_index, snp_chunk)| {
                           let mut arr = pheno_arr.dot(&snp_chunk).as_slice().unwrap().to_owned();
                           let offset = chunk_index * chunk_size;
                           for (i, x) in arr.iter_mut().enumerate() {
                               *x = (*x - pheno_sum * snp_means[offset + i]) / snp_stds[offset + i];
                           }
                           acc + sum_of_squares_f32(arr.iter())
                       })
                       .sum::<f32>();
    yggy as f64 / snp_range.size() as f64
}

#[derive(Clone, PartialEq, Debug)]
pub struct PartitionedJackknifeEstimates {
    partition_names: Option<Vec<String>>,
    pub partition_means_and_stds: Vec<(f64, f64)>,
    pub sum_estimates: Option<(f64, f64)>,
}

fn get_jackknife_mean_and_std(estimates: &Vec<f64>) -> (f64, f64) {
    let num_knives = estimates.len();
    let s: f64 = estimates.iter().sum();
    let num_knives_minus_one_f64 = (num_knives - 1) as f64;
    let mut x_i = Vec::new();
    for i in 0..num_knives {
        x_i.push((s - estimates[i]) / num_knives_minus_one_f64);
    }
    let x_avg = s / num_knives as f64;
    let standard_error = (x_i.into_iter().map(|x| {
        let delta = x - x_avg;
        delta * delta
    }).sum::<f64>() * num_knives_minus_one_f64 / num_knives as f64).sqrt();
    (x_avg, standard_error)
}

impl PartitionedJackknifeEstimates {
    pub fn from_jackknife_estimates(jackknife_iteration_estimates: &Vec<Vec<f64>>, partition_names: Option<Vec<String>>) -> Result<PartitionedJackknifeEstimates, String> {
        if jackknife_iteration_estimates.iter().map(|estimates| estimates.len()).collect::<HashSet<usize>>().len() > 1 {
            return Err(format!("inconsistent number of partitioned estimates across Jackknife iterations"));
        }
        if jackknife_iteration_estimates.len() == 0 {
            return Ok(PartitionedJackknifeEstimates {
                partition_names: None,
                partition_means_and_stds: Vec::new(),
                sum_estimates: None,
            });
        }
        let num_partitions = jackknife_iteration_estimates[0].len();
        if let Some(names) = &partition_names {
            if names.len() != num_partitions {
                return Err(format!("partition_names.len() {} != the number of partitions in the jackknife estimates {}",
                                   names.len(), num_partitions));
            }
        }
        let mut partition_estimates = vec![vec![0f64; jackknife_iteration_estimates.len()]; num_partitions];
        for (i, estimates) in jackknife_iteration_estimates.iter().enumerate() {
            for p in 0..num_partitions {
                partition_estimates[p][i] = estimates[p];
            }
        }
        let total_variance_estimates: Vec<f64> = jackknife_iteration_estimates.iter()
                                                                              .map(|v| v.iter()
                                                                                        .map(|&x| x as f64)
                                                                                        .sum())
                                                                              .collect();
        let sum_estimates = {
            if total_variance_estimates.len() > 1 {
                Some(get_jackknife_mean_and_std(&total_variance_estimates))
            } else {
                None
            }
        };
        Ok(PartitionedJackknifeEstimates {
            partition_names,
            partition_means_and_stds: partition_estimates.iter()
                                                         .map(|estimates| get_jackknife_mean_and_std(estimates))
                                                         .collect(),
            sum_estimates,
        })
    }
}

impl std::fmt::Display for PartitionedJackknifeEstimates {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let num_decimals = 7;
        if let Some(partition_names) = &self.partition_names {
            for (name, (m, s)) in partition_names.iter().zip(self.partition_means_and_stds.iter()) {
                writeln!(f, "estimate for partition named {}: {:.*} standard error: {:.*}",
                         name, num_decimals, m, num_decimals, s)?;
            }
        } else {
            for (i, (m, s)) in self.partition_means_and_stds.iter().enumerate() {
                writeln!(f, "estimate for partition {}: {:.*} standard error: {:.*}",
                         i, num_decimals, m, num_decimals, s)?;
            }
        }
        if let Some(sum_estimates) = self.sum_estimates {
            writeln!(f, "total estimate: {:.*} standard error: {:.*}",
                     num_decimals, sum_estimates.0, num_decimals, sum_estimates.1)?;
        }
        Ok(())
    }
}

// TODO: test
fn get_column_mean_and_std(geno_bed: &PlinkBed, snp_range: &OrderedIntegerSet<usize>) -> (Array<f32, Ix1>, Array<f32, Ix1>) {
    let chunk_size = DEFAULT_NUM_SNPS_PER_CHUNK;
    let mut snp_means = Vec::new();
    let mut snp_stds = Vec::new();
    geno_bed
        .col_chunk_iter(chunk_size, Some(snp_range.clone()))
        .into_par_iter()
        .flat_map(|snp_chunk| {
            let mut m_and_s = Vec::new();
            for col in snp_chunk.gencolumns() {
                m_and_s.push((mean(col.iter()) as f32, std(col.iter(), 0) as f32));
            }
            m_and_s
        })
        .collect::<Vec<(f32, f32)>>()
        .into_iter()
        .for_each(|(m, s)| {
            snp_means.push(m);
            snp_stds.push(s);
        });
    (Array::from_shape_vec(snp_means.len(), snp_means).unwrap(), Array::from_shape_vec(snp_stds.len(), snp_stds).unwrap())
}

pub fn estimate_heritability(mut geno_arr_bed: PlinkBed, plink_bim: PlinkBim,
                             mut pheno_arr: Array<f32, Ix1>, num_random_vecs: usize,
                             num_jackknife_partitions: usize,
) -> Result<PartitionedJackknifeEstimates, String> {
    fn get_normal_eqn_matrices(num_partitions: usize, num_people: usize, yy: f64) -> (Array<f64, Ix2>, Array<f64, Ix1>) {
        let num_people = num_people as f64;
        let mut a = Array::zeros((num_partitions + 1, num_partitions + 1));
        let mut b = Array::zeros(num_partitions + 1);
        a[[num_partitions, num_partitions]] = num_people;
        for i in 0..num_partitions {
            a[[i, num_partitions]] = num_people as f64;
            a[[num_partitions, i]] = num_people as f64;
        }
        b[num_partitions] = yy;
        (a, b)
    }

    let key_to_partition = plink_bim.get_fileline_partitions().unwrap_or(
        HashMap::from_iter(vec![
            (DEFAULT_PARTITION_NAME.to_string(), OrderedIntegerSet::from_slice(&[[0, geno_arr_bed.num_snps - 1]]))
        ].into_iter())
    );
    let partition_keys = {
        let mut keys: Vec<PartitionKeyType> = key_to_partition.keys().map(|s| s.to_string()).collect::<Vec<PartitionKeyType>>();
        if keys.iter().filter(|&k| k.parse::<i32>().is_err()).count() > 0 {
            keys.sort();
        } else {
            keys.sort_by_key(|k| k.parse::<i32>().unwrap());
        }
        keys
    };

    let jackknife_partitions = JackknifePartitions::from_integer_set(
        partition_keys.iter().map(|k| key_to_partition[k].clone()).collect(),
        num_jackknife_partitions,
        false);

    let num_partitions = partition_keys.len();
    let total_num_snps = key_to_partition.values().fold(0, |acc, partition| acc + partition.size());
    let num_people = geno_arr_bed.num_people;
    println!("num_people: {}\ntotal_num_snps: {}\n", num_people, total_num_snps);
    for key in partition_keys.iter() {
        println!("partition named {} has {} SNPs", key, key_to_partition[key].size());
    }

    println!("\n=> normalizing the phenotype vector");
    normalize_vector_inplace(&mut pheno_arr, 0);

    println!("\n=> computing yy");
    let yy = sum_of_squares(pheno_arr.iter());

    let mut heritability_estimates = Vec::new();

    let mut snp_partitions = Vec::new();
    let mut num_snps = Vec::new();
    let mut ggz_jackknife = Vec::new();
    let mut yky_jackknives = Vec::new();
    let random_vecs = generate_plus_minus_one_bernoulli_matrix(num_people, num_random_vecs);
    for key in partition_keys.iter() {
        println!("=> processing partition named {}", key);
        let partition = key_to_partition[key].clone();
        ggz_jackknife.push(
            AdditiveJackknife::from_op_over_jackknife_partitions(&jackknife_partitions, |_, knife| {
                let range_intersect = knife.intersect(&partition);
                let (snp_mean_i, snp_std_i) = get_column_mean_and_std(&geno_arr_bed, &range_intersect);
                normalized_g_g_transpose_dot_matrix(&mut geno_arr_bed, Some(range_intersect), &snp_mean_i, &snp_std_i, &random_vecs, None)
            })
        );
        let means_and_std_jackknife = Jackknife::from_op_over_jackknife_partitions(&jackknife_partitions, |jackknife_p|
            get_column_mean_and_std(&geno_arr_bed, &jackknife_p.intersect(&partition)),
        );
        let yky_jackknife = AdditiveJackknife::from_op_over_jackknife_partitions(&jackknife_partitions, |k, knife| {
            let sub_range = knife.intersect(&partition);
            let num_snps_in_sub_range = sub_range.size() as f64;
            pheno_k_pheno(&pheno_arr,
                          &sub_range,
                          &geno_arr_bed,
                          &means_and_std_jackknife.components[k].0,
                          &means_and_std_jackknife.components[k].1,
                          DEFAULT_NUM_SNPS_PER_CHUNK) * num_snps_in_sub_range
        });
        yky_jackknives.push(yky_jackknife);
        num_snps.push(partition.size());
        snp_partitions.push(partition);
    }

    use ndarray_parallel::prelude::*;
    for (k, jackknife_partition) in jackknife_partitions.iter().enumerate() {
        println!("\n=> leaving out jackknife partition with index {}", k);
        let (mut a, mut b) = get_normal_eqn_matrices(num_partitions, num_people, yy);
        for i in 0..num_partitions {
            let num_snps_i = (num_snps[i] - jackknife_partition.intersect(&snp_partitions[i]).size()) as f64;
            let ggz_i = ggz_jackknife[i].sum_minus_component(k);
            a[[i, i]] = sum_of_squares_f32(ggz_i.iter()) as f64 / num_snps_i / num_snps_i / num_random_vecs as f64;
//            println!("tr(k_{}_k_{})_est: {} num_snps_i: {}", i, i, a[[i, i]], num_snps_i);
            b[i] = yky_jackknives[i].sum_minus_component(k) / num_snps_i;
            for j in i + 1..num_partitions {
                let num_snps_j = (num_snps[j] - jackknife_partition.intersect(&snp_partitions[j]).size()) as f64;
                let ggz_j = ggz_jackknife[j].sum_minus_component(k);
                let ssq = ggz_j.axis_iter(Axis(1))
                               .into_par_iter()
                               .enumerate()
                               .map(|(b, col)| col.t().dot(&ggz_i.slice(s![.., b])))
                               .sum::<f32>();
                let tr_ki_kj_est = ssq as f64 / num_snps_i / num_snps_j / num_random_vecs as f64;
//                println!("tr(k_{}_k_{})_est: {}", i, j, tr_ki_kj_est);
                a[[i, j]] = tr_ki_kj_est;
                a[[j, i]] = tr_ki_kj_est;
            }
        }
//        println!("solving ax=b\na = {:?}\nb = {:?}", a, b);
        let mut sig_sq = a.solve_into(b).unwrap().as_slice().unwrap().to_owned();
//        for i in 0..num_partitions {
//            println!("variance estimate for partition named {}: {}", partition_keys[i], sig_sq[i] as f64);
//        }
//        println!("total var estimate: {}", sig_sq[..num_partitions].iter().map(|x| *x as f64).sum::<f64>());
//        println!("noise estimate: {}", sig_sq[num_partitions]);
        sig_sq.truncate(num_partitions);
        heritability_estimates.push(sig_sq.to_vec());
    }
    PartitionedJackknifeEstimates::from_jackknife_estimates(&heritability_estimates, Some(partition_keys))
}

/// `geno_arr` is the genotype matrix for the G component
/// Each array in `le_snps_arr` contains the gxg basis SNPs for the corresponding gxg component
/// Returns (a, b, var_estimates, normalized_geno_arr, normalized_le_snps_arr, normalized_pheno_arr),
/// where `a` and `b` are the matrix A and vector b in Ax = b that is solved for the heritability estimates.
/// `var_estimates` is a vector of the variance estimates due to G, the GxG components, and noise, in that order.
/// The phenotypes are normalized to have unit variance so the `var_estimates` are the fractions of the total
/// phenotypic variance due to the various components.
pub fn estimate_g_and_multi_gxg_heritability(geno_arr: &mut PlinkBed, mut le_snps_arr: Vec<Array<f32, Ix2>>,
                                             mut pheno_arr: Array<f32, Ix1>, num_random_vecs: usize,
) -> Result<(Array<f64, Ix2>, Array<f64, Ix1>, Vec<f64>, Vec<Array<f32, Ix2>>, Array<f32, Ix1>), Error> {
    let (num_people, num_snps) = (geno_arr.num_people, geno_arr.num_snps);
    let num_gxg_components = le_snps_arr.len();
    println!("\n=> estimating heritability due to G and GxG\nnum_people: {}\nnum_snps: {}\nnumber of GxG components: {}",
             num_people, num_snps, num_gxg_components);
    for (i, arr) in le_snps_arr.iter().enumerate() {
        println!("GxG component [{}/{}]: {} LE SNPs", i + 1, num_gxg_components, arr.dim().1);
    }

    for (i, arr) in le_snps_arr.iter_mut().enumerate() {
        println!("=> normalizing GxG component [{}/{}]", i + 1, num_gxg_components);
        normalize_matrix_columns_inplace(arr, 0);
    }

    println!("\n=> normalizing the phenotype vector");
    normalize_vector_inplace(&mut pheno_arr, 0);

    let mut a = Array::<f64, Ix2>::zeros((num_gxg_components + 2, num_gxg_components + 2));

    println!("\n=> estimating traces related to the G matrix");
    let num_rand_z = 100usize;
    let tr_kk_est = estimate_tr_kk(geno_arr, None, num_rand_z, None);
    a[[0, 0]] = tr_kk_est;
    println!("tr_kk_est: {}", tr_kk_est);

    println!("\n=> estimating traces related to the GxG component pairs");
    for i in 0..num_gxg_components {
        for j in i + 1..num_gxg_components {
            a[[1 + i, 1 + j]] = estimate_tr_gxg_ki_gxg_kj(&le_snps_arr[i], &le_snps_arr[j], num_random_vecs);
            a[[1 + j, 1 + i]] = a[[1 + i, 1 + j]];
            println!("tr(gxg_k{} gxg_k{}) est: {}", i + 1, j + 1, a[[1 + i, 1 + j]]);
        }
    }

    println!("\n=> estimating traces related to the GxG components");
    for i in 0..num_gxg_components {
        println!("\nGXG component {}", i + 1);
        let mm = n_choose_2(le_snps_arr[i].dim().1) as f64;

        let gxg_tr_kk_est = estimate_gxg_kk_trace(&le_snps_arr[i], num_random_vecs)?;
        a[[1 + i, 1 + i]] = gxg_tr_kk_est;
        println!("gxg_tr_kk{}_est: {}", i + 1, gxg_tr_kk_est);

        let gxg_tr_k_est = estimate_gxg_gram_trace(&le_snps_arr[i], num_random_vecs)? / mm;
        a[[num_gxg_components + 1, 1 + i]] = gxg_tr_k_est;
        a[[1 + i, num_gxg_components + 1]] = gxg_tr_k_est;
        println!("gxg_tr_k{}_est: {}", i + 1, gxg_tr_k_est);

        let tr_gk_est = estimate_tr_k_gxg_k(geno_arr, &le_snps_arr[i], num_random_vecs, None);
        a[[0, 1 + i]] = tr_gk_est;
        a[[1 + i, 0]] = tr_gk_est;
        println!("tr_gk{}_est: {}", i + 1, tr_gk_est);
    }

    let n = num_people as f64;
    a[[num_gxg_components + 1, 0]] = n;
    a[[0, num_gxg_components + 1]] = n;
    a[[num_gxg_components + 1, num_gxg_components + 1]] = n;
    let b = get_yky_gxg_yky_and_yy(geno_arr,
                                   &pheno_arr,
                                   &le_snps_arr,
                                   num_random_vecs);
    println!("solving ax=b\na = {:?}\nb = {:?}", a, b);
    let sig_sq = a.solve_into(b.clone()).unwrap();

    println!("variance estimates: {:?}", sig_sq);
    let mut var_estimates = Vec::new();
    for i in 0..num_gxg_components + 2 {
        var_estimates.push(sig_sq[i]);
    }
    Ok((a, b, var_estimates, le_snps_arr, pheno_arr))
}

/// `saved_traces` is the matrix A in the normal equation Ax = y for heritability estimation
pub fn estimate_g_and_multi_gxg_heritability_from_saved_traces(geno_bed: &mut PlinkBed, mut le_snps_arr: Vec<Array<f32, Ix2>>,
                                                               mut pheno_arr: Array<f32, Ix1>, num_random_vecs: usize, saved_traces: Array<f64, Ix2>)
    -> Result<(Array<f64, Ix2>, Array<f64, Ix1>, Vec<f64>, Vec<Array<f32, Ix2>>, Array<f32, Ix1>), Error> {
    let (num_people, num_snps) = (geno_bed.num_people, geno_bed.num_snps);
    let num_gxg_components = le_snps_arr.len();
    println!("\n=> estimating heritability due to G and GxG\nnum_people: {}\nnum_snps: {}\nnumber of GxG components: {}",
             num_people, num_snps, num_gxg_components);
    for (i, arr) in le_snps_arr.iter().enumerate() {
        println!("GxG component [{}/{}]: {} LE SNPs", i + 1, num_gxg_components, arr.dim().1);
    }

    for (i, arr) in le_snps_arr.iter_mut().enumerate() {
        println!("=> normalizing GxG component [{}/{}]", i + 1, num_gxg_components);
        normalize_matrix_columns_inplace(arr, 0);
    }

    println!("\n=> normalizing the phenotype vector");
    normalize_vector_inplace(&mut pheno_arr, 0);

    println!("\n=> computing yy yky and estimating gxg_yky");
    let b = get_yky_gxg_yky_and_yy(geno_bed,
                                   &pheno_arr,
                                   &le_snps_arr,
                                   num_random_vecs);

    println!("solving ax=b\na = {:?}\nb = {:?}", saved_traces, b);
    let sig_sq = saved_traces.solve_into(b.clone()).unwrap();

    println!("variance estimates: {:?}", sig_sq);
    let mut var_estimates = Vec::new();
    for i in 0..num_gxg_components + 2 {
        var_estimates.push(sig_sq[i]);
    }
    Ok((saved_traces, b, var_estimates, le_snps_arr, pheno_arr))
}

fn get_yky_gxg_yky_and_yy(geno_arr: &mut PlinkBed, normalized_pheno_arr: &Array<f32, Ix1>,
                          normalized_le_snps_arr: &Vec<Array<f32, Ix2>>, num_random_vecs: usize)
    -> Array<f64, Ix1> {
    let num_snps = geno_arr.num_snps;
    let num_gxg_components = normalized_le_snps_arr.len();

    let mut b = Array::<f64, Ix1>::zeros(num_gxg_components + 2);

    let yky = geno_arr
        .col_chunk_iter(1000, None)
        .into_par_iter()
        .fold(|| 0f32, |mut acc, mut snp_chunk| {
            normalize_matrix_columns_inplace(&mut snp_chunk, 0);
            let arr = snp_chunk.t().dot(normalized_pheno_arr).as_slice().unwrap().to_owned();
            acc += sum_of_squares_f32(arr.iter());
            acc
        })
        .reduce(|| 0f32, |a, b| {
            a + b
        }) / num_snps as f32;
    let yy = sum_of_squares(normalized_pheno_arr.iter());
    b[0] = yky as f64;
    b[num_gxg_components + 1] = yy;
    println!("yky: {}\nyy: {}", yky, yy);

    println!("\n=> estimating traces related to y and the GxG components");
    for i in 0..num_gxg_components {
        println!("\nGXG component {}", i + 1);
        let mm = n_choose_2(normalized_le_snps_arr[i].dim().1) as f64;
        println!("estimate_gxg_dot_y_norm_sq using {} random vectors", num_random_vecs * 50);
        let gxg_yky = estimate_gxg_dot_y_norm_sq(&normalized_le_snps_arr[i], &normalized_pheno_arr, num_random_vecs * 50) / mm;
        b[1 + i] = gxg_yky;
        println!("gxg{}_yky_est: {}", i + 1, gxg_yky);
    }
    b
}

pub fn estimate_gxg_heritability(gxg_basis_arr: Array<f32, Ix2>, mut pheno_arr: Array<f32, Ix1>, num_random_vecs: usize) -> Result<f64, String> {
    println!("\n=> estimate_gxg_heritability");
    let (num_people, num_basis_snps) = gxg_basis_arr.dim();
    let mm = n_choose_2(num_basis_snps) as f64;
    println!("num_people: {}\nnum_basis_snps: {}\nnumber of equivalent GxG SNPs: {}",
             num_people, num_basis_snps, n_choose_2(num_basis_snps));

    println!("\n=> normalizing the phenotype vector");
    normalize_vector_inplace(&mut pheno_arr, 0);

    let gxg_kk_trace_est = estimate_gxg_kk_trace(&gxg_basis_arr, num_random_vecs)?;
    let gxg_k_trace_est = estimate_gxg_gram_trace(&gxg_basis_arr, num_random_vecs)? / mm;

    println!("gxg_k_trace_est: {}", gxg_k_trace_est);
    println!("gxg_kk_trace_est: {}", gxg_kk_trace_est);

    let yky = estimate_gxg_dot_y_norm_sq(&gxg_basis_arr, &pheno_arr, num_random_vecs) / mm;
    let yy = sum_of_squares(pheno_arr.iter());
    println!("yky: {}", yky);
    println!("yy: {}", yy);

    let a = array![[gxg_kk_trace_est, gxg_k_trace_est], [gxg_k_trace_est, num_people as f64]];
    let b = array![yky, yy];
    println!("solving ax=b\na = {:?}\nb = {:?}", a, b);
    let sig_sq = a.solve_into(b).unwrap();

    let sig_sq_g = sig_sq[0] as f64;
    let sig_sq_e = sig_sq[1] as f64;
    println!("\nsig_sq: {} {}", sig_sq_g, sig_sq_e);
    let heritability = sig_sq_g / (sig_sq_g + sig_sq_e);
    bold_print(&format!("heritability: {}", heritability));

    Ok(heritability)
}

/// `geno_arr` is the genotype matrix for the G component
/// `le_snps_arr` contains the gxg basis SNPs
#[deprecated(note = "use estimate_g_and_multi_gxg_heritability instead")]
pub fn estimate_g_and_single_gxg_heritability(geno_arr_bed: &mut PlinkBed, mut le_snps_arr: Array<f32, Ix2>,
                                              mut pheno_arr: Array<f32, Ix1>, num_random_vecs: usize,
) -> Result<(f64, f64, f64), Error> {
    let mut geno_arr: Array<f32, Ix2> = geno_arr_bed.get_genotype_matrix(None)?;
    let (num_people, num_snps) = geno_arr.dim();
    let num_independent_snps = le_snps_arr.dim().1;
    println!("\n=> estimating heritability due to G and GxG\nnum_people: {}\nnum_snps: {}\nnum_independent_snps: {}",
             num_people, num_snps, num_independent_snps);

    println!("\n=> normalizing the genotype matrices");
    normalize_matrix_columns_inplace(&mut geno_arr, 0);
    normalize_matrix_columns_inplace(&mut le_snps_arr, 0);

    println!("\n=> normalizing the phenotype vector");
    normalize_vector_inplace(&mut pheno_arr, 0);

    println!("\n=> estimating traces related to the G matrix");
    let num_rand_z = 100usize;
    let tr_kk_est = estimate_tr_kk(geno_arr_bed, None, num_rand_z, None);
    println!("tr_kk_est: {}", tr_kk_est);
    let xy = geno_arr.t().dot(&pheno_arr);
    let yky = sum_of_squares(xy.iter()) / num_snps as f64;
    let yy = sum_of_squares(pheno_arr.iter());

    println!("\n=> estimating traces related to the GxG matrix");
    let mm = n_choose_2(num_independent_snps) as f64;

    let gxg_tr_kk_est = estimate_gxg_kk_trace(&le_snps_arr, num_random_vecs)?;
    let gxg_tr_k_est = estimate_gxg_gram_trace(&le_snps_arr, num_random_vecs)? / mm;

    println!("gxg_tr_k_est: {}", gxg_tr_k_est);
    println!("gxg_tr_kk_est: {}", gxg_tr_kk_est);

    println!("estimate_gxg_dot_y_norm_sq using {} random vectors", num_random_vecs * 50);
    let gxg_yky = estimate_gxg_dot_y_norm_sq(&le_snps_arr, &pheno_arr, num_random_vecs * 50) / mm;
    println!("gxg_yky: {}", gxg_yky);

    let tr_gk_est = estimate_tr_k_gxg_k(geno_arr_bed, &le_snps_arr, num_random_vecs, None);
    println!("tr_gk_est: {}", tr_gk_est);

    let n = num_people as f64;
    let a = array![[tr_kk_est, tr_gk_est, n], [tr_gk_est, gxg_tr_kk_est, gxg_tr_k_est], [n, gxg_tr_k_est, n]];
    let b = array![yky, gxg_yky, yy];
    println!("solving ax=b\na = {:?}\nb = {:?}", a, b);
    let sig_sq = a.solve_into(b).unwrap();

    println!("variance estimates: {:?}", sig_sq);
    Ok((sig_sq[0], sig_sq[1], sig_sq[2]))
}

#[deprecated(note = "use estimate_heritability instead")]
pub fn estimate_heritability_directly(mut geno_arr: Array<f32, Ix2>, mut pheno_arr: Array<f32, Ix1>, num_random_vecs: usize)
    -> Result<f64, String> {
    let (num_people, num_snps) = geno_arr.dim();
    println!("num_people: {}\nnum_snps: {}", num_people, num_snps);

    println!("\n=> normalizing the genotype matrix column-wise");
    normalize_matrix_columns_inplace(&mut geno_arr, 0);

    println!("\n=> normalizing the phenotype vector");
    normalize_vector_inplace(&mut pheno_arr, 0);

    println!("\n=> generating random estimators");
    let rand_vecs = generate_plus_minus_one_bernoulli_matrix(num_people, num_random_vecs);

    println!("\n=> MatMul geno_arr{:?} with rand_mat{:?}", geno_arr.dim(), rand_vecs.dim());
    let xz_arr = geno_arr.t().dot(&rand_vecs);

    println!("\n=> MatMul geno_arr{:?}.T with xz_arr{:?}", geno_arr.dim(), xz_arr.dim());
    let xxz = geno_arr.dot(&xz_arr);

    println!("\n=> calculating trace estimate through L2 squared");
    let trace_kk_est = sum_of_squares(xxz.iter()) / (num_snps * num_snps * num_random_vecs) as f64;
    println!("trace_kk_est: {}", trace_kk_est);

    println!("\n=> calculating yKy and yy");
    let yky = sum_of_squares(pheno_arr.dot(&geno_arr).iter()) / num_snps as f64;
    let yy = sum_of_squares(pheno_arr.iter());

    let n = num_people as f64;
    let a = array![[trace_kk_est, n], [n, n]];
    let b = array![yky, yy];
    println!("solving ax=b\na = {:?}\nb = {:?}", a, b);
    let sig_sq = a.solve_into(b).unwrap();
    println!("sig_sq: {:?}", sig_sq);

    let g_var = sig_sq[0] as f64;
    let noise_var = sig_sq[1] as f64;
    let heritability = g_var / (g_var + noise_var);
    println!("heritability: {}", heritability);

    Ok(heritability)
}
