use std::{fmt, io};
use std::collections::HashSet;

use analytic::set::ordered_integer_set::OrderedIntegerSet;
use analytic::set::traits::{Finite, Set};
use analytic::traits::ToIterator;
use biofile::error::Error as PlinkBedError;
use biofile::plink_bed::PlinkBed;
use biofile::plink_bim::PlinkBim;
use colored::Colorize;
use ndarray::{Array, array, Axis, Ix1, Ix2, s};
use ndarray_linalg::Solve;
use rayon::prelude::*;

use crate::jackknife::{AdditiveJackknife, Jackknife, JackknifePartitions};
use crate::trace_estimator::{DEFAULT_NUM_SNPS_PER_CHUNK, estimate_gxg_dot_y_norm_sq, estimate_gxg_gram_trace,
                             estimate_gxg_kk_trace, estimate_tr_gxg_ki_gxg_kj, estimate_tr_k_gxg_k, estimate_tr_kk,
                             normalized_g_dot_matrix, normalized_gxg_ssq, normalized_g_transpose_dot_matrix};
use crate::util::matrix_util::{generate_plus_minus_one_bernoulli_matrix, normalize_matrix_columns_inplace,
                               normalize_vector_inplace};
use crate::util::stats_util::{mean, n_choose_2, standard_deviation, sum_f32, sum_of_squares, sum_of_squares_f32};

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
    pub subset_sum_estimates: Option<Vec<(String, (f64, f64))>>,
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
    pub fn from_jackknife_estimates(jackknife_iteration_estimates: &Vec<Vec<f64>>,
                                    partition_names: Option<Vec<String>>,
                                    subset_sum_indices: Option<Vec<(String, OrderedIntegerSet<usize>)>>)
        -> Result<PartitionedJackknifeEstimates, String> {
        if jackknife_iteration_estimates.iter().map(|estimates| estimates.len()).collect::<HashSet<usize>>().len() > 1 {
            return Err(format!("inconsistent number of partitioned estimates across Jackknife iterations"));
        }
        if jackknife_iteration_estimates.len() == 0 {
            return Ok(PartitionedJackknifeEstimates {
                partition_names: None,
                partition_means_and_stds: Vec::new(),
                sum_estimates: None,
                subset_sum_estimates: None,
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

        let subset_sum_estimates = match subset_sum_indices {
            None => None,
            Some(indices_list) => {
                Some(indices_list.iter().map(|(subset_key, subset_indices)| {
                    (subset_key.to_string(),
                     get_jackknife_mean_and_std(&jackknife_iteration_estimates
                         .iter()
                         .map(|single_iter_estimates| {
                             subset_indices.to_iter().fold(0f64, |acc, i| acc + single_iter_estimates[i])
                         })
                         .collect::<Vec<f64>>()))
                }).collect::<Vec<(String, (f64, f64))>>())
            }
        };
        Ok(PartitionedJackknifeEstimates {
            partition_names,
            partition_means_and_stds: partition_estimates.iter()
                                                         .map(|estimates| get_jackknife_mean_and_std(estimates))
                                                         .collect(),
            sum_estimates,
            subset_sum_estimates,
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

        if let Some(subset_sum_estimates) = &self.subset_sum_estimates {
            for (key, (m, s)) in subset_sum_estimates.iter() {
                writeln!(f, "estimate for subset {}: {:.*} standard error: {:.*}",
                         key, num_decimals, m, num_decimals, s)?;
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
//        .into_par_iter()
        .flat_map(|snp_chunk| {
            let mut m_and_s = Vec::new();
            for col in snp_chunk.gencolumns() {
                m_and_s.push((mean(col.iter()) as f32, standard_deviation(col.iter(), 0) as f32));
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

pub fn estimate_heritability(geno_arr_bed: PlinkBed, plink_bim: PlinkBim,
                             mut pheno_arr: Array<f32, Ix1>, num_random_vecs: usize,
                             num_jackknife_partitions: usize)
    -> Result<PartitionedJackknifeEstimates, String> {
    let partitions = plink_bim.get_fileline_partitions_or(
        DEFAULT_PARTITION_NAME, OrderedIntegerSet::from_slice(&[[0, geno_arr_bed.num_snps - 1]]));
    let partition_array: Vec<OrderedIntegerSet<usize>> = partitions.iter().map(|(_, p)| p.clone()).collect();
    let partition_sizes: Vec<usize> = partition_array.iter().map(|p| p.size()).collect();

    let jackknife_partitions = JackknifePartitions::from_integer_set(
        partition_array.clone(),
        num_jackknife_partitions,
        false);

    let num_partitions = partition_array.len();
    let num_people = geno_arr_bed.num_people;

    println!("num_people: {}\ntotal_num_snps: {}\n", num_people, partition_sizes.iter().fold(0, |acc, size| acc + *size));
    partitions.ordered_partition_keys().iter().enumerate().for_each(|(i, k)| {
        println!("partition named {} has {} SNPs", k, partition_sizes[i]);
    });

    println!("\n=> normalizing the phenotype vector");
    normalize_vector_inplace(&mut pheno_arr, 0);

    let yy = sum_of_squares(pheno_arr.iter());
    println!("\n=> yy: {}", yy);

    let mut ggz_jackknife = Vec::new();
    let mut yky_jackknives = Vec::new();
    let random_vecs = generate_plus_minus_one_bernoulli_matrix(num_people, num_random_vecs);
    for (key, partition) in partitions.iter() {
        println!("=> processing partition named {}", key);
        ggz_jackknife.push(
            AdditiveJackknife::from_op_over_jackknife_partitions(&jackknife_partitions, |_, knife| {
                let range_intersect = knife.intersect(partition);
                let (snp_mean_i, snp_std_i) = get_column_mean_and_std(&geno_arr_bed, &range_intersect);
                let gtz = normalized_g_transpose_dot_matrix(&geno_arr_bed,
                                                            Some(range_intersect.clone()),
                                                            &snp_mean_i,
                                                            &snp_std_i,
                                                            &random_vecs,
                                                            None);
                normalized_g_dot_matrix(&geno_arr_bed,
                                        Some(range_intersect),
                                        &snp_mean_i,
                                        &snp_std_i,
                                        &gtz,
                                        Some(2048))
            })
        );
        let means_and_std_jackknife = Jackknife::from_op_over_jackknife_partitions(&jackknife_partitions, |jackknife_p|
            get_column_mean_and_std(&geno_arr_bed, &jackknife_p.intersect(partition)),
        );
        yky_jackknives.push(
            AdditiveJackknife::from_op_over_jackknife_partitions(&jackknife_partitions, |k, knife| {
                let sub_range = knife.intersect(partition);
                let num_snps_in_sub_range = sub_range.size() as f64;
                pheno_k_pheno(&pheno_arr,
                              &sub_range,
                              &geno_arr_bed,
                              &means_and_std_jackknife.components[k].0,
                              &means_and_std_jackknife.components[k].1,
                              DEFAULT_NUM_SNPS_PER_CHUNK) * num_snps_in_sub_range
            })
        );
    }

    use ndarray_parallel::prelude::*;
    let mut heritability_estimates = Vec::new();
    for (k, jackknife_partition) in jackknife_partitions.iter().enumerate() {
        println!("\n=> leaving out jackknife partition with index {}", k);
        let (mut a, mut b) = get_normal_eqn_matrices(num_partitions, num_people, yy);
        for i in 0..num_partitions {
            let num_snps_i = (partition_sizes[i] - jackknife_partition.intersect(&partition_array[i]).size()) as f64;
            let ggz_i = ggz_jackknife[i].sum_minus_component(k);
            a[[i, i]] = sum_of_squares_f32(ggz_i.iter()) as f64 / num_snps_i / num_snps_i / num_random_vecs as f64;
//            println!("tr(k_{}_k_{})_est: {} num_snps_i: {}", i, i, a[[i, i]], num_snps_i);
            b[i] = yky_jackknives[i].sum_minus_component(k) / num_snps_i;
            for j in i + 1..num_partitions {
                let num_snps_j = (partition_sizes[j] - jackknife_partition.intersect(&partition_array[j]).size()) as f64;
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
    PartitionedJackknifeEstimates::from_jackknife_estimates(&heritability_estimates,
                                                            Some(partitions.ordered_partition_keys().clone()),
                                                            None)
}

fn get_gxg_dot_semi_kronecker_z_from_gz_and_ssq_jackknife(gz_jackknife: &AdditiveJackknife<Array<f32, Ix2>>,
                                                          g_ssq_jackknife: &AdditiveJackknife<Array<f32, Ix1>>,
                                                          jackknife_leave_out_index: usize) -> Array<f32, Ix2> {
    get_gxg_dot_semi_kronecker_z_from_gz_and_ssq(gz_jackknife.sum_minus_component(jackknife_leave_out_index),
                                                 &g_ssq_jackknife.sum_minus_component(jackknife_leave_out_index))
}

pub fn get_gxg_dot_semi_kronecker_z_from_gz_and_ssq(mut gz: Array<f32, Ix2>, ssq: &Array<f32, Ix1>) -> Array<f32, Ix2> {
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

pub fn sum_of_column_wise_dot_square(arr1: &Array<f32, Ix2>, arr2: &Array<f32, Ix2>) -> f32 {
    use ndarray_parallel::prelude::*;
    arr1.axis_iter(Axis(1)).into_par_iter().enumerate().map(|(b, col_1)| {
        let x = col_1.dot(&arr2.slice(s![.., b]));
        x * x
    }).sum::<f32>()
}

pub fn estimate_g_gxg_heritability(g_bed: PlinkBed, g_bim: PlinkBim,
                                   gxg_basis_bed: PlinkBed, gxg_basis_bim: PlinkBim,
                                   mut pheno_arr: Array<f32, Ix1>,
                                   num_rand_vecs_g: usize,
                                   num_rand_vecs_gxg: usize,
                                   num_jackknife_partitions: usize)
    -> Result<PartitionedJackknifeEstimates, String> {
    let g_partitions = g_bim.get_fileline_partitions_or(DEFAULT_PARTITION_NAME, OrderedIntegerSet::from_slice(&[[0, g_bed.num_snps - 1]]));
    let g_partition_array: Vec<OrderedIntegerSet<usize>> = g_partitions.iter().map(|(_, p)| p.clone()).collect();
    let g_partition_sizes: Vec<usize> = g_partition_array.iter().map(|p| p.size()).collect();

    let gxg_partitions = gxg_basis_bim.get_fileline_partitions_or(DEFAULT_PARTITION_NAME, OrderedIntegerSet::from_slice(&[[0, gxg_basis_bed.num_snps - 1]]));
    let gxg_partition_array: Vec<OrderedIntegerSet<usize>> = gxg_partitions.iter().map(|(_, p)| p.clone()).collect();
    let gxg_partition_sizes: Vec<usize> = gxg_partition_array.iter().map(|p| p.size()).collect();

    let g_jackknife_partitions = JackknifePartitions::from_integer_set(
        g_partition_array.clone(), num_jackknife_partitions, false);

    let gxg_basis_jackknife_partitions = JackknifePartitions::from_integer_set(
        gxg_partition_array.clone(), num_jackknife_partitions, false);

    let num_g_partitions = g_partition_array.len();
    let num_gxg_partitions = gxg_partition_array.len();
    let total_num_partitions = num_g_partitions + num_gxg_partitions;
    let num_people = g_bed.num_people;

    assert_eq!(num_people, gxg_basis_bed.num_people,
               "g_bed has {} people but gxg_basis_bed has {} people", num_people, gxg_basis_bed.num_people);
    println!("num_people: {}\ntotal_num_g_snps: {}\ntotal_num_gxg_basis_snps: {}",
             num_people,
             g_partition_sizes.iter().fold(0, |acc, size| acc + *size),
             gxg_partition_sizes.iter().fold(0, |acc, size| acc + *size));
    g_partitions.ordered_partition_keys().iter().enumerate().for_each(|(i, k)| {
        println!("G partition named {} has {} SNPs", k, g_partition_sizes[i]);
    });
    gxg_partitions.ordered_partition_keys().iter().enumerate().for_each(|(i, k)| {
        println!("GxG partition named {} has {} SNPs", k, gxg_partition_sizes[i]);
    });

    normalize_vector_inplace(&mut pheno_arr, 0);
    println!("\n=> normalized the phenotype vector");

    let yy = sum_of_squares(pheno_arr.iter());
    println!("\n=> yy: {}", yy);

    let g_random_vecs = generate_plus_minus_one_bernoulli_matrix(num_people, num_rand_vecs_g);
    println!("=> generating ggz_jackknife");
    let ggz_jackknife: Vec<AdditiveJackknife<Array<f32, Ix2>>> = g_partition_array.par_iter().map(|partition| {
        AdditiveJackknife::from_op_over_jackknife_partitions(&g_jackknife_partitions, |_, knife| {
            let range_intersect = knife.intersect(partition);
            let (snp_mean_i, snp_std_i) = get_column_mean_and_std(&g_bed, &range_intersect);
            let gtz = normalized_g_transpose_dot_matrix(&g_bed,
                                                        Some(range_intersect.clone()),
                                                        &snp_mean_i,
                                                        &snp_std_i,
                                                        &g_random_vecs,
                                                        None);
            normalized_g_dot_matrix(&g_bed,
                                    Some(range_intersect),
                                    &snp_mean_i,
                                    &snp_std_i,
                                    &gtz,
                                    Some(2048))
        })
    }).collect();

    println!("=> generating gz_jackknife");
    let gz_jackknife: Vec<AdditiveJackknife<Array<f32, Ix2>>> = g_partition_array.par_iter().map(|partition| {
        AdditiveJackknife::from_op_over_jackknife_partitions(&g_jackknife_partitions, |_, knife| {
            let range_intersect = knife.intersect(partition);
            let (snp_mean_i, snp_std_i) = get_column_mean_and_std(&g_bed, &range_intersect);
            let rand_vecs = generate_plus_minus_one_bernoulli_matrix(range_intersect.size(), num_rand_vecs_g);
            normalized_g_dot_matrix(&g_bed, Some(range_intersect), &snp_mean_i, &snp_std_i, &rand_vecs, Some(2048))
        })
    }).collect();

    println!("=> generating ygy_jackknives");
    let ygy_jackknives: Vec<AdditiveJackknife<f64>> = g_partition_array.par_iter().map(|partition| {
        let means_and_std_jackknife = Jackknife::from_op_over_jackknife_partitions(&g_jackknife_partitions, |jackknife_p|
            get_column_mean_and_std(&g_bed, &jackknife_p.intersect(partition)),
        );
        AdditiveJackknife::from_op_over_jackknife_partitions(&g_jackknife_partitions, |k, knife| {
            let sub_range = knife.intersect(partition);
            let num_snps_in_sub_range = sub_range.size() as f64;
            pheno_k_pheno(&pheno_arr,
                          &sub_range,
                          &g_bed,
                          &means_and_std_jackknife.components[k].0,
                          &means_and_std_jackknife.components[k].1,
                          DEFAULT_NUM_SNPS_PER_CHUNK) * num_snps_in_sub_range
        })
    }).collect();

    println!("=> generating gxg_gz_jackknife");
    let gxg_gz_jackknife: Vec<AdditiveJackknife<Array<f32, Ix2>>> = gxg_partition_array.par_iter().map(|partition| {
        AdditiveJackknife::from_op_over_jackknife_partitions(&gxg_basis_jackknife_partitions, |_, knife| {
            let range_intersect = knife.intersect(partition);
            let (snp_mean_i, snp_std_i) = get_column_mean_and_std(&gxg_basis_bed, &range_intersect);
            let gxg_random_vecs = generate_plus_minus_one_bernoulli_matrix(range_intersect.size(), num_rand_vecs_gxg);
            normalized_g_dot_matrix(&gxg_basis_bed, Some(range_intersect), &snp_mean_i, &snp_std_i, &gxg_random_vecs, Some(2048))
        })
    }).collect();

    println!("=> generating gxg_gu_jackknife");
    let gxg_gu_jackknife: Vec<AdditiveJackknife<Array<f32, Ix2>>> = gxg_partition_array.par_iter().map(|partition| {
        AdditiveJackknife::from_op_over_jackknife_partitions(&gxg_basis_jackknife_partitions, |_, knife| {
            let range_intersect = knife.intersect(partition);
            let (snp_mean_i, snp_std_i) = get_column_mean_and_std(&gxg_basis_bed, &range_intersect);
            let gxg_random_vecs = generate_plus_minus_one_bernoulli_matrix(range_intersect.size(), num_rand_vecs_gxg);
            normalized_g_dot_matrix(&gxg_basis_bed, Some(range_intersect), &snp_mean_i, &snp_std_i, &gxg_random_vecs, Some(2048))
        })
    }).collect();

    println!("=> generating gxg_ssq_jackknife");
    let gxg_ssq_jackknife: Vec<AdditiveJackknife<Array<f32, Ix1>>> = gxg_partition_array.par_iter().map(|partition| {
        AdditiveJackknife::from_op_over_jackknife_partitions(&gxg_basis_jackknife_partitions, |_, knife| {
            let range_intersect = knife.intersect(partition);
            normalized_gxg_ssq(&gxg_basis_bed, Some(range_intersect), None)
        })
    }).collect();

    use ndarray_parallel::prelude::*;
    let mut heritability_estimates = Vec::new();
    let nrv_g = num_rand_vecs_g as f64;
    let nrv_gxg = num_rand_vecs_gxg as f64;
    for (k, (g_jackknife_range, gxg_jackknife_range)) in g_jackknife_partitions.iter()
                                                                               .zip(gxg_basis_jackknife_partitions.iter())
                                                                               .enumerate() {
        println!("\n=> leaving out jackknife partition with index {}", k);
        let (mut a, mut b) = get_normal_eqn_matrices(total_num_partitions, num_people, yy);
        let g_pairwise_est: Vec<(f64, Vec<f64>, Vec<f64>, f64)> = (0..num_g_partitions).collect::<Vec<usize>>().par_iter().map(|&i| {
            let num_snps_i = (g_partition_sizes[i] - g_jackknife_range.intersect(&g_partition_array[i]).size()) as f64;
            let ggz_i = ggz_jackknife[i].sum_minus_component(k);

            let tr_gk_i_gk_j_est_list: Vec<f64> = (i + 1..num_g_partitions).collect::<Vec<usize>>().par_iter().map(|&j| {
                let num_snps_j = (g_partition_sizes[j] - g_jackknife_range.intersect(&g_partition_array[j]).size()) as f64;
                let ggz_j = ggz_jackknife[j].sum_minus_component(k);
                let ssq = ggz_j.axis_iter(Axis(1))
                               .into_par_iter()
                               .enumerate()
                               .map(|(b, col)| col.t().dot(&ggz_i.slice(s![.., b])))
                               .sum::<f32>();
                let tr_ki_kj_est = ssq as f64 / num_snps_i / num_snps_j / nrv_g;
                tr_ki_kj_est
            }).collect();

            // tr(g_k gxg_k)
            let gz = gz_jackknife[i].sum_minus_component(k);
            let tr_g_gxg_est_list: Vec<f64> = (0..num_gxg_partitions).collect::<Vec<usize>>().par_iter().map(|&gxg_i| {
                let num_gxg_snps_i = n_choose_2(gxg_partition_sizes[gxg_i] - gxg_jackknife_range.intersect(&gxg_partition_array[gxg_i]).size()) as f64;
                let gxg_i_dot_semi_kronecker_z = get_gxg_dot_semi_kronecker_z_from_gz_and_ssq_jackknife(&gxg_gz_jackknife[gxg_i], &gxg_ssq_jackknife[gxg_i], k);
                let tr_g_gxg_est = sum_of_squares_f32(gxg_i_dot_semi_kronecker_z.t().dot(&gz).iter()) as f64 / num_gxg_snps_i / num_snps_i / nrv_g / nrv_gxg;
                tr_g_gxg_est
            }).collect();

            (sum_of_squares_f32(ggz_i.iter()) as f64 / num_snps_i / num_snps_i / nrv_g,
             tr_gk_i_gk_j_est_list,
             tr_g_gxg_est_list,
             ygy_jackknives[i].sum_minus_component(k) / num_snps_i)
        }).collect();

        for (i, (tr_kk_est, tr_gk_i_gk_j_est_list, tr_g_gxg_est_list, yky_est)) in g_pairwise_est.into_iter().enumerate() {
            a[[i, i]] = tr_kk_est;
            b[i] = yky_est;
            for (j, tr_ki_kj_est) in tr_gk_i_gk_j_est_list.into_iter().enumerate() {
                a[[i, i + 1 + j]] = tr_ki_kj_est;
                a[[i + 1 + j, i]] = tr_ki_kj_est;
                println!("tr_gk{}_gk{}_est: {}", i, j, tr_ki_kj_est);
            }
            for (gxg_i, tr_g_gxg_est) in tr_g_gxg_est_list.into_iter().enumerate() {
                let global_gxg_i = num_g_partitions + gxg_i;
                a[[global_gxg_i, i]] = tr_g_gxg_est;
                a[[i, global_gxg_i]] = tr_g_gxg_est;
                println!("tr_g_k{}_gxg_k{}_est: {}", i, gxg_i, tr_g_gxg_est);
            }
        }

        let gxg_pairwise_est: Vec<(f64, f64, Vec<f64>, f64)> = (0..num_gxg_partitions).collect::<Vec<usize>>().par_iter().map(|&i| {
            let num_gxg_snps_i = n_choose_2(gxg_partition_sizes[i] - gxg_jackknife_range.intersect(&gxg_partition_array[i]).size()) as f64;
            let gxg_i_dot_semi_kronecker_z = get_gxg_dot_semi_kronecker_z_from_gz_and_ssq_jackknife(&gxg_gz_jackknife[i], &gxg_ssq_jackknife[i], k);
            let gxg_i_dot_semi_kronecker_u = get_gxg_dot_semi_kronecker_z_from_gz_and_ssq_jackknife(&gxg_gu_jackknife[i], &gxg_ssq_jackknife[i], k);
            let gxg_upper_triangular: Vec<f64> = (i + 1..num_gxg_partitions).collect::<Vec<usize>>().par_iter().map(|&j| {
                let num_gxg_snps_j = n_choose_2(gxg_partition_sizes[j] - gxg_jackknife_range.intersect(&gxg_partition_array[j]).size()) as f64;
                let gxg_j_dot_semi_kronecker_z = get_gxg_dot_semi_kronecker_z_from_gz_and_ssq_jackknife(&gxg_gz_jackknife[j], &gxg_ssq_jackknife[j], k);
                let tr_gxg_i_gxg_j_est = sum_of_squares_f32(
                    gxg_i_dot_semi_kronecker_z.t().dot(&gxg_j_dot_semi_kronecker_z).iter()
                ) as f64 / num_gxg_snps_i / num_gxg_snps_j / nrv_gxg / nrv_gxg;
                tr_gxg_i_gxg_j_est
            }).collect();
            (sum_of_squares_f32(gxg_i_dot_semi_kronecker_z.iter()) as f64 / num_gxg_snps_i / nrv_gxg,
             sum_of_squares_f32(gxg_i_dot_semi_kronecker_z.t().dot(&gxg_i_dot_semi_kronecker_u).iter()) as f64 / num_gxg_snps_i / num_gxg_snps_i / nrv_gxg / nrv_gxg,
             gxg_upper_triangular,
             sum_of_squares_f32(pheno_arr.t().dot(&gxg_i_dot_semi_kronecker_z).iter()) as f64 / num_gxg_snps_i / nrv_gxg)
        }).collect();

        for (i, (tr_gxg_ki_est, tr_gxg_kk_est, gxg_upper_triangular, y_gxg_k_y_est)) in gxg_pairwise_est.into_iter().enumerate() {
            let global_i = num_g_partitions + i;
            a[[global_i, total_num_partitions]] = tr_gxg_ki_est;
            a[[total_num_partitions, global_i]] = tr_gxg_ki_est;
            a[[global_i, global_i]] = tr_gxg_kk_est;
            b[global_i] = y_gxg_k_y_est;
            println!("tr_gxg_k{}_est: {}", i, tr_gxg_ki_est);
            println!("tr_gxg_kk{}_est: {}", i, tr_gxg_kk_est);
            println!("tr_y_gxg_k{}_y_est: {}", i, y_gxg_k_y_est);
            for (j, tr_gxg_i_gxg_j_est) in gxg_upper_triangular.into_iter().enumerate() {
                let global_j = num_g_partitions + i + 1 + j;
                a[[global_i, global_j]] = tr_gxg_i_gxg_j_est;
                a[[global_j, global_i]] = tr_gxg_i_gxg_j_est;
                println!("tr_gxg_k{}_gxg_k{}: {}", i, i + 1 + j, tr_gxg_i_gxg_j_est);
            }
        }
        println!("solving A={:?} b={:?}", a, b);
        let mut sig_sq = a.solve_into(b).unwrap().as_slice().unwrap().to_owned();
        sig_sq.truncate(total_num_partitions);
        println!("sig_sq: {:?}", sig_sq);
        heritability_estimates.push(sig_sq.to_vec());
    }
    let mut total_partition_keys: Vec<String> = g_partitions.ordered_partition_keys().iter().map(|k| {
        let mut key = "G ".to_string();
        key.push_str(k);
        key
    }).collect();
    gxg_partitions.ordered_partition_keys().iter().for_each(|k| {
        let mut key = "GxG ".to_string();
        key.push_str(k);
        total_partition_keys.push(key);
    });
    PartitionedJackknifeEstimates::from_jackknife_estimates(
        &heritability_estimates,
        Some(total_partition_keys),
        Some(vec![
            ("G".to_string(), OrderedIntegerSet::from_slice(&[[0, num_g_partitions - 1]])),
            ("GxG".to_string(), OrderedIntegerSet::from_slice(&[[num_g_partitions, total_num_partitions - 1]]))
        ]),
    )
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
