use analytic::partition::integer_partitions::Partition;
use analytic::set::ordered_integer_set::OrderedIntegerSet;
use analytic::set::traits::{Finite, Set};
use biofile::plink_bed::PlinkBed;
use biofile::plink_bim::PlinkBim;
use colored::Colorize;
use ndarray::{Array, array, Ix1, Ix2};
use ndarray_linalg::Solve;
use ndarray_parallel::prelude::*;
use rayon::prelude::*;

use crate::error::Error;
use crate::jackknife::{AdditiveJackknife, Jackknife, JackknifePartitions};
use crate::matrix_ops::{column_normalized_row_ssq, DEFAULT_NUM_SNPS_PER_CHUNK,
                        get_column_mean_and_std, get_gxg_dot_semi_kronecker_z_from_gz_and_ssq,
                        normalized_g_dot_matrix, normalized_g_transpose_dot_matrix, pheno_k_pheno,
                        sum_of_column_wise_inner_product,
};
use crate::partitioned_jackknife_estimates::PartitionedJackknifeEstimates;
use crate::trace_estimator::{
    estimate_gxg_dot_y_norm_sq, estimate_gxg_dot_y_norm_sq_from_basis_bed, estimate_gxg_gram_trace,
    estimate_gxg_kk_trace, estimate_inter_gxg_dot_y_norm_sq_from_basis_bed,
    estimate_tr_gxg_ki_gxg_kj, estimate_tr_k_gxg_k, estimate_tr_kk,
};
use crate::util::matrix_util::{
    generate_plus_minus_one_bernoulli_matrix, normalize_matrix_columns_inplace,
    normalize_vector_inplace,
};
use crate::util::stats_util::{
    n_choose_2, sum_of_squares, sum_of_squares_f32,
};

pub const DEFAULT_PARTITION_NAME: &str = "default_partition";
const GXG_YKY_NUM_RAND_SCALING: usize = 10;

pub fn estimate_heritability(
    geno_bed: PlinkBed,
    geno_bim: PlinkBim,
    mut pheno_arr: Array<f32, Ix1>,
    num_random_vecs: usize,
    num_jackknife_partitions: usize,
) -> Result<PartitionedJackknifeEstimates, String> {
    let partitions = geno_bim.get_fileline_partitions_or(
        DEFAULT_PARTITION_NAME, OrderedIntegerSet::from_slice(&[[0, geno_bed.num_snps - 1]]));
    let partition_array: Vec<Partition> = partitions
        .iter()
        .map(|(_, p)| p.clone())
        .collect();
    let partition_sizes: Vec<usize> = partition_array
        .iter()
        .map(|p| p.size())
        .collect();

    let jackknife_partitions = JackknifePartitions::from_integer_set(
        partition_array.clone(),
        num_jackknife_partitions,
        true);

    let num_partitions = partition_array.len();
    let num_people = geno_bed.num_people;

    println!(
        "num_people: {}\n\
        total_num_snps: {}\n",
        num_people,
        partition_sizes.iter().fold(0, |acc, size| acc + *size)
    );
    partitions.ordered_partition_keys().iter().enumerate().for_each(|(i, k)| {
        println!("partition named {} has {} SNPs", k, partition_sizes[i]);
    });

    println!("\n=> normalizing the phenotype vector");
    normalize_vector_inplace(&mut pheno_arr, 0);

    let yy = sum_of_squares(pheno_arr.iter());
    println!("\n=> yy: {}", yy);

    println!("=> generating ggz_jackknife");
    let random_vecs = generate_plus_minus_one_bernoulli_matrix(num_people, num_random_vecs);
    let ggz_jackknife = get_partitioned_ggz_jackknife(
        &geno_bed,
        &partition_array,
        &jackknife_partitions,
        &random_vecs,
    );

    println!("=> generating ygy_jackknife");
    let ygy_jackknife = get_partitioned_ygy_jackknife(
        &geno_bed,
        &partition_array,
        &jackknife_partitions,
        &pheno_arr,
    );

    let get_heritability_point_estimate = |k: Option<usize>,
                                           jackknife_partition: Option<&Partition>| {
        let (mut a, mut b) = get_normal_eqn_matrices(num_partitions, num_people, yy);
        for i in 0..num_partitions {
            let num_snps_i = match jackknife_partition {
                Some(jackknife_partition) => (
                    partition_sizes[i] - jackknife_partition.intersect(&partition_array[i]).size()
                ) as f64,
                None => partition_sizes[i] as f64,
            };
            let ggz_i = match k {
                Some(k) => ggz_jackknife[i].sum_minus_component(k),
                None => ggz_jackknife[i].get_component_sum().unwrap().clone(),
            };
            a[[i, i]] = sum_of_squares_f32(ggz_i.iter()) as f64
                / num_snps_i
                / num_snps_i
                / num_random_vecs as f64;
            println!("tr(k_{}_k_{})_est: {} num_snps_i: {}", i, i, a[[i, i]], num_snps_i);
            b[i] = match k {
                Some(k) => ygy_jackknife[i].sum_minus_component(k) / num_snps_i,
                None => ygy_jackknife[i].get_component_sum().unwrap() / num_snps_i,
            };
            for j in i + 1..num_partitions {
                let num_snps_j = match jackknife_partition {
                    Some(jackknife_partition) => (
                        partition_sizes[j]
                            - jackknife_partition.intersect(&partition_array[j]).size()
                    ) as f64,
                    None => partition_sizes[j] as f64,
                };
                let ggz_j = match k {
                    Some(k) => ggz_jackknife[j].sum_minus_component(k),
                    None => ggz_jackknife[j].get_component_sum().unwrap().clone(),
                };

                let tr_ki_kj_est = sum_of_column_wise_inner_product(&ggz_i, &ggz_j) as f64
                    / num_snps_i
                    / num_snps_j
                    / num_random_vecs as f64;
                println!("tr(k_{}_k_{})_est: {}", i, j, tr_ki_kj_est);
                a[[i, j]] = tr_ki_kj_est;
                a[[j, i]] = tr_ki_kj_est;
            }
        }
        println!("solving ax=b\na = {:?}\nb = {:?}", a, b);
        let mut sig_sq = a.solve_into(b).unwrap().as_slice().unwrap().to_owned();
        sig_sq.truncate(num_partitions);
        sig_sq
    };
    let mut heritability_estimates = Vec::new();
    for (k, jackknife_partition) in jackknife_partitions.iter().enumerate() {
        println!("\n=> leaving out jackknife partition with index {}", k);
        let sig_sq = get_heritability_point_estimate(Some(k), Some(&jackknife_partition));
        println!("\nsig_sq: {:?}", sig_sq);
        heritability_estimates.push(sig_sq.to_vec());
    }
    let est_without_jackknife = get_heritability_point_estimate(None, None);
    println!("\nest_without_knife: {:?}", est_without_jackknife);
    PartitionedJackknifeEstimates::from_jackknife_estimates(
        &est_without_jackknife,
        &heritability_estimates,
        Some(partitions.ordered_partition_keys().clone()),
        None)
}

pub fn estimate_g_gxg_heritability(
    g_bed: PlinkBed,
    g_bim: PlinkBim,
    gxg_basis_bed: PlinkBed,
    gxg_basis_bim: PlinkBim,
    mut pheno_arr: Array<f32, Ix1>,
    num_rand_vecs_g: usize,
    num_rand_vecs_gxg: usize,
    num_jackknife_partitions: usize,
) -> Result<PartitionedJackknifeEstimates, Error> {
    let g_partitions = g_bim.get_fileline_partitions_or(
        DEFAULT_PARTITION_NAME,
        OrderedIntegerSet::from_slice(&[[0, g_bed.num_snps - 1]]),
    );
    let g_partition_array: Vec<Partition> = g_partitions
        .iter()
        .map(|(_, p)| p.clone())
        .collect();
    let g_partition_sizes: Vec<usize> = g_partition_array
        .iter()
        .map(|p| p.size())
        .collect();

    let gxg_partitions = gxg_basis_bim.get_fileline_partitions_or(
        DEFAULT_PARTITION_NAME,
        OrderedIntegerSet::from_slice(&[[0, gxg_basis_bed.num_snps - 1]]),
    );
    let gxg_partition_array: Vec<Partition> = gxg_partitions
        .iter()
        .map(|(_, p)| p.clone())
        .collect();
    let gxg_partition_sizes: Vec<usize> = gxg_partition_array
        .iter()
        .map(|p| p.size())
        .collect();

    let g_jackknife_partitions = JackknifePartitions::from_integer_set(
        g_partition_array.clone(),
        num_jackknife_partitions,
        true,
    );

    let gxg_basis_jackknife_partitions = JackknifePartitions::from_integer_set(
        gxg_partition_array.clone(),
        num_jackknife_partitions,
        true,
    );

    let num_g_partitions = g_partition_array.len();
    let num_gxg_partitions = gxg_partition_array.len();
    let num_inter_gxg_partitions = n_choose_2(num_gxg_partitions);
    // G partitions, GxG intra-chromosome partitions, GxG inter-chromosome combinations
    let total_num_partitions = num_g_partitions + num_gxg_partitions + num_inter_gxg_partitions;
    let num_people = g_bed.num_people;
    check_and_print_g_and_gxg_partition_info(
        &g_bed,
        &gxg_basis_bed,
        &g_partition_sizes,
        &gxg_partition_sizes,
        num_inter_gxg_partitions,
        g_partitions.ordered_partition_keys(),
        gxg_partitions.ordered_partition_keys(),
    )?;

    normalize_vector_inplace(&mut pheno_arr, 0);
    println!("\n=> normalized the phenotype vector");

    let yy = sum_of_squares(pheno_arr.iter());
    println!("\n=> yy: {}", yy);

    println!("=> generating ggz_jackknife");
    let g_random_vecs = generate_plus_minus_one_bernoulli_matrix(num_people, num_rand_vecs_g);
    let ggz_jackknife = get_partitioned_ggz_jackknife(
        &g_bed,
        &g_partition_array,
        &g_jackknife_partitions,
        &g_random_vecs,
    );

    println!("=> generating gz_jackknife");
    let gz_jackknife = get_partitioned_gz_jackknife(
        &g_bed,
        &g_partition_array,
        &g_jackknife_partitions,
        num_rand_vecs_g,
    );

    println!("=> generating ygy_jackknives");
    let ygy_jackknives = get_partitioned_ygy_jackknife(
        &g_bed,
        &g_partition_array,
        &g_jackknife_partitions,
        &pheno_arr,
    );

    println!("=> generating gxg_gz_jackknife");
    let gxg_gz_jackknife = get_partitioned_gz_jackknife(
        &gxg_basis_bed,
        &gxg_partition_array,
        &gxg_basis_jackknife_partitions,
        num_rand_vecs_gxg,
    );

    println!("=> generating gxg_gu_jackknife");
    let gxg_gu_jackknife = get_partitioned_gz_jackknife(
        &gxg_basis_bed,
        &gxg_partition_array,
        &gxg_basis_jackknife_partitions,
        num_rand_vecs_gxg,
    );

    println!("=> generating gxg_ssq_jackknife");
    let gxg_ssq_jackknife: Vec<AdditiveJackknife<Array<f32, Ix1>>> = gxg_partition_array
        .par_iter()
        .map(|partition| {
            AdditiveJackknife::from_op_over_jackknife_partitions(
                &gxg_basis_jackknife_partitions,
                |_, knife| {
                    column_normalized_row_ssq(&gxg_basis_bed, Some(knife.intersect(partition)), None)
                })
        })
        .collect();

    let mut heritability_estimates = Vec::new();
    let nrv_g = num_rand_vecs_g as f64;
    let nrv_gxg = num_rand_vecs_gxg as f64;
    let get_heritability_point_estimate =
        |k: Option<usize>,
         g_jackknife_range: Option<&Partition>,
         gxg_jackknife_range: Option<&Partition>| {
            let (mut a, mut b) = get_normal_eqn_matrices(total_num_partitions, num_people, yy);
            // g_pairwise_est contains Vec<(
            // tr_kk_est,
            // tr_gk_i_gk_j_est_list,
            // tr_g_gxg_est_list,
            // tr_g_inter_gxg_est_list,
            // yky_est)>
            let g_pairwise_est: Vec<(f64, Vec<f64>, Vec<f64>, Vec<f64>, f64)> = (0..num_g_partitions)
                .collect::<Vec<usize>>()
                .par_iter()
                .map(|&i| {
                    let num_snps_i = partition_minus_knife(
                        &g_partition_array[i], g_jackknife_range,
                    ).size() as f64;
                    let ggz_i = match k {
                        Some(k) => ggz_jackknife[i].sum_minus_component(k),
                        None => ggz_jackknife[i].get_component_sum().unwrap().clone()
                    };

                    let tr_gk_i_gk_j_est_list: Vec<f64> = (i + 1..num_g_partitions)
                        .collect::<Vec<usize>>()
                        .par_iter()
                        .map(|&j| {
                            let num_snps_j = partition_minus_knife(
                                &g_partition_array[j], g_jackknife_range,
                            ).size() as f64;
                            let ggz_j = match k {
                                Some(k) => ggz_jackknife[j].sum_minus_component(k),
                                None => ggz_jackknife[j].get_component_sum().unwrap().clone(),
                            };

                            let tr_ki_kj_est = sum_of_column_wise_inner_product(&ggz_i, &ggz_j) as f64
                                / num_snps_i
                                / num_snps_j
                                / nrv_g;
                            tr_ki_kj_est
                        })
                        .collect();

                    // tr(g_k gxg_k)
                    let gz = match k {
                        Some(k) => gz_jackknife[i].sum_minus_component(k),
                        None => gz_jackknife[i].get_component_sum().unwrap().clone(),
                    };
                    let tr_g_gxg_est_list: Vec<f64> = (0..num_gxg_partitions)
                        .collect::<Vec<usize>>()
                        .par_iter()
                        .map(|&gxg_i| {
                            let num_gxg_snps_i = n_choose_2(
                                partition_minus_knife(
                                    &gxg_partition_array[gxg_i],
                                    gxg_jackknife_range,
                                ).size()
                            ) as f64;
                            let gxg_i_dot_semi_kronecker_z =
                                get_gxg_dot_semi_kronecker_z_from_gz_and_ssq_jackknife(
                                    &gxg_gz_jackknife[gxg_i],
                                    &gxg_ssq_jackknife[gxg_i],
                                    k,
                                );
                            get_mean_ssq_of_z1g1g2z2(&gxg_i_dot_semi_kronecker_z, &gz)
                                / num_snps_i
                                / num_gxg_snps_i
                        })
                        .collect();

                    let tr_g_inter_gxg_est_list: Vec<f64> = (0..num_gxg_partitions)
                        .collect::<Vec<usize>>()
                        .par_iter()
                        .flat_map(|&gxg_i| {
                            let num_gxg_basis_i = partition_minus_knife(
                                &gxg_partition_array[gxg_i],
                                gxg_jackknife_range,
                            ).size();
                            (gxg_i + 1..num_gxg_partitions)
                                .collect::<Vec<usize>>()
                                .par_iter()
                                .map(|&gxg_j| {
                                    let num_inter_gxg_snps = (
                                        num_gxg_basis_i
                                            * partition_minus_knife(
                                            &gxg_partition_array[gxg_j],
                                            gxg_jackknife_range,
                                        ).size()
                                    ) as f64;

                                    get_mean_ssq_of_z1g1g2z2(
                                        &gz,
                                        &get_inter_chrom_gxg_zz_from_gz_gz_jackknife(
                                            &gxg_gz_jackknife[gxg_i],
                                            &gxg_gz_jackknife[gxg_j],
                                            k,
                                        ),
                                    ) / num_inter_gxg_snps / num_snps_i
                                })
                                .collect::<Vec<f64>>()
                        })
                        .collect();

                    let yky_est = match k {
                        Some(k) => ygy_jackknives[i].sum_minus_component(k) / num_snps_i,
                        None => ygy_jackknives[i].get_component_sum().unwrap() / num_snps_i,
                    };
                    (sum_of_squares_f32(ggz_i.iter()) as f64 / num_snps_i / num_snps_i / nrv_g,
                     tr_gk_i_gk_j_est_list,
                     tr_g_gxg_est_list,
                     tr_g_inter_gxg_est_list,
                     yky_est)
                })
                .collect();

            for (i, (
                tr_kk_est,
                tr_gk_i_gk_j_est_list,
                tr_g_gxg_est_list,
                tr_g_inter_gxg_est_list,
                yky_est)
            ) in g_pairwise_est
                .into_iter()
                .enumerate() {
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
                for (inter_gxg_ij, tr_g_inter_gxg_est) in tr_g_inter_gxg_est_list.into_iter().enumerate() {
                    let global_inter_gxg_ij = num_g_partitions + num_gxg_partitions + inter_gxg_ij;
                    a[[global_inter_gxg_ij, i]] = tr_g_inter_gxg_est;
                    a[[i, global_inter_gxg_ij]] = tr_g_inter_gxg_est;
                    println!("tr_g_k{}_inter_gxg_k{}_est: {}", i, inter_gxg_ij, tr_g_inter_gxg_est);
                }
            }

            // gxg_pairwise_est contains Vec<(
            // tr_gxg_ki_est,
            // tr_gxg_kki_est,
            // tr_gxg_ki_kj_est_list,
            // tr_gxg_inter_gxg_est_list,
            // y_gxg_ki_y_est)>
            let gxg_pairwise_est: Vec<(f64, f64, Vec<f64>, Vec<f64>, f64)> = (0..num_gxg_partitions)
                .collect::<Vec<usize>>()
                .par_iter()
                .map(|&i| {
                    let range_i = match gxg_jackknife_range {
                        None => gxg_partition_array[i].clone(),
                        Some(r) => gxg_partition_array[i].clone() - r,
                    };
                    let num_gxg_snps_i = n_choose_2(range_i.size()) as f64;

                    let gxg_i_dot_semi_kronecker_z =
                        get_gxg_dot_semi_kronecker_z_from_gz_and_ssq_jackknife(
                            &gxg_gz_jackknife[i],
                            &gxg_ssq_jackknife[i],
                            k,
                        );
                    let gxg_i_dot_semi_kronecker_u =
                        get_gxg_dot_semi_kronecker_z_from_gz_and_ssq_jackknife(
                            &gxg_gu_jackknife[i],
                            &gxg_ssq_jackknife[i],
                            k,
                        );
                    let tr_gxg_ki_kj_est_list: Vec<f64> = (i + 1..num_gxg_partitions)
                        .collect::<Vec<usize>>()
                        .par_iter()
                        .map(|&j| {
                            let num_gxg_snps_j = n_choose_2(
                                partition_minus_knife(&gxg_partition_array[j], gxg_jackknife_range).size()
                            ) as f64;
                            let gxg_j_dot_semi_kronecker_z =
                                get_gxg_dot_semi_kronecker_z_from_gz_and_ssq_jackknife(
                                    &gxg_gu_jackknife[j],
                                    &gxg_ssq_jackknife[j],
                                    k,
                                );

                            get_mean_ssq_of_z1g1g2z2(
                                &gxg_i_dot_semi_kronecker_z,
                                &gxg_j_dot_semi_kronecker_z,
                            ) / num_gxg_snps_i / num_gxg_snps_j
                        })
                        .collect();

                    let tr_gxg_inter_gxg_est_list: Vec<f64> = (0..num_gxg_partitions)
                        .collect::<Vec<usize>>()
                        .par_iter()
                        .flat_map(|&ii| {
                            let num_snps_ii = partition_minus_knife(
                                &gxg_partition_array[ii],
                                gxg_jackknife_range,
                            ).size();
                            (ii + 1..num_gxg_partitions)
                                .collect::<Vec<usize>>()
                                .par_iter()
                                .map(|&jj| {
                                    let num_inter_gxg_snps = (
                                        num_snps_ii
                                            * partition_minus_knife(
                                            &gxg_partition_array[jj],
                                            gxg_jackknife_range,
                                        ).size()
                                    ) as f64;

                                    get_mean_ssq_of_z1g1g2z2(
                                        &gxg_i_dot_semi_kronecker_z,
                                        &get_inter_chrom_gxg_zz_from_gz_gz_jackknife(
                                            &gxg_gu_jackknife[ii],
                                            &gxg_gu_jackknife[jj],
                                            k,
                                        ),
                                    ) / num_gxg_snps_i / num_inter_gxg_snps
                                })
                                .collect::<Vec<f64>>()
                        })
                        .collect();

                    let (snp_mean_i, snp_std_i) = get_column_mean_and_std(
                        &gxg_basis_bed,
                        &range_i,
                        DEFAULT_NUM_SNPS_PER_CHUNK,
                    );
                    let y_gxg_k_y_est = estimate_gxg_dot_y_norm_sq_from_basis_bed(
                        &gxg_basis_bed,
                        Some(range_i),
                        &snp_mean_i,
                        &snp_std_i,
                        &pheno_arr,
                        num_rand_vecs_gxg * GXG_YKY_NUM_RAND_SCALING,
                    ) / num_gxg_snps_i;

                    (sum_of_squares_f32(gxg_i_dot_semi_kronecker_z.iter()) as f64 / num_gxg_snps_i / nrv_gxg,
                     get_mean_ssq_of_z1g1g2z2(&gxg_i_dot_semi_kronecker_z, &gxg_i_dot_semi_kronecker_u)
                         / num_gxg_snps_i
                         / num_gxg_snps_i,
                     tr_gxg_ki_kj_est_list,
                     tr_gxg_inter_gxg_est_list,
                     y_gxg_k_y_est)
                })
                .collect();

            for (i, (
                tr_gxg_ki_est,
                tr_gxg_kki_est,
                tr_gxg_ki_kj_est_list,
                tr_gxg_inter_gxg_est_list,
                y_gxg_k_y_est)
            ) in gxg_pairwise_est
                .into_iter()
                .enumerate() {
                let global_i = num_g_partitions + i;
                a[[global_i, total_num_partitions]] = tr_gxg_ki_est;
                a[[total_num_partitions, global_i]] = tr_gxg_ki_est;
                a[[global_i, global_i]] = tr_gxg_kki_est;
                b[global_i] = y_gxg_k_y_est;
                println!("tr_gxg_k{}_est: {}", i, tr_gxg_ki_est);
                println!("tr_gxg_kk{}_est: {}", i, tr_gxg_kki_est);
                println!("tr_y_gxg_k{}_y_est: {}", i, y_gxg_k_y_est);
                for (j, tr_gxg_i_gxg_j_est) in tr_gxg_ki_kj_est_list.into_iter().enumerate() {
                    let global_j = num_g_partitions + i + 1 + j;
                    a[[global_i, global_j]] = tr_gxg_i_gxg_j_est;
                    a[[global_j, global_i]] = tr_gxg_i_gxg_j_est;
                    println!("tr_gxg_k{}_gxg_k{}: {}", i, i + 1 + j, tr_gxg_i_gxg_j_est);
                }
                for (inter_gxg_ij, tr_gxg_ki_inter_gxg_kij_est) in tr_gxg_inter_gxg_est_list
                    .into_iter()
                    .enumerate() {
                    let global_inter_gxg_ij = num_g_partitions + num_gxg_partitions + inter_gxg_ij;
                    a[[global_i, global_inter_gxg_ij]] = tr_gxg_ki_inter_gxg_kij_est;
                    a[[global_inter_gxg_ij, global_i]] = tr_gxg_ki_inter_gxg_kij_est;
                    println!("tr_gxg_k{}_inter_gxg_k{}: {}", i, inter_gxg_ij, tr_gxg_ki_inter_gxg_kij_est);
                }
            }

            // inter_gxg_pairwise_est contains Vec<(
            // tr_inter_k_ij_est,
            // tr_inter_kk_ij_est,
            // tr_inter_i1j1_i2j2_list,
            // y_gxg_k_y_est)>
            let inter_gxg_pairwise_est: Vec<(f64, f64, Vec<f64>, f64)> = (0..num_gxg_partitions)
                .collect::<Vec<usize>>()
                .par_iter()
                .flat_map(|&i1| {
                    let range_i1 = partition_minus_knife(
                        &gxg_partition_array[i1],
                        gxg_jackknife_range,
                    );
                    let (snp_mean_i1, snp_std_i1) = get_column_mean_and_std(
                        &gxg_basis_bed,
                        &range_i1,
                        DEFAULT_NUM_SNPS_PER_CHUNK,
                    );
                    (i1 + 1..num_gxg_partitions)
                        .collect::<Vec<usize>>()
                        .par_iter()
                        .map(|&j1| {
                            let range_j1 = partition_minus_knife(
                                &gxg_partition_array[j1],
                                gxg_jackknife_range,
                            );
                            let num_gxg_snps_i1j1 = (range_i1.size() * range_j1.size()) as f64;

                            let inter_chrom_gxg_zz_i1j1 = get_inter_chrom_gxg_zz_from_gz_gz_jackknife(
                                &gxg_gz_jackknife[i1],
                                &gxg_gz_jackknife[j1],
                                k,
                            );
                            let tr_inter_kk_ij_est = get_mean_ssq_of_z1g1g2z2(
                                &inter_chrom_gxg_zz_i1j1,
                                &get_inter_chrom_gxg_zz_from_gz_gz_jackknife(
                                    &gxg_gu_jackknife[i1],
                                    &gxg_gu_jackknife[j1],
                                    k,
                                ),
                            )
                                / num_gxg_snps_i1j1
                                / num_gxg_snps_i1j1;

                            let tr_inter_i1j1_i2j2_list: Vec<f64> = (i1..num_gxg_partitions)
                                .collect::<Vec<usize>>()
                                .par_iter()
                                .flat_map(|&i2| {
                                    let range_i2 = partition_minus_knife(
                                        &gxg_partition_array[i2],
                                        gxg_jackknife_range,
                                    );
                                    let j2_start = if i1 == i2 { j1 + 1 } else { i2 + 1 };
                                    (j2_start..num_gxg_partitions)
                                        .collect::<Vec<usize>>()
                                        .par_iter()
                                        .map(|&j2| {
                                            let range_j2 = partition_minus_knife(
                                                &gxg_partition_array[j2],
                                                gxg_jackknife_range,
                                            );
                                            let num_gxg_snps_i2j2 = (range_i2.size() * range_j2.size()) as f64;

                                            get_mean_ssq_of_z1g1g2z2(
                                                &inter_chrom_gxg_zz_i1j1,
                                                &get_inter_chrom_gxg_zz_from_gz_gz_jackknife(
                                                    &gxg_gu_jackknife[i2],
                                                    &gxg_gu_jackknife[j2],
                                                    k,
                                                ),
                                            )
                                                / num_gxg_snps_i1j1
                                                / num_gxg_snps_i2j2
                                        })
                                        .collect::<Vec<f64>>()
                                })
                                .collect();

                            let (snp_mean_j1, snp_std_j1) = get_column_mean_and_std(
                                &gxg_basis_bed,
                                &range_j1,
                                DEFAULT_NUM_SNPS_PER_CHUNK,
                            );

                            let y_gxg_k_y_est = estimate_inter_gxg_dot_y_norm_sq_from_basis_bed(
                                &gxg_basis_bed,
                                Some(range_i1.clone()),
                                Some(range_j1),
                                &snp_mean_i1,
                                &snp_std_i1,
                                &snp_mean_j1,
                                &snp_std_j1,
                                &pheno_arr,
                                num_rand_vecs_gxg,
                            ) / num_gxg_snps_i1j1;

                            (
                                sum_of_squares_f32(inter_chrom_gxg_zz_i1j1.iter()) as f64 / nrv_gxg / num_gxg_snps_i1j1,
                                tr_inter_kk_ij_est,
                                tr_inter_i1j1_i2j2_list,
                                y_gxg_k_y_est
                            )
                        })
                        .collect::<Vec<(f64, f64, Vec<f64>, f64)>>()
                })
                .collect();
            for (i1j1, (
                tr_inter_k_ij_est,
                tr_inter_kk_ij_est,
                tr_inter_i1j1_i2j2_list,
                y_gxg_k_y_est)
            ) in inter_gxg_pairwise_est.into_iter().enumerate() {
                let global_ij = num_g_partitions + num_gxg_partitions + i1j1;
                a[[global_ij, global_ij]] = tr_inter_kk_ij_est;
                a[[global_ij, total_num_partitions]] = tr_inter_k_ij_est;
                a[[total_num_partitions, global_ij]] = tr_inter_k_ij_est;
                b[global_ij] = y_gxg_k_y_est;
                println!("tr_inter_gg_k{}_est: {}", i1j1, tr_inter_k_ij_est);
                println!("tr_inter_gg_kk{}_est: {}", i1j1, tr_inter_kk_ij_est);
                println!("tr_inter_gg_y_k{}_y_est: {}", i1j1, y_gxg_k_y_est);
                for (i2j2, est) in tr_inter_i1j1_i2j2_list.into_iter().enumerate() {
                    let global_i2j2 = num_g_partitions + num_gxg_partitions + i1j1 + 1 + i2j2;
                    a[[global_ij, global_i2j2]] = est;
                    a[[global_i2j2, global_ij]] = est;
                    println!("tr_inter_gg_k{}_k{}_est: {}", i1j1, i2j2, est);
                }
            }
            println!("solving A={:?} b={:?}", a, b);
            let mut sig_sq = a.solve_into(b).unwrap().as_slice().unwrap().to_owned();
            sig_sq.truncate(total_num_partitions);
            sig_sq
        };
    for (k, (g_jackknife_range, gxg_jackknife_range)) in g_jackknife_partitions
        .iter()
        .zip(gxg_basis_jackknife_partitions.iter())
        .enumerate() {
        println!("\n=> leaving out jackknife partition with index {}", k);
        let sig_sq = get_heritability_point_estimate(
            Some(k),
            Some(&g_jackknife_range),
            Some(&gxg_jackknife_range),
        );
        println!("\nsig_sq: {:?}", sig_sq);
        heritability_estimates.push(sig_sq.to_vec());
    }

    println!("\n=> Computing heritability without Jackknife");
    let est_without_knife = get_heritability_point_estimate(None, None, None);
    println!("\nest_without_knife: {:?}", est_without_knife);

    let mut total_partition_keys: Vec<String> = g_partitions
        .ordered_partition_keys()
        .iter()
        .map(|k| format!("G {}", k))
        .collect();
    total_partition_keys.append(
        &mut gxg_partitions
            .ordered_partition_keys()
            .iter()
            .map(|k| format!("GxG {}", k))
            .collect::<Vec<String>>()
    );
    total_partition_keys.append(
        &mut gxg_partitions
            .ordered_partition_keys()
            .iter()
            .enumerate()
            .flat_map(|(i, k1)| {
                gxg_partitions
                    .ordered_partition_keys()
                    .iter()
                    .skip(i + 1)
                    .map(|k2| format!("G({})xG({})", k1, k2))
                    .collect::<Vec<String>>()
            })
            .collect()
    );

    Ok(PartitionedJackknifeEstimates::from_jackknife_estimates(
        &est_without_knife,
        &heritability_estimates,
        Some(total_partition_keys),
        Some(vec![
            ("G".to_string(), OrderedIntegerSet::from_slice(&[[0, num_g_partitions - 1]])),
            ("GxG".to_string(), OrderedIntegerSet::from_slice(&[[
                num_g_partitions, num_g_partitions + num_gxg_partitions - 1
            ]])),
            ("inter-chromosome GxG".to_string(), OrderedIntegerSet::from_slice(&[[
                num_g_partitions + num_gxg_partitions, total_num_partitions - 1
            ]]))
        ]),
    )?)
}

fn check_and_print_g_and_gxg_partition_info(
    g_bed: &PlinkBed,
    gxg_basis_bed: &PlinkBed,
    g_partition_sizes: &Vec<usize>,
    gxg_partition_sizes: &Vec<usize>,
    num_inter_gxg_partitions: usize,
    g_partition_names: &Vec<String>,
    gxg_partition_names: &Vec<String>,
) -> Result<(), Error> {
    if g_bed.num_people != gxg_basis_bed.num_people {
        return Err(
            Error::Generic(
                format!("g_bed has {} people but gxg_basis_bed has {} people",
                        g_bed.num_people, gxg_basis_bed.num_people)
            )
        );
    }
    println!(
        "num_people: {}\n\
        total_num_g_snps: {}\n\
        total_num_gxg_basis_snps: {}\n\
        num_inter_gxg_partitions: {}",
        g_bed.num_people,
        g_partition_sizes.iter().fold(0, |acc, size| acc + *size),
        gxg_partition_sizes.iter().fold(0, |acc, size| acc + *size),
        num_inter_gxg_partitions
    );
    g_partition_names.iter().enumerate().for_each(|(i, k)| {
        println!("G partition named {} has {} SNPs", k, g_partition_sizes[i]);
    });
    gxg_partition_names.iter().enumerate().for_each(|(i, k)| {
        println!("GxG partition named {} has {} SNPs", k, gxg_partition_sizes[i]);
    });
    Ok(())
}

fn get_normal_eqn_matrices(
    num_partitions: usize,
    num_people: usize,
    yy: f64,
) -> (Array<f64, Ix2>, Array<f64, Ix1>) {
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

fn partition_minus_knife(
    partition_range: &Partition,
    knife: Option<&Partition>)
    -> Partition {
    match knife {
        Some(r) => partition_range.clone() - r,
        None => partition_range.clone()
    }
}

fn get_gxg_dot_semi_kronecker_z_from_gz_and_ssq_jackknife(
    gz_jackknife: &AdditiveJackknife<Array<f32, Ix2>>,
    g_ssq_jackknife: &AdditiveJackknife<Array<f32, Ix1>>,
    jackknife_leave_out_index: Option<usize>,
) -> Array<f32, Ix2> {
    match jackknife_leave_out_index {
        Some(jackknife_leave_out_index) => get_gxg_dot_semi_kronecker_z_from_gz_and_ssq(
            gz_jackknife.sum_minus_component(jackknife_leave_out_index),
            &g_ssq_jackknife.sum_minus_component(jackknife_leave_out_index),
        ),
        None => get_gxg_dot_semi_kronecker_z_from_gz_and_ssq(
            gz_jackknife.get_component_sum().unwrap().clone(),
            &g_ssq_jackknife.get_component_sum().unwrap(),
        ),
    }
}

fn get_inter_chrom_gxg_zz_from_gz_gz_jackknife(
    gi_zi_jackknife: &AdditiveJackknife<Array<f32, Ix2>>,
    gj_zj_jackknife: &AdditiveJackknife<Array<f32, Ix2>>,
    jackknife_leave_out_index: Option<usize>,
) -> Array<f32, Ix2> {
    match jackknife_leave_out_index {
        Some(k) => {
            gi_zi_jackknife.sum_minus_component(k)
                * gj_zj_jackknife.sum_minus_component(k)
        }
        None => {
            gi_zi_jackknife.get_component_sum().unwrap()
                * gj_zj_jackknife.get_component_sum().unwrap()
        }
    }
}

/// `g1z1` has shape (num_people x num_rand_vecs_1)
/// `g2z2` has shape (num_people x num_rand_vecs_2)
fn get_mean_ssq_of_z1g1g2z2(
    g1z1: &Array<f32, Ix2>,
    g2z2: &Array<f32, Ix2>,
) -> f64 {
    let b1 = g1z1.dim().1;
    let b2 = g2z2.dim().1;
    sum_of_squares(g1z1.t().dot(g2z2).iter())
        / b1 as f64
        / b2 as f64
}

fn get_partitioned_gz_jackknife(bed: &PlinkBed,
                                snp_partition_array: &Vec<Partition>,
                                jackknife_partitions: &JackknifePartitions,
                                num_rand_vecs: usize) -> Vec<AdditiveJackknife<Array<f32, Ix2>>> {
    snp_partition_array.par_iter().map(|partition| {
        AdditiveJackknife::from_op_over_jackknife_partitions(
            jackknife_partitions,
            |_, knife| {
                let range_intersect = knife.intersect(partition);
                let range_size = range_intersect.size();
                let (snp_mean, snp_std) = get_column_mean_and_std(
                    bed,
                    &range_intersect,
                    DEFAULT_NUM_SNPS_PER_CHUNK,
                );
                normalized_g_dot_matrix(
                    bed,
                    Some(range_intersect),
                    &snp_mean,
                    &snp_std,
                    &generate_plus_minus_one_bernoulli_matrix(range_size, num_rand_vecs),
                    None,
                    Some(2048),
                )
            })
    }).collect::<Vec<AdditiveJackknife<Array<f32, Ix2>>>>()
}

fn get_partitioned_ggz_jackknife(
    bed: &PlinkBed,
    snp_partition_array: &Vec<Partition>,
    jackknife_partitions: &JackknifePartitions,
    rand_vecs: &Array<f32, Ix2>,
) -> Vec<AdditiveJackknife<Array<f32, Ix2>>> {
    snp_partition_array.par_iter().map(|partition| {
        AdditiveJackknife::from_op_over_jackknife_partitions(&jackknife_partitions, |_, knife| {
            let range_intersect = knife.intersect(partition);
            let (snp_mean, snp_std) = get_column_mean_and_std(
                &bed,
                &range_intersect,
                DEFAULT_NUM_SNPS_PER_CHUNK,
            );
            let gtz = normalized_g_transpose_dot_matrix(
                &bed,
                Some(range_intersect.clone()),
                &snp_mean,
                &snp_std,
                &rand_vecs,
                None,
            );
            normalized_g_dot_matrix(
                &bed,
                Some(range_intersect),
                &snp_mean,
                &snp_std,
                &gtz,
                None,
                Some(2048),
            )
        })
    }).collect()
}

fn get_partitioned_ygy_jackknife(
    bed: &PlinkBed,
    snp_partition_array: &Vec<Partition>,
    jackknife_partitions: &JackknifePartitions,
    pheno_arr: &Array<f32, Ix1>,
) -> Vec<AdditiveJackknife<f64>> {
    snp_partition_array.par_iter().map(|partition| {
        let means_and_stds_jackknife = Jackknife::from_op_over_jackknife_partitions(
            jackknife_partitions, |knife| get_column_mean_and_std(
                bed,
                &knife.intersect(partition),
                DEFAULT_NUM_SNPS_PER_CHUNK,
            ),
        );
        AdditiveJackknife::from_op_over_jackknife_partitions(jackknife_partitions, |k, knife| {
            let range = knife.intersect(partition);
            let num_snps_in_range = range.size() as f64;
            pheno_k_pheno(
                pheno_arr,
                &range,
                bed,
                &means_and_stds_jackknife.components[k].0,
                &means_and_stds_jackknife.components[k].1,
                DEFAULT_NUM_SNPS_PER_CHUNK,
            ) * num_snps_in_range
        })
    }).collect()
}

#[inline]
fn bold_print(msg: &String) {
    println!("{}", msg.bold());
}

/// Similar to `estimate_g_gxg_heritability`, but without forming the inter-chromosome
/// components.
/// Kept here for analysis and debugging purposes
#[deprecated]
pub fn estimate_g_intra_gxg_heritability(
    g_bed: PlinkBed,
    g_bim: PlinkBim,
    gxg_basis_bed: PlinkBed,
    gxg_basis_bim: PlinkBim,
    mut pheno_arr: Array<f32, Ix1>,
    num_rand_vecs_g: usize,
    num_rand_vecs_gxg: usize,
    num_jackknife_partitions: usize,
) -> Result<PartitionedJackknifeEstimates, Error> {
    let g_partitions = g_bim.get_fileline_partitions_or(
        DEFAULT_PARTITION_NAME, OrderedIntegerSet::from_slice(&[[0, g_bed.num_snps - 1]]),
    );
    let g_partition_array: Vec<Partition> = g_partitions
        .iter()
        .map(|(_, p)| p.clone())
        .collect();
    let g_partition_sizes: Vec<usize> = g_partition_array
        .iter()
        .map(|p| p.size())
        .collect();

    let gxg_partitions = gxg_basis_bim.get_fileline_partitions_or(
        DEFAULT_PARTITION_NAME, OrderedIntegerSet::from_slice(&[[0, gxg_basis_bed.num_snps - 1]]),
    );
    let gxg_partition_array: Vec<Partition> = gxg_partitions
        .iter()
        .map(|(_, p)| p.clone())
        .collect();
    let gxg_partition_sizes: Vec<usize> = gxg_partition_array
        .iter()
        .map(|p| p.size())
        .collect();

    let g_jackknife_partitions = JackknifePartitions::from_integer_set(
        g_partition_array.clone(),
        num_jackknife_partitions,
        true,
    );

    let gxg_basis_jackknife_partitions = JackknifePartitions::from_integer_set(
        gxg_partition_array.clone(),
        num_jackknife_partitions,
        true,
    );

    let num_g_partitions = g_partition_array.len();
    let num_gxg_partitions = gxg_partition_array.len();
    let total_num_partitions = num_g_partitions + num_gxg_partitions;
    let num_people = g_bed.num_people;
    check_and_print_g_and_gxg_partition_info(
        &g_bed,
        &gxg_basis_bed,
        &g_partition_sizes,
        &gxg_partition_sizes,
        0,
        g_partitions.ordered_partition_keys(),
        gxg_partitions.ordered_partition_keys(),
    )?;

    normalize_vector_inplace(&mut pheno_arr, 0);
    println!("\n=> normalized the phenotype vector");

    let yy = sum_of_squares(pheno_arr.iter());
    println!("\n=> yy: {}", yy);

    println!("=> generating ggz_jackknife");
    let g_random_vecs = generate_plus_minus_one_bernoulli_matrix(num_people, num_rand_vecs_g);
    let ggz_jackknife = get_partitioned_ggz_jackknife(
        &g_bed,
        &g_partition_array,
        &g_jackknife_partitions,
        &g_random_vecs,
    );

    println!("=> generating gz_jackknife");
    let gz_jackknife = get_partitioned_gz_jackknife(
        &g_bed,
        &g_partition_array,
        &g_jackknife_partitions,
        num_rand_vecs_g,
    );

    println!("=> generating ygy_jackknives");
    let ygy_jackknives = get_partitioned_ygy_jackknife(
        &g_bed,
        &g_partition_array,
        &g_jackknife_partitions,
        &pheno_arr,
    );

    println!("=> generating gxg_gz_jackknife");
    let gxg_gz_jackknife = get_partitioned_gz_jackknife(
        &gxg_basis_bed,
        &gxg_partition_array,
        &gxg_basis_jackknife_partitions,
        num_rand_vecs_gxg);

    println!("=> generating gxg_gu_jackknife");
    let gxg_gu_jackknife = get_partitioned_gz_jackknife(
        &gxg_basis_bed,
        &gxg_partition_array,
        &gxg_basis_jackknife_partitions,
        num_rand_vecs_gxg);

    println!("=> generating gxg_ssq_jackknife");
    let gxg_ssq_jackknife: Vec<AdditiveJackknife<Array<f32, Ix1>>> = gxg_partition_array
        .par_iter()
        .map(|partition| {
            AdditiveJackknife::from_op_over_jackknife_partitions(
                &gxg_basis_jackknife_partitions,
                |_, knife| {
                    column_normalized_row_ssq(&gxg_basis_bed, Some(knife.intersect(partition)), None)
                })
        }).collect();

    let mut heritability_estimates = Vec::new();
    let nrv_g = num_rand_vecs_g as f64;
    let nrv_gxg = num_rand_vecs_gxg as f64;
    let get_heritability_point_estimate =
        |k: Option<usize>,
         g_jackknife_range: Option<&Partition>,
         gxg_jackknife_range: Option<&Partition>| {
            let (mut a, mut b) = get_normal_eqn_matrices(total_num_partitions, num_people, yy);
            // g_pairwise_est contains Vec<(str_kk_est, tr_gk_i_gk_j_est_list, tr_g_gxg_est_list, yky_est)>
            let g_pairwise_est: Vec<(f64, Vec<f64>, Vec<f64>, f64)> = (0..num_g_partitions)
                .collect::<Vec<usize>>().par_iter().map(|&i| {
                let num_snps_i = match g_jackknife_range {
                    Some(g_jackknife_range) => (
                        g_partition_sizes[i] - g_jackknife_range.intersect(&g_partition_array[i]).size()
                    ) as f64,
                    None => g_partition_sizes[i] as f64,
                };

                let ggz_i = match k {
                    Some(k) => ggz_jackknife[i].sum_minus_component(k),
                    None => ggz_jackknife[i].get_component_sum().unwrap().clone()
                };

                let tr_gk_i_gk_j_est_list: Vec<f64> = (i + 1..num_g_partitions)
                    .collect::<Vec<usize>>().par_iter().map(|&j| {
                    let num_snps_j = match g_jackknife_range {
                        Some(g_jackknife_range) => (
                            g_partition_sizes[j] - g_jackknife_range.intersect(&g_partition_array[j]).size()
                        ) as f64,
                        None => g_partition_sizes[j] as f64,
                    };
                    let ggz_j = match k {
                        Some(k) => ggz_jackknife[j].sum_minus_component(k),
                        None => ggz_jackknife[j].get_component_sum().unwrap().clone(),
                    };

                    let tr_ki_kj_est = sum_of_column_wise_inner_product(&ggz_i, &ggz_j) as f64
                        / num_snps_i
                        / num_snps_j
                        / nrv_g;
                    tr_ki_kj_est
                }).collect();

                // tr(g_k gxg_k)
                let gz = match k {
                    Some(k) => gz_jackknife[i].sum_minus_component(k),
                    None => gz_jackknife[i].get_component_sum().unwrap().clone(),
                };
                let tr_g_gxg_est_list: Vec<f64> = (0..num_gxg_partitions)
                    .collect::<Vec<usize>>()
                    .par_iter()
                    .map(|&gxg_i| {
                        let num_gxg_snps_i = match gxg_jackknife_range {
                            Some(gxg_jackknife_range) => n_choose_2(
                                gxg_partition_sizes[gxg_i]
                                    - gxg_jackknife_range.intersect(&gxg_partition_array[gxg_i]).size()
                            ) as f64,
                            None => n_choose_2(gxg_partition_sizes[gxg_i]) as f64,
                        };
                        let gxg_i_dot_semi_kronecker_z =
                            get_gxg_dot_semi_kronecker_z_from_gz_and_ssq_jackknife(
                                &gxg_gz_jackknife[gxg_i],
                                &gxg_ssq_jackknife[gxg_i],
                                k,
                            );
                        sum_of_squares_f32(
                            gxg_i_dot_semi_kronecker_z.t().dot(&gz).iter()
                        ) as f64
                            / num_gxg_snps_i
                            / num_snps_i
                            / nrv_g
                            / nrv_gxg
                    }).collect();

                let yky_est = match k {
                    Some(k) => ygy_jackknives[i].sum_minus_component(k) / num_snps_i,
                    None => ygy_jackknives[i].get_component_sum().unwrap() / num_snps_i,
                };
                (sum_of_squares_f32(ggz_i.iter()) as f64 / num_snps_i / num_snps_i / nrv_g,
                 tr_gk_i_gk_j_est_list,
                 tr_g_gxg_est_list,
                 yky_est)
            }).collect();

            for (i, (tr_kk_est, tr_gk_i_gk_j_est_list, tr_g_gxg_est_list, yky_est)) in g_pairwise_est
                .into_iter()
                .enumerate() {
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

            // gxg_pairwise_est contains Vec<(tr_gxg_ki_est, tr_gxg_kk_est, gxg_upper_triangular, y_gxg_k_y_est)>
            let gxg_pairwise_est: Vec<(f64, f64, Vec<f64>, f64)> = (0..num_gxg_partitions)
                .collect::<Vec<usize>>()
                .par_iter()
                .map(|&i| {
                    let range = match gxg_jackknife_range {
                        None => gxg_partition_array[i].clone(),
                        Some(r) => gxg_partition_array[i].clone() - r,
                    };
                    let num_gxg_snps_i = n_choose_2(range.size()) as f64;

                    let gxg_i_dot_semi_kronecker_z = get_gxg_dot_semi_kronecker_z_from_gz_and_ssq_jackknife(
                        &gxg_gz_jackknife[i],
                        &gxg_ssq_jackknife[i],
                        k,
                    );
                    let gxg_i_dot_semi_kronecker_u = get_gxg_dot_semi_kronecker_z_from_gz_and_ssq_jackknife(
                        &gxg_gu_jackknife[i],
                        &gxg_ssq_jackknife[i],
                        k,
                    );
                    let gxg_upper_triangular: Vec<f64> = (i + 1..num_gxg_partitions)
                        .collect::<Vec<usize>>().par_iter().map(|&j| {
                        let num_gxg_snps_j = match gxg_jackknife_range {
                            Some(gxg_jackknife_range) => n_choose_2(
                                gxg_partition_sizes[j] - gxg_jackknife_range.intersect(&gxg_partition_array[j]).size()
                            ) as f64,
                            None => n_choose_2(gxg_partition_sizes[j]) as f64,
                        };
                        let gxg_j_dot_semi_kronecker_z = get_gxg_dot_semi_kronecker_z_from_gz_and_ssq_jackknife(
                            &gxg_gu_jackknife[j],
                            &gxg_ssq_jackknife[j],
                            k,
                        );
                        sum_of_squares_f32(
                            gxg_i_dot_semi_kronecker_z.t().dot(&gxg_j_dot_semi_kronecker_z).iter()
                        ) as f64
                            / num_gxg_snps_i
                            / num_gxg_snps_j
                            / nrv_gxg
                            / nrv_gxg
                    }).collect();

                    let (snp_mean_i, snp_std_i) = get_column_mean_and_std(
                        &gxg_basis_bed,
                        &range,
                        DEFAULT_NUM_SNPS_PER_CHUNK,
                    );
                    let y_gxg_k_y_est = estimate_gxg_dot_y_norm_sq_from_basis_bed(
                        &gxg_basis_bed,
                        Some(range),
                        &snp_mean_i,
                        &snp_std_i,
                        &pheno_arr,
                        num_rand_vecs_gxg * GXG_YKY_NUM_RAND_SCALING,
                    ) / num_gxg_snps_i;

                    (sum_of_squares_f32(gxg_i_dot_semi_kronecker_z.iter()) as f64 / num_gxg_snps_i / nrv_gxg,
                     sum_of_squares_f32(gxg_i_dot_semi_kronecker_z.t().dot(&gxg_i_dot_semi_kronecker_u).iter()) as f64
                         / num_gxg_snps_i
                         / num_gxg_snps_i
                         / nrv_gxg
                         / nrv_gxg,
                     gxg_upper_triangular,
                     y_gxg_k_y_est)
                }).collect();

            for (i, (tr_gxg_ki_est, tr_gxg_kk_est, gxg_upper_triangular, y_gxg_k_y_est)) in gxg_pairwise_est
                .into_iter()
                .enumerate() {
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
            sig_sq
        };
    for (k, (g_jackknife_range, gxg_jackknife_range)) in g_jackknife_partitions
        .iter()
        .zip(gxg_basis_jackknife_partitions.iter())
        .enumerate() {
        println!("\n=> leaving out jackknife partition with index {}", k);
        let sig_sq = get_heritability_point_estimate(Some(k), Some(&g_jackknife_range), Some(&gxg_jackknife_range));
        println!("\nsig_sq: {:?}", sig_sq);
        heritability_estimates.push(sig_sq.to_vec());
    }

    println!("\n=> Computing heritability without Jackknife");
    let est_without_knife = get_heritability_point_estimate(None, None, None);
    println!("\nest_without_knife: {:?}", est_without_knife);

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
    Ok(PartitionedJackknifeEstimates::from_jackknife_estimates(
        &est_without_knife,
        &heritability_estimates,
        Some(total_partition_keys),
        Some(vec![
            ("G".to_string(), OrderedIntegerSet::from_slice(&[[0, num_g_partitions - 1]])),
            ("GxG".to_string(), OrderedIntegerSet::from_slice(&[[num_g_partitions, total_num_partitions - 1]]))
        ]),
    )?)
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
    println!("\n\
    => estimating heritability due to G and GxG\n\
    num_people: {}\n\
    num_snps: {}\n\
    number of GxG components: {}",
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
            a[[1 + i, 1 + j]] = estimate_tr_gxg_ki_gxg_kj(
                &le_snps_arr[i],
                &le_snps_arr[j],
                num_random_vecs,
            );
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
pub fn estimate_g_and_multi_gxg_heritability_from_saved_traces(
    geno_bed: &mut PlinkBed,
    mut le_snps_arr: Vec<Array<f32, Ix2>>,
    mut pheno_arr: Array<f32, Ix1>,
    num_random_vecs: usize,
    saved_traces: Array<f64, Ix2>,
) -> Result<(Array<f64, Ix2>, Array<f64, Ix1>, Vec<f64>, Vec<Array<f32, Ix2>>, Array<f32, Ix1>), Error> {
    let (num_people, num_snps) = (geno_bed.num_people, geno_bed.num_snps);
    let num_gxg_components = le_snps_arr.len();
    println!("\n\
    => estimating heritability due to G and GxG\n\
    num_people: {}\n\
    num_snps: {}\n\
    number of GxG components: {}",
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
        let gxg_yky = estimate_gxg_dot_y_norm_sq(
            &normalized_le_snps_arr[i],
            &normalized_pheno_arr,
            num_random_vecs * 50,
        ) / mm;
        b[1 + i] = gxg_yky;
        println!("gxg{}_yky_est: {}", i + 1, gxg_yky);
    }
    b
}

#[deprecated]
pub fn estimate_gxg_heritability(
    gxg_basis_arr: Array<f32, Ix2>,
    mut pheno_arr: Array<f32, Ix1>,
    num_random_vecs: usize,
) -> Result<f64, String> {
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
pub fn estimate_g_and_single_gxg_heritability(
    geno_arr_bed: &mut PlinkBed,
    mut le_snps_arr: Array<f32, Ix2>,
    mut pheno_arr: Array<f32, Ix1>,
    num_random_vecs: usize,
) -> Result<(f64, f64, f64), Error> {
    let mut geno_arr: Array<f32, Ix2> = geno_arr_bed.get_genotype_matrix(None)?;
    let (num_people, num_snps) = geno_arr.dim();
    let num_independent_snps = le_snps_arr.dim().1;
    println!("\n\
    => estimating heritability due to G and GxG\n\
    num_people: {}\n\
    num_snps: {}\n\
    num_independent_snps: {}",
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
    let gxg_yky = estimate_gxg_dot_y_norm_sq(
        &le_snps_arr,
        &pheno_arr,
        num_random_vecs * 50,
    ) / mm;
    println!("gxg_yky: {}", gxg_yky);

    let tr_gk_est = estimate_tr_k_gxg_k(
        geno_arr_bed,
        &le_snps_arr,
        num_random_vecs,
        None,
    );
    println!("tr_gk_est: {}", tr_gk_est);

    let n = num_people as f64;
    let a = array![
        [tr_kk_est, tr_gk_est, n],
        [tr_gk_est, gxg_tr_kk_est, gxg_tr_k_est],
        [n, gxg_tr_k_est, n]
    ];
    let b = array![yky, gxg_yky, yy];
    println!("solving ax=b\na = {:?}\nb = {:?}", a, b);
    let sig_sq = a.solve_into(b).unwrap();

    println!("variance estimates: {:?}", sig_sq);
    Ok((sig_sq[0], sig_sq[1], sig_sq[2]))
}

#[deprecated(note = "use estimate_heritability instead")]
pub fn estimate_heritability_directly(
    mut geno_arr: Array<f32, Ix2>,
    mut pheno_arr: Array<f32, Ix1>,
    num_random_vecs: usize,
) -> Result<f64, String> {
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
