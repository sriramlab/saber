use std::{
    collections::{HashMap, HashSet},
    fs::OpenOptions,
    io::{BufWriter, Write},
};

use biofile::{plink_bed::PlinkBed, plink_bim::PlinkBim};
use math::{
    set::{ordered_integer_set::OrderedIntegerSet, traits::Finite},
    stats::{mean, n_choose_2, variance},
};
use ndarray::{s, Array, Axis, Ix1, Ix2, ShapeBuilder};
use ndarray_parallel::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::Normal;
use rayon::prelude::*;

use crate::{
    heritability_estimator::{Coordinate, DEFAULT_PARTITION_NAME},
    util::matrix_util::normalize_matrix_columns_inplace,
};

/// 
/// * `geno_arr` is the 2D genotype array, of shape (num_individuals, num_snps)
/// * `effect_variance` is the variance of the total effect sizes,
/// i.e. each coefficient will have a variance of effect_variance /
/// num_individuals
/// * `noise_variance` is the variance of the noise
pub fn generate_pheno_arr(
    geno_arr: &Array<f32, Ix2>,
    effect_variance: f64,
    noise_variance: f64,
) -> Array<f32, Ix1> {
    let (num_individuals, num_snps) = geno_arr.dim();
    let effect_size_matrix = Array::random(
        num_snps,
        Normal::new(0f64, (effect_variance / num_snps as f64).sqrt()),
    )
    .mapv(|e| e as f32);

    let mut noise = Array::random(
        num_individuals,
        Normal::new(0f64, noise_variance.sqrt()),
    )
    .mapv(|e| e as f32);

    noise -= mean(noise.iter()) as f32;

    println!(
        "beta mean: {}  noise mean: {}",
        mean(effect_size_matrix.iter()),
        mean(noise.iter())
    );
    println!(
        "beta variance: {}  noise variance: {}",
        variance(effect_size_matrix.iter(), 1),
        variance(noise.iter(), 1)
    );

    geno_arr.dot(&effect_size_matrix) + &noise
}

pub fn generate_g_contribution(
    mut geno_arr: Array<f32, Ix2>,
    g_var: f64,
) -> Array<f32, Ix1> {
    let (num_people, num_snps) = geno_arr.dim();
    println!(
        "\n=> generate_g_contribution\nnum_people: {}\nnum_snps: {}\ng_var: {}",
        num_people, num_snps, g_var
    );

    println!("\n=> normalizing geno_arr");
    normalize_matrix_columns_inplace(&mut geno_arr, 0);

    println!("\n=> creating G effects");
    let effect_size_matrix = Array::random(
        num_snps,
        Normal::new(0f64, (g_var / num_snps as f64).sqrt()),
    )
    .mapv(|e| e as f32);
    geno_arr.dot(&effect_size_matrix)
}

pub fn generate_g_contribution_from_bed_bim(
    bed: &PlinkBed,
    bim: &PlinkBim<Coordinate>,
    partition_to_variances: &HashMap<String, Vec<f64>>,
    fill_noise: bool,
    chunk_size: usize,
) -> Result<Array<f32, Ix2>, String> {
    let partitions = bim.get_fileline_partitions_or(
        DEFAULT_PARTITION_NAME,
        OrderedIntegerSet::from_slice(&[[0, bed.total_num_snps() - 1]]),
    );
    let num_people = bed.num_people;
    let num_phenotypes: usize = {
        let s: HashSet<usize> = partition_to_variances
            .values()
            .map(|variances| variances.len())
            .collect();
        if s.len() != 1 {
            return Err(format!(
                "inconsistent number of phenotypes in partition_to_variances: \
                {} different number of variances found",
                s.len()
            ));
        }
        *s.iter().next().unwrap()
    };
    let mut effects: Array<f32, Ix2> = partitions
        .to_hash_map()
        .into_par_iter()
        .fold_with(
            Array::zeros((num_people, num_phenotypes)),
            |acc, (name, partition)| {
                let num_partition_snps = partition.size();
                let single_snp_stds: Vec<f64> = partition_to_variances[&name]
                    .iter()
                    .map(|v| (*v / num_partition_snps as f64).sqrt())
                    .collect();

                bed.col_chunk_iter(chunk_size, Some(partition))
                    .into_par_iter()
                    .fold_with(
                        Array::zeros((num_people, num_phenotypes)),
                        |acc, mut snp_chunk| {
                            normalize_matrix_columns_inplace(&mut snp_chunk, 0);
                            let num_chunk_snps = snp_chunk.dim().1;
                            let effect_size_matrix = Array::from_shape_vec(
                                (num_chunk_snps, num_phenotypes)
                                    .strides((1, num_chunk_snps)),
                                single_snp_stds
                                    .iter()
                                    .flat_map(|s| {
                                        Array::random(
                                            num_chunk_snps,
                                            Normal::new(0f64, *s),
                                        )
                                        .as_slice()
                                        .unwrap()
                                        .to_vec()
                                    })
                                    .collect::<Vec<f64>>(),
                            )
                            .unwrap()
                            .mapv(|e| e as f32);
                            acc + snp_chunk.dot(&effect_size_matrix)
                        },
                    )
                    .reduce(
                        || Array::zeros((num_people, num_phenotypes)),
                        |chunk_acc, chunk_effects| chunk_acc + chunk_effects,
                    )
                    + acc
            },
        )
        .reduce(
            || Array::zeros((num_people, num_phenotypes)),
            |acc, partition_effects| acc + partition_effects,
        );
    if fill_noise {
        let variance_sums: Vec<f64> = partition_to_variances.values().fold(
            vec![0f64; num_phenotypes],
            |mut acc, variances| {
                for (i, v) in variances.iter().enumerate() {
                    acc[i] += *v;
                }
                acc
            },
        );
        let noise = Array::from_shape_vec(
            (num_people, num_phenotypes).strides((1, num_people)),
            variance_sums
                .iter()
                .map(|s| {
                    let noise_var = 1. - *s;
                    if noise_var < 0. {
                        Err(format!(
                            "cannot fill the simulated phenotype with noise when the total variance is larger than 1."
                        ))
                    } else {
                        let noise_std = noise_var.sqrt();
                        Ok(Array::random(num_people, Normal::new(0f64, noise_std))
                            .mapv(|e| e as f32)
                            .as_slice()
                            .unwrap()
                            .to_vec()
                        )
                    }
                })
                .collect::<Result<Vec<Vec<f32>>, String>>()?
                .into_iter()
                .flat_map(|v| v)
                .collect::<Vec<f32>>(),
        ).unwrap();
        effects += &noise;
    }
    Ok(effects)
}

pub fn generate_gxg_contribution_from_gxg_basis(
    mut gxg_basis: Array<f32, Ix2>,
    gxg_variance: f64,
) -> Array<f32, Ix1> {
    let (num_people, num_basis) = gxg_basis.dim();
    let num_gxg_pairs = n_choose_2(num_basis);
    println!("\n=> generate_gxg_contribution_from_gxg_basis\nnum_people: {}\nnum_basis: {}\nequivalent # gxg pairs: {}\ngxg_variance: {}",
             num_people, num_basis, num_gxg_pairs, gxg_variance);

    println!("\n=> normalizing the gxg_basis");
    normalize_matrix_columns_inplace(&mut gxg_basis, 0);

    println!("\n=> creating GxG effects");
    let gxg_single_std_dev = (gxg_variance / num_gxg_pairs as f64).sqrt();
    let mut gxg_effects = Array::zeros(num_people);
    for i in 0..num_basis - 1 {
        let snp_i = gxg_basis.slice(s![.., i]);
        let mut gxg = gxg_basis.slice(s![.., i + 1..]).to_owned();
        gxg.axis_iter_mut(Axis(1))
            .into_par_iter()
            .for_each(|mut col| {
                col *= &snp_i;
            });
        let gxg_effect_sizes =
            Array::random(gxg.dim().1, Normal::new(0f64, gxg_single_std_dev))
                .mapv(|e| e as f32);
        gxg_effects += &gxg.dot(&gxg_effect_sizes);
    }
    gxg_effects
}

pub fn get_sim_output_path(
    prefix: &str,
    effect_mechanism: SimEffectMechanism,
) -> String {
    match effect_mechanism {
        SimEffectMechanism::G => format!("{}.g.effects", prefix),
        SimEffectMechanism::GxG(component_index) => {
            format!("{}.gxg{}.effects", prefix, component_index)
        }
    }
}

pub fn write_effects_to_file(
    effects: &Array<f32, Ix1>,
    fid_iid_list: &Vec<(String, String)>,
    out_path: &str,
) -> Result<(), std::io::Error> {
    assert_eq!(
        effects.dim(),
        fid_iid_list.len(),
        "length of the phenotype array {} not equal to the length of the fid_iid_list {}",
        effects.dim(),
        fid_iid_list.len()
    );
    let mut buf = BufWriter::new(
        OpenOptions::new()
            .create(true)
            .truncate(true)
            .write(true)
            .open(out_path)?,
    );
    buf.write_fmt(format_args!("FID IID pheno\n"))?;
    for (val, (fid, iid)) in effects.iter().zip(fid_iid_list.iter()) {
        buf.write_fmt(format_args!("{} {} {}\n", fid, iid, val))?;
    }
    Ok(())
}

pub enum SimEffectMechanism {
    G,
    // GxG component index
    GxG(usize),
}

#[cfg(test)]
mod tests {
    use math::stats::variance;
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use rand::distributions::Uniform;

    use super::{
        generate_g_contribution, generate_gxg_contribution_from_gxg_basis,
    };

    #[test]
    fn test_generate_gxg_contribution_from_gxg_basis() {
        let (num_people, num_basis) = (10000, 100);
        let gxg_basis =
            Array::random((num_people, num_basis), Uniform::from(0..3))
                .mapv(|e| e as f32);
        let desired_variance = 0.05;
        let gxg_effects = generate_gxg_contribution_from_gxg_basis(
            gxg_basis,
            desired_variance,
        );
        let actual_variance = variance(gxg_effects.iter(), 0);
        assert!((actual_variance - desired_variance).abs() < 0.01);
    }

    #[test]
    fn test_generate_g_contribution() {
        let (num_people, num_basis) = (10000, 1000);
        let geno_arr =
            Array::random((num_people, num_basis), Uniform::from(0..3))
                .mapv(|e| e as f32);
        let desired_variance = 0.05;
        let gxg_effects = generate_g_contribution(geno_arr, desired_variance);
        let actual_variance = variance(gxg_effects.iter(), 0);
        assert!((actual_variance - desired_variance).abs() < 0.01);
    }
}
