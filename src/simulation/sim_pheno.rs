use ndarray::{Array, Axis, Ix1, Ix2, s};
use ndarray_parallel::prelude::*;
use ndarray_rand::RandomExt;
use rand::distributions::Normal;

use crate::util::matrix_util::normalize_matrix_columns_inplace;
use crate::util::stats_util::{mean, n_choose_2, variance};

/// * `geno_arr` is the 2D genotype array, of shape (num_individuals, num_snps)
/// * `effect_variance` is the variance of the total effect sizes, i.e. each coefficient will have a variance of effect_variance / num_individuals
/// * `noise_variance` is the variance of the noise
pub fn generate_pheno_arr(geno_arr: &Array<f32, Ix2>, effect_variance: f64, noise_variance: f64) -> Array<f32, Ix1> {
    let (num_individuals, num_snps) = geno_arr.dim();
    let effect_size_matrix = Array::random(
        num_snps, Normal::new(0f64, (effect_variance / num_snps as f64).sqrt()))
        .mapv(|e| e as f32);

    let mut noise = Array::random(
        num_individuals, Normal::new(0f64, noise_variance.sqrt()))
        .mapv(|e| e as f32);

    noise -= mean(noise.iter()) as f32;

    println!("beta mean: {}  noise mean: {}", mean(effect_size_matrix.iter()), mean(noise.iter()));
    println!("beta variance: {}  noise variance: {}", variance(effect_size_matrix.iter(), 1),
             variance(noise.iter(), 1));

    geno_arr.dot(&effect_size_matrix) + &noise
}

pub fn generate_g_contribution(mut geno_arr: Array<f32, Ix2>, g_var: f64) -> Array<f32, Ix1> {
    let (num_people, num_snps) = geno_arr.dim();
    println!("\n=> generate_g_contribution\nnum_people: {}\nnum_snps: {}\ng_var: {}",
             num_people, num_snps, g_var);

    println!("\n=> normalizing geno_arr");
    normalize_matrix_columns_inplace(&mut geno_arr, 0);

    println!("\n=> creating G effects");
    let effect_size_matrix = Array::random(
        num_snps, Normal::new(0f64, (g_var / num_snps as f64).sqrt()))
        .mapv(|e| e as f32);
    geno_arr.dot(&effect_size_matrix)
}

/// all the genotype matrix has shape num_people x num_snps
pub fn generate_gxg_pheno_arr(geno_arr: &Array<f32, Ix2>, gxg_arr: &Array<f32, Ix2>,
    g_variance: f64, gxg_variance: f64, noise_variance: f64) -> Array<f32, Ix1> {
    println!("g_variance: {}\ngxg_variance: {}\nnoise_variance: {}", g_variance, gxg_variance, noise_variance);
    let (num_individuals, num_snps) = geno_arr.dim();
    let num_gxg_pairs = gxg_arr.dim().1;

    let g_effect_sizes = Array::random(
        num_snps, Normal::new(0f64, (g_variance / num_snps as f64).sqrt()))
        .mapv(|e| e as f32);

    let gxg_effect_sizes = Array::random(
        num_gxg_pairs, Normal::new(0f64, (gxg_variance / num_gxg_pairs as f64).sqrt()))
        .mapv(|e| e as f32);

    let mut noise = Array::random(
        num_individuals, Normal::new(0f64, noise_variance.sqrt()))
        .mapv(|e| e as f32);
    noise -= mean(noise.iter()) as f32;

    geno_arr.dot(&g_effect_sizes) + gxg_arr.dot(&gxg_effect_sizes) + noise
}

pub fn generate_gxg_contribution_from_gxg_basis(mut gxg_basis: Array<f32, Ix2>, gxg_variance: f64) -> Array<f32, Ix1> {
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
        let mut gxg = gxg_basis.slice(s![.., i+1..]).to_owned();
        gxg.axis_iter_mut(Axis(1))
           .into_par_iter()
           .for_each(|mut col| {
               col *= &snp_i;
           });
        let gxg_effect_sizes = Array::random(gxg.dim().1, Normal::new(0f64, gxg_single_std_dev))
            .mapv(|e| e as f32);
        gxg_effects += &gxg.dot(&gxg_effect_sizes);
    }
    gxg_effects
}

pub fn generate_gxg_pheno_arr_from_gxg_basis(geno_arr: &Array<f32, Ix2>, gxg_basis: &Array<f32, Ix2>,
    g_variance: f64, gxg_variance: f64, noise_variance: f64) -> Array<f32, Ix1> {
    println!("g_variance: {}\ngxg_variance: {}\nnoise_variance: {}", g_variance, gxg_variance, noise_variance);
    let (num_people, num_snps) = geno_arr.dim();

    let num_basis = gxg_basis.dim().1;
    let num_gxg_pairs = num_basis * (num_basis - 1) / 2;

    let g_effect_sizes = Array::random(
        num_snps, Normal::new(0f64, (g_variance / num_snps as f64).sqrt()))
        .mapv(|e| e as f32);

    let gxg_single_std_dev = (gxg_variance / num_gxg_pairs as f64).sqrt();
    let mut gxg_effects = Array::zeros(num_people);
    for i in 0..num_basis - 1 {
        let snp_i = gxg_basis.slice(s![.., i]);
        let mut gxg = gxg_basis.slice(s![.., i+1..]).to_owned();
        gxg.axis_iter_mut(Axis(1))
           .into_par_iter()
           .for_each(|mut col| {
               col *= &snp_i;
           });
        let gxg_effect_sizes = Array::random(gxg.dim().1, Normal::new(0f64, gxg_single_std_dev))
            .mapv(|e| e as f32);
        gxg_effects += &gxg.dot(&gxg_effect_sizes);
    }

    let noise = Array::random(
        num_people, Normal::new(0f64, noise_variance.sqrt()))
        .mapv(|e| e as f32);

    geno_arr.dot(&g_effect_sizes) + gxg_effects + noise
}

#[cfg(test)]
mod tests {
    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use rand::distributions::Uniform;

    use crate::util::stats_util::variance;

    use super::{generate_g_contribution, generate_gxg_contribution_from_gxg_basis};

    #[test]
    fn test_generate_gxg_contribution_from_gxg_basis() {
        let (num_people, num_basis) = (10000, 100);
        let gxg_basis = Array::random((num_people, num_basis), Uniform::from(0..3))
            .mapv(|e| e as f32);
        let desired_variance = 0.05;
        let gxg_effects = generate_gxg_contribution_from_gxg_basis(gxg_basis, desired_variance);
        let actual_variance = variance(gxg_effects.iter(), 0);
        assert!((actual_variance - desired_variance).abs() < 0.01);
    }

    #[test]
    fn test_generate_g_contribution() {
        let (num_people, num_basis) = (10000, 1000);
        let geno_arr = Array::random((num_people, num_basis), Uniform::from(0..3))
            .mapv(|e| e as f32);
        let desired_variance = 0.05;
        let gxg_effects = generate_g_contribution(geno_arr, desired_variance);
        let actual_variance = variance(gxg_effects.iter(), 0);
        assert!((actual_variance - desired_variance).abs() < 0.01);
    }
}
