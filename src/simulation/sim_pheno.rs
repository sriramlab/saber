use ndarray::{Array, Ix1, Ix2};
use ndarray_rand::RandomExt;
use rand::distributions::Normal;

use crate::util::stats_util::{mean, variance};

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

pub fn generate_gxg_pheno_arr_from_gxg_basis(geno_arr: &Array<f32, Ix2>, gxg_basis: &Array<f32, Ix2>,
    g_variance: f64, gxg_variance: f64, noise_variance: f64) -> Array<f32, Ix1> {
    println!("g_variance: {}\ngxg_variance: {}\nnoise_variance: {}", g_variance, gxg_variance, noise_variance);
    let (num_people, num_snps) = geno_arr.dim();

    let num_basis = gxg_basis.dim().1;
    let num_gxg_pairs = num_basis * (num_basis - 1) / 2;

    let g_effect_sizes = Array::random(
        num_snps, Normal::new(0f64, (g_variance / num_snps as f64).sqrt()))
        .mapv(|e| e as f32);

    let mut gxg_effects = Array::zeros(num_people);
    for i in 0..num_basis - 1 {
        let snp_i = gxg_basis.slice(s![.., i]);
        let mut gxg = gxg_basis.slice(s![.., i+1..]).clone().to_owned();
        for mut col in gxg.gencolumns_mut() {
            for k in 0..num_people {
                col[k] *= snp_i[k];
            }
        }
        let gxg_effect_sizes = Array::random(
            gxg.dim().1, Normal::new(0f64, (gxg_variance / num_gxg_pairs as f64).sqrt()))
            .mapv(|e| e as f32);
        gxg_effects += &gxg.dot(&gxg_effect_sizes);
    }

    let noise = Array::random(
        num_people, Normal::new(0f64, noise_variance.sqrt()))
        .mapv(|e| e as f32);

    geno_arr.dot(&g_effect_sizes) + gxg_effects + noise
}
