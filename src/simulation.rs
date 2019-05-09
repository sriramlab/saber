use ndarray::{Array, Ix1, Ix2};
use ndarray::prelude::aview1;
use ndarray_rand::RandomExt;
use rand::distributions::{Normal, WeightedIndex, Distribution};
use crate::stats_util::{mean, variance};

/// generate a G matrix with elements drawn independently from {0, 1, 2}
/// `zero_prob` is the probability of an element being 0
/// `one_prob` is the probability of an element being 1
/// `1 - zero_prob - one_prob` is the probability of an element being 1
pub fn generate_g_matrix(num_people: usize, num_snps: usize, zero_prob: f64, two_prob: f64) -> Result<Array<u8, Ix2>, String> {
    if zero_prob < 0. || two_prob < 0. || zero_prob + two_prob > 1. {
        return Err(format!("invalid probabilities {} {} {}", zero_prob, 1. - zero_prob - two_prob, two_prob));
    }
    let weights = [zero_prob, 1. - zero_prob - two_prob, two_prob];
    let dist = WeightedIndex::new(&weights).unwrap();
    Ok(Array::random((num_people, num_snps), dist).mapv(|e| e as u8))
}

/// `geno_arr`: each row is an individual consisting of M snps
/// the returned array will have the same number of rows corresponding to the same indidivuals
/// but each row will consist of M*(M-1)/2 snps formed by g_i * g_j for all i < j
pub fn get_gxg_arr(geno_arr: &Array<f32, Ix2>) -> Array<f32, Ix2> {
    let (num_rows, num_cols) = geno_arr.dim();
    let num_cols_gxg = num_cols * (num_cols - 1) / 2;
    let mut gxg = Array::zeros((num_rows, num_cols_gxg));
    let mut k = 0;
    for row in geno_arr.genrows() {
        let mut gxg_col_j = 0usize;
        for i in 0..num_cols {
            for j in i + 1..num_cols {
                gxg[[k, gxg_col_j]] = row[i] * row[j];
                gxg_col_j += 1;
            }
        }
        k += 1;
    }
    gxg
}

/// * `geno_arr` is the 2D genotype array, of shape (num_individuals, num_snps)
/// * `effect_variance` is the variance of the total effect sizes, i.e. each coefficient will have a variance of effect_variance / num_individuals
/// * `noise_variance` is the variance of the noise
pub fn generate_pheno_arr(geno_arr: &Array<f32, Ix2>, effect_variance: f64, noise_variance: f64) -> Array<f32, Ix1> {
    let (num_individuals, num_snps) = geno_arr.dim();
    let effect_size_matrix = Array::random(
        (num_snps), Normal::new(0f64, (effect_variance / num_snps as f64).sqrt()))
        .mapv(|e| e as f32);

    let noise = Array::random(
        (num_individuals), Normal::new(0f64, noise_variance.sqrt()))
        .mapv(|e| e as f32);

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

    let noise = Array::random(
        num_individuals, Normal::new(0f64, noise_variance.sqrt()))
        .mapv(|e| e as f32);

    geno_arr.dot(&g_effect_sizes) + gxg_arr.dot(&gxg_effect_sizes) + noise
}
