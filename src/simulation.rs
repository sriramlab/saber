use ndarray::{Array, Ix1, Ix2};
use ndarray::prelude::aview1;
use ndarray_rand::RandomExt;
use rand::distributions::Normal;

/// * `geno_arr` is the 2D genotype array, of shape (num_snps, num_individuals)
/// * `effect_variance` is the variance of the total effect sizes, i.e. each coefficient will have a variance of effect_variance / num_individuals
/// * `noise_variance` is the variance of the noise
pub fn generate_pheno_arr(geno_arr: &Array<f32, Ix2>, effect_variance: f64, noise_variance: f64) -> Array<f32, Ix1> {
    let num_snps = geno_arr.dim().0;
    let num_individuals = geno_arr.dim().1;
    let effect_size_matrix = Array::random(
        (num_snps, 1), Normal::new(0f64, (effect_variance / num_snps as f64).sqrt()))
        .mapv(|e| e as f32);

    let noise = Array::random(
        (num_individuals, 1), Normal::new(0f64, noise_variance.sqrt()))
        .mapv(|e| e as f32);

    let pheno = geno_arr.t().dot(&effect_size_matrix) + noise;
    aview1(pheno.as_slice().unwrap()).into_owned()
}
