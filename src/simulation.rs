use ndarray::{Array, Ix1, Ix2};
use ndarray_rand::RandomExt;
use rand::distributions::{Normal, StandardNormal};

pub fn generate_random_pheno_vec(num_cols: usize) -> Array<f32, Ix1> {
    Array::random(num_cols, StandardNormal).mapv(|e| e as f32)
}

/// geno_arr is the 2D genotype array, of shape (num_snps, num_individuals)
/// var1 is the variance of the total effect sizes, i.e. each beta_i will have a variance of var1 / num_cols
/// var2 is the variance of the noise
pub fn generate_pheno_arr(geno_arr: &Array<f32, Ix2>, var1: f64, var2: f64) -> Array<f32, Ix2> {
    let num_snps = geno_arr.dim().0;
    let num_individuals = geno_arr.dim().1;
    let effect_size_matrix = Array::random(
        (num_snps, 1), Normal::new(0f64, (var1 / num_snps as f64).sqrt()))
        .mapv(|e| e as f32);

    let noise = Array::random(
        num_individuals, Normal::new(0f64, var2.sqrt()))
        .mapv(|e| e as f32);

    geno_arr.t().dot(&effect_size_matrix) + noise
}
