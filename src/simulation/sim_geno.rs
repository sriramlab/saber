use ndarray::{Array, Ix2};
use ndarray_rand::RandomExt;
use rand::distributions::WeightedIndex;

/// generate a G matrix with elements drawn independently from {0, 1, 2}
/// `zero_prob` is the probability of an element being 0
/// `one_prob` is the probability of an element being 1
/// `1 - zero_prob - one_prob` is the probability of an element being 1
pub fn generate_g_matrix(
    num_people: usize,
    num_snps: usize,
    zero_prob: f64,
    two_prob: f64,
) -> Result<Array<u8, Ix2>, String> {
    if zero_prob < 0. || two_prob < 0. || zero_prob + two_prob > 1. {
        return Err(format!(
            "invalid probabilities {} {} {}",
            zero_prob,
            1. - zero_prob - two_prob,
            two_prob
        ));
    }
    let weights = [zero_prob, 1. - zero_prob - two_prob, two_prob];
    let dist = WeightedIndex::new(&weights).unwrap();
    Ok(Array::random((num_people, num_snps), dist).mapv(|e| e as u8))
}

/// `geno_arr`: each row is an individual consisting of M snps
/// the returned array will have the same number of rows corresponding to the
/// same indidivuals but each row will consist of M*(M-1)/2 snps formed by g_i *
/// g_j for all i < j
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
