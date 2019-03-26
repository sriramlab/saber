use crate::histogram::Histogram;

pub struct SparsityStats {
    // the percentage of 0's for each row of the matrix
    row_sparsity_vec: Vec<f32>,
}

impl SparsityStats {
    pub fn new(matrix: &Vec<Vec<u8>>) -> SparsityStats {
        let mut row_sparsity_vec = Vec::new();
        for snp_variants in matrix.iter() {
            let num_zeros = snp_variants.iter().fold(0, |acc, &a| acc + (a == 0u8) as usize);
            row_sparsity_vec.push(num_zeros as f32 / snp_variants.len() as f32);
        }
        SparsityStats { row_sparsity_vec }
    }

    pub fn histogram(&self, num_intervals: usize) -> Result<Histogram<f32>, String> {
        Histogram::new(&self.row_sparsity_vec, num_intervals, 0., 1.)
    }

    pub fn avg_sparsity(&self) -> f32 {
        let mut iter = self.row_sparsity_vec.iter();
        let mut current_mean = match iter.next() {
            None => return 0f32,
            Some(a) => *a
        };

        for (i, a) in iter.enumerate() {
            let ratio = i as f32 / (i + 1) as f32;
            current_mean = current_mean * ratio + a / (i + 1) as f32
        }
        current_mean
    }
}
