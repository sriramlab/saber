use bio_file_reader::plink_bed::MatrixIR;

use crate::histogram::Histogram;
use crate::stats_util::mean;

pub struct SparsityStats {
    // the percentage of 0's for each row of the matrix
    row_sparsity_vec: Vec<f32>,
}

impl SparsityStats {
    pub fn new(matrix: &MatrixIR<u8>) -> SparsityStats {
        let mut row_sparsity_vec = Vec::new();
        for i in 0..matrix.num_rows {
            let snp_variants = &matrix.data[i * matrix.num_columns..(i + 1) * matrix.num_columns];
            let num_zeros = snp_variants.iter().fold(0, |acc, &a| acc + (a == 0u8) as usize);
            row_sparsity_vec.push(num_zeros as f32 / matrix.num_columns as f32);
        }
        SparsityStats { row_sparsity_vec }
    }

    pub fn histogram(&self, num_intervals: usize) -> Result<Histogram<f32>, String> {
        Histogram::new(&self.row_sparsity_vec, num_intervals, 0., 1.)
    }

    pub fn avg_sparsity(&self) -> f64 {
        mean(self.row_sparsity_vec.iter())
    }
}
