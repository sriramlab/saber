#![feature(unboxed_closures)]
#[macro_use]
extern crate clap;
extern crate colored;
#[macro_use]
extern crate ndarray;
extern crate ndarray_linalg;
extern crate ndarray_rand;
extern crate num_traits;
extern crate rand;
extern crate time;

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};

use clap::ArgMatches;
use ndarray::{Array, Array2, Ix1, ShapeError};
use ndarray::prelude::aview1;
use ndarray_linalg::Solve;
use ndarray_rand::RandomExt;
use rand::distributions::Bernoulli;

use bio_file_reader::plink_bed::{MatrixIR, PlinkBed};
use stats_util::sum_of_squares;
use timer::Timer;

pub mod histogram;
pub mod stats_util;
pub mod sparsity_stats;
pub mod timer;
pub mod simulation;

fn estimate_heritability(genotype_matrix_ir: MatrixIR<u8>, mut pheno_arr: Array<f32, Ix1>, num_random_vecs: usize) -> Result<f64, ShapeError> {
    println!("\n=> creating the genotype ndarray");
    let mut timer = Timer::new();
    // geno_arr is num_snps x num_people
    let mut geno_arr = Array2::from_shape_vec(
        (genotype_matrix_ir.num_rows, genotype_matrix_ir.num_columns),
        genotype_matrix_ir.data)?.mapv(|e| e as f32);
    let num_cols = genotype_matrix_ir.num_columns;
    let num_rows = genotype_matrix_ir.num_rows;
    println!("\n=> geno_arr dim: {:?}", geno_arr.dim());
    timer.print();

    println!("\n=> calculating the mean vector");
    let ones_vec = Array::from_shape_vec(
        (num_cols, 1), vec![1f32; num_cols]).unwrap();
    let mean_vec = geno_arr.dot(&ones_vec) / num_cols as f32;
    println!("mean_vec dim: {:?}", mean_vec.dim());
    timer.print();
    println!("\n=> subtracting the means");
    geno_arr -= &mean_vec;
    timer.print();

    println!("\n=> calculating the standard deviation");
    let mut std_vec = vec![0f32; num_rows];
    let mut i = 0;
    for row in geno_arr.genrows() {
        let row_var = row.iter().fold(0f32, |acc, &a| (a as f32).mul_add(a as f32, acc)) / (num_cols - 1) as f32;
        std_vec[i] = row_var.sqrt();
        i += 1;
    };
    timer.print();
    println!("\n=> dividing by the standard deviation");
    let std_arr = Array::from_shape_vec((num_rows, 1), std_vec).unwrap();
    println!("std_arr dim: {:?}", std_arr.dim());
    geno_arr /= &std_arr;
    timer.print();

    println!("\n=> mean centering the phenotype vector");
    let pheno_mean = pheno_arr.dot(&aview1(ones_vec.as_slice().unwrap())) / pheno_arr.dim() as f32;
    println!("pheno_mean: {}", pheno_mean);
    pheno_arr -= pheno_mean;
    timer.print();

    println!("\n=> generating random estimators");
    let rand_mat = Array::random(
        (num_cols, num_random_vecs),
        Bernoulli::new(0.5)).mapv(|e| (e as i32 * 2 - 1) as f32);
    timer.print();

    println!("\n=> MatMul geno_arr{:?} with rand_mat{:?}", geno_arr.dim(), rand_mat.dim());
    let intermediate_arr = geno_arr.dot(&rand_mat);
    println!("intermediate_arr: {:?}", intermediate_arr.dim());
    timer.print();

    println!("\n=> MatMul geno_arr{:?}.T with intermediate_arr{:?}", geno_arr.dim(), intermediate_arr.dim());
    let xxz = geno_arr.t().dot(&intermediate_arr);
    println!("xxz dim: {:?}", xxz.dim());
    timer.print();

    println!("\n=> calculating trace estimate through L2 squared");
    let trace_est = sum_of_squares(xxz.iter()) / (num_rows * num_rows * num_random_vecs) as f64;
    println!("trace_est: {}", trace_est);
    timer.print();

    println!("\n=> calculating Xy");
    let xy = geno_arr.dot(&pheno_arr);
    timer.print();

    // yky
    println!("\n=> calculating yky");
    let yky = sum_of_squares(xy.iter()) / num_rows as f64;
    timer.print();

    // yy
    println!("\n=> calculating yy");
    let yy = sum_of_squares(pheno_arr.iter());
    timer.print();

    println!("\n=> solving for heritability");
    let a = array![[trace_est, num_cols as f64],[num_cols as f64, num_cols as f64]];
    let b = array![yky, yy];
    println!("solving ax=b\na = {:?}\nb = {:?}", a, b);
    let sig_sq = a.solve_into(b).unwrap();

    println!("sig_sq: {:?}", sig_sq);
    let s_y_sq = yy / (num_cols - 1) as f64;
    let heritability = sig_sq[0] as f64 / s_y_sq;
    println!("heritability: {}  s_y^2: {}", heritability, s_y_sq);

    Ok(heritability)
}

fn extract_filename_arg(matches: &ArgMatches, arg_name: &str) -> String {
    match matches.value_of(arg_name) {
        Some(filename) => filename.to_string(),
        None => {
            eprintln!("the argument {} is required", arg_name);
            std::process::exit(1);
        }
    }
}

fn get_pheno_arr(pheno_filename: &String) -> Result<Array<f32, Ix1>, String> {
    let buf = match OpenOptions::new().read(true).open(pheno_filename.as_str()) {
        Err(why) => return Err(format!("failed to open {}: {}", pheno_filename, why)),
        Ok(f) => BufReader::new(f)
    };
    let pheno_vec: Vec<f32> = buf.lines().map(|l| l.unwrap().parse::<f32>().unwrap()).collect();
    Ok(Array::from_vec(pheno_vec))
}

fn main() {
    let matches = clap_app!(Saber =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg plink_bed_filename: --bed <BED> "required")
        (@arg plink_bim_filename: --bim <BIM> "required")
        (@arg plink_fam_filename: --fam <FAM> "required")
        (@arg pheno_filename: --pheno <PHENO> "required; each row is one individual containing one phenotype value")
    ).get_matches();

    let plink_bed_filename = extract_filename_arg(&matches, "plink_bed_filename");
    let plink_bim_filename = extract_filename_arg(&matches, "plink_bim_filename");
    let plink_fam_filename = extract_filename_arg(&matches, "plink_fam_filename");
    let pheno_filename = extract_filename_arg(&matches, "pheno_filename");

    println!("PLINK bed filename: {}\nPLINK bim filename: {}\nPLINK fam filename: {}\npheno_filename: {}",
             plink_bed_filename, plink_bim_filename, plink_fam_filename, pheno_filename);

    let mut bed = match PlinkBed::new(&plink_bed_filename, &plink_bim_filename, &plink_fam_filename) {
        Err(why) => {
            println!("{}", why);
            std::process::exit(1);
        }
        Ok(bed) => bed
    };
    println!("=> generating pheno_arr");
    let pheno_arr = match get_pheno_arr(&pheno_filename) {
        Err(why) => {
            println!("{}", why);
            std::process::exit(1);
        }
        Ok(arr) => arr
    };

    println!("\n=> generating the genotype matrix");
    let genotype_matrix = match bed.get_genotype_matrix() {
        Err(io_error) => {
            eprintln!("failed to get the genotype matrix: {}", io_error);
            std::process::exit(1);
        }
        Ok(matrix) => matrix
    };
    println!("genotype_matrix.shape: ({}, {})", genotype_matrix.num_rows, genotype_matrix.num_columns);

    match estimate_heritability(genotype_matrix, pheno_arr, 100) {
        Ok(mat) => mat,
        Err(why) => {
            eprintln!("{}", why);
            return ();
        }
    };
}

#[cfg(test)]
mod tests {
    extern crate rand;

    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use rand::distributions::{Bernoulli, StandardNormal};

    use crate::stats_util::sum_of_squares;

    #[test]
    fn test_trace_estimator() {
        let n = 1000;
        let num_random_vecs = 40;
        let x = Array::random(
            (n, n),
            StandardNormal).mapv(|e| e as i32 as f32);
        // want to estimate the trace of x.t().dot(&x)
        let true_trace = sum_of_squares(x.iter());
        println!("true trace: {}", true_trace);

        let rand_mat = Array::random(
            (n, num_random_vecs),
            Bernoulli::new(0.5)).mapv(|e| (e as i32 * 2 - 1) as f32);

        let trace_est = sum_of_squares(x.dot(&rand_mat).iter()) / num_random_vecs as f64;
        println!("trace_est: {}", trace_est);
    }
}
