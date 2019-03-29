#[macro_use]
extern crate clap;
extern crate rand;
extern crate ndarray;
extern crate ndarray_linalg;
extern crate ndarray_rand;
extern crate colored;
//extern crate cblas_sys;

use clap::ArgMatches;
use bio_file_reader::plink_bed::{PlinkBed, MatrixIR};
use sparsity_stats::SparsityStats;
use ndarray::{Array, Array2, ShapeError, ArrayView, Ix};
use ndarray::prelude::*;
use rand::distributions::Bernoulli;
use ndarray_rand::RandomExt;
use std::time::SystemTime;
use colored::Colorize;

pub mod histogram;
mod stats_util;
mod sparsity_stats;

use num_traits::{NumOps, NumAssign};

fn extract_filename_arg(matches: &ArgMatches, arg_name: &str) -> String {
    match matches.value_of(arg_name) {
        Some(filename) => filename.to_string(),
        None => {
            eprintln!("the argument {} is required", arg_name);
            std::process::exit(1);
        }
    }
}

fn bold_print(msg: &String) {
    println!("{}", msg.bold());
}

struct Timer {
    start_time: SystemTime,
    last_print_time: SystemTime,
}

impl Timer {
    fn new() -> Timer {
        Timer { start_time: SystemTime::now(), last_print_time: SystemTime::now() }
    }
    fn print(&mut self) {
        let elapsed = self.last_print_time.elapsed().unwrap();
        let total_elapsed = self.start_time.elapsed().unwrap();

        bold_print(&format!("Timer since last print: {:.3} sec; since creation: {:.3} sec",
                            elapsed.as_secs() as f64 + elapsed.as_millis() as f64 * 1e-3,
                            total_elapsed.as_secs() as f64 + total_elapsed.as_millis() as f64 * 1e-3));
        self.last_print_time = SystemTime::now();
    }
}

fn estimate_trace(genotype_matrix_ir: MatrixIR<u8>, num_random_vecs: usize) -> Result<f64, ShapeError> {
    // geno_arr is num_snps x num_people
    let mut geno_arr = Array2::from_shape_vec(
        (genotype_matrix_ir.num_rows, genotype_matrix_ir.num_columns),
        genotype_matrix_ir.data)?.mapv(|e| e as f32);
    let num_cols = genotype_matrix_ir.num_columns;
    let num_rows = genotype_matrix_ir.num_rows;

    let mut timer = Timer::new();
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
    println!("std_arr.dim: {:?}", std_arr.dim());
    geno_arr /= &std_arr;
    timer.print();

    println!("\n=> generating random estimators");
    let rand_mat = Array::random(
        (num_cols, num_random_vecs),
        Bernoulli::new(0.5)).mapv(|e| e as i32 as f32);
    timer.print();

    println!("\n=> MatMul");
    let intermediate_arr = geno_arr.dot(&rand_mat);
    println!("intermediate_arr: {:?}", intermediate_arr.dim());
    timer.print();

    let xxz = geno_arr.t().dot(&intermediate_arr);
    println!("xxz dim: {:?}", xxz.dim());
    timer.print();

    let l2_sq: f64 = xxz.iter().fold(0f64, |acc, &x| (x as f64).mul_add(x as f64, acc));
    println!("L2 squared: {}", l2_sq);
    timer.print();

    let trace_est = l2_sq / (num_rows * num_rows * num_random_vecs) as f64;
    println!("trace_est: {}", trace_est);
    Ok(trace_est)
}

fn main() {
    let matches = clap_app!(Saber =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg plink_bed_filename: --bed <BED> "required")
        (@arg plink_bim_filename: --bim <BIM> "required")
        (@arg plink_fam_filename: --fam <FAM> "required")
    ).get_matches();

    let plink_bed_filename = extract_filename_arg(&matches, "plink_bed_filename");
    let plink_bim_filename = extract_filename_arg(&matches, "plink_bim_filename");
    let plink_fam_filename = extract_filename_arg(&matches, "plink_fam_filename");

    println!("PLINK bed filename: {}\nPLINK bim filename: {}\nPLINK fam filename: {}",
             plink_bed_filename, plink_bim_filename, plink_fam_filename);

    let mut bed = match PlinkBed::new(&plink_bed_filename, &plink_bim_filename, &plink_fam_filename) {
        Err(why) => {
            println!("{}", why);
            std::process::exit(1);
        }
        Ok(bed) => bed
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

//    let stats = SparsityStats::new(&genotype_matrix);
//    println!("avg sparsity: {}", stats.avg_sparsity());
//
//    match stats.histogram(20usize) {
//        Err(why) => eprintln!("failed to construct the histogram: {}", why),
//        Ok(histogram) => println!("{}", histogram)
//    };

    let mat = match estimate_trace(genotype_matrix, 100) {
        Ok(mat) => mat,
        Err(why) => {
            eprintln!("{}", why);
            return ();
        }
    };
}
