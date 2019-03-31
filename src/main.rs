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
use ndarray::{Array, Ix1, ShapeError};
use ndarray_linalg::Solve;

use bio_file_reader::plink_bed::{MatrixIR, PlinkBed};
use matrix_util::{generate_plus_minus_one_bernoulli_matrix, matrix_ir_to_ndarray, normalize_matrix_row_wise,
                  mean_center_vector};
use program_flow::OrExit;
use stats_util::sum_of_squares;
use timer::Timer;

pub mod histogram;
pub mod matrix_op;
pub mod matrix_util;
pub mod program_flow;
pub mod sparsity_stats;
pub mod timer;
pub mod simulation;
pub mod stats_util;

fn estimate_heritability(genotype_matrix_ir: MatrixIR<u8>, mut pheno_arr: Array<f32, Ix1>, num_random_vecs: usize) -> Result<f64, ShapeError> {
    println!("\n=> creating the genotype ndarray and starting the timer for profiling");
    let mut timer = Timer::new();
    // geno_arr is num_snps x num_people
    let mut geno_arr = matrix_ir_to_ndarray(genotype_matrix_ir)?.mapv(|e| e as f32);
    let (num_rows, num_cols) = geno_arr.dim();
    println!("geno_arr dim: {:?}", geno_arr.dim());
    timer.print();

    println!("\n=> normalizing the genotype matrix row-wise");
    geno_arr = normalize_matrix_row_wise(geno_arr, 1);
    timer.print();

    println!("\n=> mean centering the phenotype vector");
    pheno_arr = mean_center_vector(pheno_arr);
    timer.print();

    println!("\n=> generating random estimators");
    let rand_mat = generate_plus_minus_one_bernoulli_matrix(num_cols, num_random_vecs);
    timer.print();

    println!("\n=> MatMul geno_arr{:?} with rand_mat{:?}", geno_arr.dim(), rand_mat.dim());
    let xz_arr = geno_arr.dot(&rand_mat);
    println!("xz_arr: {:?}", xz_arr.dim());
    timer.print();

    println!("\n=> MatMul geno_arr{:?}.T with xz_arr{:?}", geno_arr.dim(), xz_arr.dim());
    let xxz = geno_arr.t().dot(&xz_arr);
    println!("xxz dim: {:?}", xxz.dim());
    timer.print();

    println!("\n=> calculating trace estimate through L2 squared");
    let trace_kk_est = sum_of_squares(xxz.iter()) / (num_rows * num_rows * num_random_vecs) as f64;
    println!("trace_kk_est: {}", trace_kk_est);
    timer.print();

    println!("\n=> calculating Xy");
    let xy = geno_arr.dot(&pheno_arr);
    timer.print();

    println!("\n=> calculating yKy and yy");
    let yky = sum_of_squares(xy.iter()) / num_rows as f64;
    let yy = sum_of_squares(pheno_arr.iter());
    timer.print();

    println!("\n=> solving for heritability");
    let a = array![[trace_kk_est, (num_cols - 1) as f64],[(num_cols - 1) as f64, num_cols as f64]];
    let b = array![yky, yy];
    println!("solving ax=b\na = {:?}\nb = {:?}", a, b);
    let sig_sq = a.solve_into(b).unwrap();

    println!("sig_sq: {:?}", sig_sq);
    let s_y_sq = yy / (num_cols - 1) as f64;
    let heritability = sig_sq[0] as f64 / s_y_sq;
    println!("heritability: {}  s_y^2: {}", heritability, s_y_sq);

    let standard_error = (2. / (trace_kk_est - num_cols as f64)).sqrt();
    println!("standard error: {}", standard_error);

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

    let mut bed = PlinkBed::new(&plink_bed_filename, &plink_bim_filename, &plink_fam_filename).unwrap_or_exit(None::<String>);

    println!("=> generating the phenotype array and the genotype matrix");
    let pheno_arr = get_pheno_arr(&pheno_filename).unwrap_or_exit(None::<String>);
    let genotype_matrix = bed.get_genotype_matrix().unwrap_or_exit(Some("failed to get the genotype matrix"));
    println!("genotype_matrix dim: {:?}\npheno_arr dim: {:?}", genotype_matrix.dim(), pheno_arr.dim());

    match estimate_heritability(genotype_matrix, pheno_arr, 100) {
        Ok(h) => h,
        Err(why) => {
            eprintln!("{}", why);
            return ();
        }
    };
}

#[cfg(test)]
mod tests {
    extern crate rand;

    use std::collections::HashSet;

    use ndarray::Array;
    use ndarray_rand::RandomExt;
    use rand::distributions::StandardNormal;

    use crate::generate_plus_minus_one_bernoulli_matrix;
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

        let rand_mat = generate_plus_minus_one_bernoulli_matrix(n, num_random_vecs);

        let trace_est = sum_of_squares(x.dot(&rand_mat).iter()) / num_random_vecs as f64;
        println!("trace_est: {}", trace_est);
    }

    #[test]
    fn test_bernoulli_matrix() {
        let n = 1000;
        let num_random_vecs = 100;
        let rand_mat = generate_plus_minus_one_bernoulli_matrix(n, num_random_vecs);
        assert_eq!((n, num_random_vecs), rand_mat.dim());
        let mut value_set = HashSet::<i32>::new();
        for a in rand_mat.iter() {
            value_set.insert(*a as i32);
        }
        // almost certainly this will contain the two values 1 and -1
        assert_eq!(2, value_set.len());
        assert_eq!(true, value_set.contains(&-1));
        assert_eq!(true, value_set.contains(&1));
    }
}
