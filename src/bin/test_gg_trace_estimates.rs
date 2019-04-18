extern crate saber;

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};

#[macro_use]
extern crate clap;

use clap::ArgMatches;
#[cfg(feature = "cuda")]
use co_blas::transpose::Transpose;

#[macro_use]
extern crate ndarray;

use ndarray_rand::RandomExt;
use ndarray::{Array, Ix1, Ix2};
use rand::distributions::Uniform;
#[cfg(feature = "cuda")]
use estimate_heritability_cublas as estimate_heritability;
use saber::program_flow::OrExit;
use saber::timer::Timer;

use saber::gg_estimates::estimate_gxg_trace;

fn extract_filename_arg(matches: &ArgMatches, arg_name: &str) -> String {
    match matches.value_of(arg_name) {
        Some(filename) => filename.to_string(),
        None => {
            eprintln!("the argument {} is required", arg_name);
            std::process::exit(1);
        }
    }
}

/// `geno_arr`: each row is an individual consisting of M snps
/// the returned array will have the same number of rows corresponding to the same indidivuals
/// but each row will consist of M*(M-1)/2 snps formed by g_i * g_j for all i < j
fn get_gxg_arr(geno_arr: &Array<f32, Ix2>) -> Array<f32, Ix2> {
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

fn main() {
    let matches = clap_app!(Saber =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg num_rows: -r +takes_value "number of rows, i.e. individuals; required")
        (@arg num_cols: -c +takes_value "number of columns, i.e. SNPs; required")
        (@arg num_random_vecs: -n +takes_value "number of random vectors; required")
    ).get_matches();
    let num_rows = extract_filename_arg(&matches, "num_rows")
        .parse::<usize>().unwrap_or_exit(Some("failed to parse num_rows"));
    let num_cols = extract_filename_arg(&matches, "num_cols")
        .parse::<usize>().unwrap_or_exit(Some("failed to parse num_cols"));
    let num_random_vecs = extract_filename_arg(&matches, "num_random_vecs")
        .parse::<usize>().unwrap_or_exit(Some("failed to parse num_random_vecs"));
    println!("num_rows: {}\nnum_cols: {}\nnum_random_vecs: {}", num_rows, num_cols, num_random_vecs);

    let geno_arr = Array::random((num_rows, num_cols), Uniform::from(0..3))
        .mapv(|e| e as f32);

    println!("\n=> generating the GxG matrix");
    let gxg = get_gxg_arr(&geno_arr);
    println!("GxG dim: {:?}", gxg.dim());

    let ones = Array::<f32, Ix1>::ones(gxg.dim().1);
    println!("\n=> calculating tr_k_true");
    let tr_k_true = (&gxg * &gxg).dot(&ones).sum() as f64;
    println!("tr_k_true: {}", tr_k_true);

    println!("\n=> estimating the trace of GxG.dot(GxG.T)");
    let mut ratio_list = Vec::new();
    for iter in 0..50 {
        let tr_k_est;
        match estimate_gxg_trace(&geno_arr, num_random_vecs) {
            Ok(t) => tr_k_est = t,
            Err(why) => {
                eprintln!("{}", why);
                return ();
            }
        };
        let r = (tr_k_est - tr_k_true) / tr_k_true;
        println!("\niter: {} tr_k_est: {}\nk_trace_est error ratio: {:.5}", iter + 1, tr_k_est, r);
        ratio_list.push(r);
    }

    let ratio_avg = ratio_list.iter().sum::<f64>() / ratio_list.len() as f64;
    let ratio_std = (ratio_list.iter().map(|r| (r - ratio_avg) * (r - ratio_avg)).sum::<f64>() / ratio_list.len() as f64).sqrt();
    println!("\nratio_avg: {}\nratio_std: {}", ratio_avg, ratio_std);

//    println!("naive trace computation for KK");
//    let k = gxg.dot(&gxg.t());
//    println!("k dim: {:?}", k.dim());
//    let true_trace = sum_of_squares(k.iter());
//    println!("kk true trace: {}", true_trace);
}
