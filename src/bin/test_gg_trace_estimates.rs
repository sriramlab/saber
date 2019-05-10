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
use saber::simulation::get_gxg_arr;

use saber::trace_estimators::{estimate_gxg_gram_trace, estimate_gxg_kk_trace};

fn extract_filename_arg(matches: &ArgMatches, arg_name: &str) -> String {
    match matches.value_of(arg_name) {
        Some(filename) => filename.to_string(),
        None => {
            eprintln!("the argument {} is required", arg_name);
            std::process::exit(1);
        }
    }
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

    let mut timer = Timer::new();
    println!("\n=> calculating tr_k_true");
    let tr_k_true = (&gxg * &gxg).dot(&Array::<f32, Ix1>::ones(gxg.dim().1)).sum() as f64;
    println!("tr_k_true: {}", tr_k_true);
    timer.print();

    println!("\n=> estimating the trace of GxG.dot(GxG.T)");
    let mut ratio_list = Vec::new();
    for iter in 0..50 {
        let tr_k_est = match estimate_gxg_gram_trace(&geno_arr, num_random_vecs) {
            Ok(t) => t,
            Err(why) => {
                eprintln!("{}", why);
                return;
            }
        };
        let r = (tr_k_est - tr_k_true) / tr_k_true;
        println!("\niter: {} tr_k_est: {}\nk_trace_est error ratio: {:.5}%", iter + 1, tr_k_est, r * 100.);
        ratio_list.push(r.abs());
    }
    timer.print();

    let abs_k_err_avg = ratio_list.iter().sum::<f64>() / ratio_list.len() as f64;
    let abs_k_err_std = (ratio_list.iter().map(|r| (r - abs_k_err_avg) * (r - abs_k_err_avg)).sum::<f64>() / ratio_list.len() as f64).sqrt();
    println!("\nabs_ratio_avg: {}%\nabs_ratio_std: {}%", abs_k_err_avg * 100., abs_k_err_std * 100.);

    println!("\n=> calculating tr_kk_true");
    let k = gxg.dot(&gxg.t());
    println!("k dim: {:?}", k.dim());
    let tr_kk_true = (&k * &k).dot(&Array::<f32, Ix1>::ones(k.dim().1)).sum() as f64;
    println!("tr_kk_true: {}", tr_kk_true);
    timer.print();

    println!("\n=> computing tr_kk_est");
    let mut kk_abs_ratio_list = Vec::new();
    for iter in 0..20 {
        let tr_kk_est = match estimate_gxg_kk_trace(&geno_arr, num_random_vecs) {
            Ok(t) => t,
            Err(why) => {
                eprintln!("{}", why);
                return;
            }
        };
        let kk_ratio = (tr_kk_est - tr_kk_true) / tr_kk_true;
        println!("\niter: {} tr_kk_est: {}\ntr_kk_est error ratio: {:.5}%", iter + 1, tr_kk_est, kk_ratio * 100.);
        kk_abs_ratio_list.push(kk_ratio.abs());
    }
    timer.print();

    let kk_abs_ratio_avg = kk_abs_ratio_list.iter().sum::<f64>() / kk_abs_ratio_list.len() as f64;
    let kk_abs_ratio_std = (kk_abs_ratio_list.iter()
                                             .map(|r| (r - kk_abs_ratio_avg) * (r - kk_abs_ratio_avg))
                                             .sum::<f64>() / kk_abs_ratio_list.len() as f64).sqrt();
    println!("\nkk_abs_ratio_avg: {}%\nkk_abs_ratio_std: {}%", kk_abs_ratio_avg * 100., kk_abs_ratio_std * 100.);
}
