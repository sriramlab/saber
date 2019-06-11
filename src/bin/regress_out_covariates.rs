#[macro_use]
extern crate clap;
#[macro_use]
extern crate ndarray;
extern crate saber;

use std::fs::OpenOptions;
use std::io::{BufWriter, Write};

use clap::Arg;
use ndarray_linalg::Solve;

use saber::program_flow::OrExit;
use saber::util::{extract_str_arg, extract_optional_str_vec_arg, get_plink_covariate_arr, get_plink_pheno_data,
                  get_plink_pheno_data_replace_missing_with_mean};
use saber::util::matrix_util::normalize_vector_inplace;

fn main() {
    let mut app = clap_app!(regress_out_covariates =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg covariate_path: --covariate -c <BFILE> "required; covariate PLINK file path")
        (@arg pheno_path: --pheno -p <PHENO> "required; each row has three fields FID IID pheno")
        (@arg out_path: --out -o <PHENO> "required; output file path")
    );
    app = app.arg(
        Arg::with_name("missing_rep")
            .long("miss-coding").short("m").takes_value(true).allow_hyphen_values(true)
            .multiple(true).number_of_values(1)
            .help("Missing value representation. If provided, will replace the missing value with the mean. \
            If there are multiple missing value representations, say REP1 and REP2, pass the representations one by one \
            as follows: -m REP1 -m REP2"));
    let matches = app.get_matches();

    let pheno_path = extract_str_arg(&matches, "pheno_path");
    let covariate_path = extract_str_arg(&matches, "covariate_path");
    let out_path = extract_str_arg(&matches, "out_path");
    let missing_rep = extract_optional_str_vec_arg(&matches, "missing_rep");

    println!("phenotype filepath: {}\ncovariate filepath: {}\noutput filepath: {}",
             pheno_path, covariate_path, out_path);

    println!("\n=> generating the covariate array");
    let cov_arr = get_plink_covariate_arr(&covariate_path)
        .unwrap_or_exit(Some("faile to create the covariate matrix"));
    println!("covariate_arr.dim: {:?}", cov_arr.dim());

    println!("\n=> generating the phenotype array");
    let (header, fid_vec, iid_vec, mut pheno_arr) =
        match missing_rep {
            None => get_plink_pheno_data(&pheno_path)
                .unwrap_or_exit(Some("failed to get the phenotype array"))
            ,
            Some(r) => {
                println!("\nmissing phenotype representation: {:?}", r);
                get_plink_pheno_data_replace_missing_with_mean(&pheno_path, &r)
                    .unwrap_or_exit(Some("failed to get the phenotype array"))
            }
        };
    println!("pheno_arr.dim: {:?}", pheno_arr.dim());

    println!("\n=> normalizing the phenotypes");
    normalize_vector_inplace(&mut pheno_arr, 0);

    println!("\n=> calculating the residual phenotype array");
    let ay = cov_arr.t().dot(&pheno_arr);
    let projection_coefficient = (cov_arr.t().dot(&cov_arr)).solve_into(ay).unwrap();
    let projection = cov_arr.dot(&projection_coefficient);
    let residual = pheno_arr - projection;

    println!("\n=> writing the residual phenotypes to {}", out_path);
    let f = OpenOptions::new().truncate(true).create(true).write(true).open(out_path.as_str())
                              .unwrap_or_exit(Some(format!("failed to create file {}", out_path)));
    let mut buf = BufWriter::new(f);
    buf.write_fmt(format_args!("{}\n", header))
       .unwrap_or_exit(Some("failed to write to the output file"));
    for (i, val) in residual.iter().enumerate() {
        buf.write_fmt(format_args!("{} {} {}\n", fid_vec[i], iid_vec[i], val))
           .unwrap_or_exit(Some("failed to write to the output file"));
    }
}
