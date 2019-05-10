#[macro_use]
extern crate clap;

#[macro_use]
extern crate ndarray;

use clap::ArgMatches;

use saber::program_flow::OrExit;
use saber::simulation::{generate_g_matrix, generate_gxg_pheno_arr, get_gxg_arr, generate_pheno_arr};
use saber::gxg_trace_estimators::{estimate_gxg_gram_trace, estimate_gxg_kk_trace};
use saber::heritability_estimator::{estimate_heritability, estimate_gxg_heritability};
use saber::stats_util::{sum_of_squares, mean};
use bio_file_reader::plink_bed::{MatrixIR, PlinkBed};
use saber::matrix_util::{matrix_ir_to_ndarray, normalize_matrix_row_wise_inplace, mean_center_vector};

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
        (@arg num_people: -r +takes_value "number of rows, i.e. individuals; required")
        (@arg num_snps: -c +takes_value "number of columns, i.e. SNPs; required")
        (@arg g_var: --g +takes_value "G variance; required")
        (@arg gxg_var: --gg +takes_value "GxG variance; required")
        (@arg noise_var: --noise +takes_value "noise variance; required")
        (@arg num_random_vecs: -n +takes_value "number of random vectors; required")
    ).get_matches();

    let num_people = extract_filename_arg(&matches, "num_people")
        .parse::<usize>()
        .unwrap_or_exit(Some("failed to parse num_people"));

    let num_snps = extract_filename_arg(&matches, "num_snps")
        .parse::<usize>()
        .unwrap_or_exit(Some("failed to parse num_snps"));

    let g_var = extract_filename_arg(&matches, "g_var")
        .parse::<f64>()
        .unwrap_or_exit(Some("failed to parse g_var"));

    let gxg_var = extract_filename_arg(&matches, "gxg_var")
        .parse::<f64>()
        .unwrap_or_exit(Some("failed to parse gxg_var"));

    let noise_var = extract_filename_arg(&matches, "noise_var")
        .parse::<f64>()
        .unwrap_or_exit(Some("failed to parse noise_var"));

    let num_random_vecs = extract_filename_arg(&matches, "num_random_vecs")
        .parse::<usize>()
        .unwrap_or_exit(Some("failed to parse num_random_vecs"));

    println!("num_people: {}\nnum_snps: {}\ng_var: {}\ngxg_var: {}\nnoise_var: {}\nnum_random_vecs: {}",
             num_people, num_snps, g_var, gxg_var, noise_var, num_random_vecs);

//    let mut bed = PlinkBed::new(&"../saber-data/nfbc/NFBC_20091001.bed".to_string(),
//                                &"../saber-data/nfbc/NFBC_20091001.bim".to_string(),
//                                &"../saber-data/nfbc/NFBC_20091001.fam".to_string()).unwrap_or_exit(None::<String>);
//
//    let genotype_matrix = bed.get_genotype_matrix().unwrap_or_exit(Some("failed to get the genotype matrix"));
//    let mut g = matrix_ir_to_ndarray(genotype_matrix).unwrap().mapv(|e| e as f32);
//    println!("=> getting slice");
//    g = g.slice(s![..num_snps, ..num_people]).t().to_owned();
    let mut g = generate_g_matrix(num_people, num_snps, 0.64, 0.04).unwrap().mapv(|e| e as f32);


    println!("\n=> creating gxg");
    let mut gxg = get_gxg_arr(&g);

    let unstandardized_gxg = gxg.clone();
//    println!("\n=> normalizing gxg");
//    gxg = normalize_matrix_row_wise_inplace(gxg, 0);
//    println!("=> normalizing g for the gxg heritability estimator");
//    g = normalize_matrix_row_wise_inplace(g.t().to_owned(), 1).t().to_owned();

    println!("some naive computation");
    let x = gxg.clone();
    let num_snp_pairs = x.dim().1;
    let tr_k = sum_of_squares(x.iter()) / num_snp_pairs as f64;
    println!("tr_K: {}", tr_k);
    let k = x.dot(&x.t()) / num_snp_pairs as f32;
    println!("{:?} {:?}", k.dim(), x.dim());
    let tr_kk = sum_of_squares(k.iter());
    println!("tr_KK: {}", tr_kk);

    for iter in 0..8 {
        println!("\n=== ITER: {}", iter);
        println!("\n=> generating phenotypes");
        let mut pheno_arr = generate_pheno_arr(&gxg, gxg_var, noise_var);
        let pheno_mean = mean(pheno_arr.iter());
        println!("pheno_mean: {}", pheno_mean);
        println!("=> centering pheno_arr");
        pheno_arr = mean_center_vector(pheno_arr);

//        let gxg_heritability_est = estimate_gxg_heritability(g.clone(), pheno_arr.clone(), num_random_vecs).unwrap();
//        let gxg_heritability_est = estimate_gxg_heritability(g.clone(), pheno_arr.clone(), num_random_vecs).unwrap();
//        let gxg_heritability_est = estimate_gxg_heritability(g.clone(), pheno_arr.clone(), num_random_vecs).unwrap();

        println!("\n=> doing it naively");
        let yky = pheno_arr.dot(&k.dot(&pheno_arr)) as f64;
        println!("yky: {}", yky);
        let yy = pheno_arr.dot(&pheno_arr) as f64;
        println!("yy: {}", yy);

        use ndarray_linalg::Solve;
        let a = array![[tr_kk, tr_k],[tr_k, num_people as f64]];
        let b = array![yky, yy];
        println!("solving ax=b\na = {:?}\nb = {:?}", a, b);
        let sig_sq = a.solve_into(b).unwrap();
        println!("{:?}", sig_sq);
        let s_y_sq = yy / (num_people - 1) as f64;
        let heritability = sig_sq[0] as f64 / (sig_sq[0] as f64 + sig_sq[1] as f64);
        println!("heritability: {}  s_y^2: {}", heritability, s_y_sq);
    }
}
