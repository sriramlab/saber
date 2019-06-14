use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};

use clap::{Arg, clap_app};

use bio_file_reader::plink_bed::PlinkBed;
use ndarray::s;
use saber::heritability_estimator::{estimate_g_and_multi_gxg_heritability,
                                    estimate_g_and_multi_gxg_heritability_from_saved_traces};
use saber::program_flow::OrExit;
use saber::util::{extract_optional_str_arg, extract_str_arg, extract_str_vec_arg, get_bed_bim_fam_path, get_pheno_arr,
                  load_trace_estimates, write_trace_estimates};

fn get_le_snp_counts(count_filename: &String) -> Result<Vec<usize>, String> {
    let buf = match OpenOptions::new().read(true).open(count_filename.as_str()) {
        Err(why) => return Err(format!("failed to open {}: {}", count_filename, why)),
        Ok(f) => BufReader::new(f)
    };
    let count_vec: Vec<usize> = buf.lines().map(|l| l.unwrap().parse::<usize>().unwrap()).collect();
    Ok(count_vec)
}

fn main() {
    let mut app = clap_app!(estimate_multi_gxg_heritability =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg bfile: --bfile -b <BFILE> "The PLINK prefix for x.bed, x.bim, x.fam is x; required")
        (@arg le_snps_path: --le <LE_SNPS> "Plink file prefix to the SNPs in linkage equilibrium to construct the GxG matrix; required")
        (@arg gxg_component_count_filename: --counts -c <COUNTS> "A file where each line is the number of LE SNPs for the corresponding GxG component; required")
        (@arg num_random_vecs: --nrv <NUM_RAND_VECS> "Number of random vectors used to estimate traces; required")
    );
    app = app
        .arg(
            Arg::with_name("pheno_path")
                .long("pheno").short("p").takes_value(true).required(true)
                .multiple(true).number_of_values(1)
                .help("Path to the phenotype file. If there are multiple phenotypes, say PHENO1 and PHENO2, \
                pass the paths one by one as follows: -p PHENO1 -p PHENO2")
        )
        .arg(
            Arg::with_name("trace_outpath")
                .long("save-trace").takes_value(true)
                .help("The output path for saving the trace estimates"))
        .arg(
            Arg::with_name("load_trace")
                .long("load-trace").takes_value(true)
                .help("Use the previously saved trace estimates instead of estimating them from scratch")
        );
    let matches = app.get_matches();

    let bfile = extract_str_arg(&matches, "bfile");
    let le_snps_path = extract_str_arg(&matches, "le_snps_path");
    let trace_outpath = extract_optional_str_arg(&matches, "trace_outpath");
    let load_trace = extract_optional_str_arg(&matches, "load_trace");
    let pheno_path_vec = extract_str_vec_arg(&matches, "pheno_path")
        .unwrap_or_exit(None::<String>);

    let [bed_path, bim_path, fam_path] = get_bed_bim_fam_path(&bfile);
    let [le_snps_bed_path, le_snps_bim_path, le_snps_fam_path] = get_bed_bim_fam_path(&le_snps_path);

    let num_random_vecs = extract_str_arg(&matches, "num_random_vecs")
        .parse::<usize>()
        .unwrap_or_exit(Some("failed to parse num_random_vecs"));
    let gxg_component_count_filename = extract_str_arg(&matches, "gxg_component_count_filename");

    println!("PLINK bed path: {}\nPLINK bim path: {}\nPLINK fam path: {}", bed_path, bim_path, fam_path);
    println!("LE SNPs bed path: {}\nLE SNPs bim path: {}\nLE SNPs fam path: {}", le_snps_bed_path, le_snps_bim_path, le_snps_fam_path);
    println!("phenotype paths:");
    for (i, path) in pheno_path_vec.iter().enumerate() {
        println!("[{}/{}] {}", i + 1, pheno_path_vec.len(), path);
    }
    println!("gxg_component_count_filename: {}\nnum_random_vecs: {}", gxg_component_count_filename, num_random_vecs);

    println!("\n=> generating the phenotype array and the genotype matrix");

    let mut bed = PlinkBed::new(&bed_path, &bim_path, &fam_path)
        .unwrap_or_exit(None::<String>);
    let mut geno_arr = bed.get_genotype_matrix()
                          .unwrap_or_exit(Some("failed to get the genotype matrix"));

    let mut le_snps_bed = PlinkBed::new(&le_snps_bed_path, &le_snps_bim_path, &le_snps_fam_path)
        .unwrap_or_exit(None::<String>);
    let le_snps_arr = le_snps_bed.get_genotype_matrix()
                                 .unwrap_or_exit(Some("failed to get the le_snps genotype matrix"));

    let counts = get_le_snp_counts(&gxg_component_count_filename)
        .unwrap_or_exit(Some("failed to get GxG component LE SNP counts"));
    let num_gxg_components = counts.len();

    let mut le_snps_arr_vec = Vec::new();
    let mut acc = 0usize;
    for c in counts.into_iter() {
        println!("GxG component {} expects {} LE SNPs", le_snps_arr_vec.len() + 1, c);
        le_snps_arr_vec.push(le_snps_arr.slice(s![..,acc..acc+c]).to_owned());
        acc += c;
    }

    let mut saved_traces_in_memory = None;
    for (pheno_index, pheno_path) in pheno_path_vec.iter().enumerate() {
        println!("\n=> [{}/{}] estimating the heritability for the phenotype at {}", pheno_index + 1, pheno_path_vec.len(), pheno_path);
        let pheno_arr = get_pheno_arr(pheno_path)
            .unwrap_or_exit(None::<String>);

        let heritability_estimate_result = match saved_traces_in_memory {
            Some(saved_traces) => estimate_g_and_multi_gxg_heritability_from_saved_traces(geno_arr,
                                                                                          le_snps_arr_vec,
                                                                                          pheno_arr,
                                                                                          num_random_vecs,
                                                                                          saved_traces),
            None => {
                match &load_trace {
                    None => estimate_g_and_multi_gxg_heritability(geno_arr,
                                                                  le_snps_arr_vec,
                                                                  pheno_arr,
                                                                  num_random_vecs),

                    Some(load_path) => {
                        let trace_estimates = load_trace_estimates(load_path)
                            .unwrap_or_exit(Some(format!("failed to load the trace estimates from {}", load_path)));
                        let expected_dim = (num_gxg_components + 2, num_gxg_components + 2);
                        assert_eq!(trace_estimates.dim(), expected_dim,
                                   "the loaded trace has dim: {:?} which does not match the expected dimension of {:?}",
                                   trace_estimates.dim(), expected_dim);
                        estimate_g_and_multi_gxg_heritability_from_saved_traces(geno_arr,
                                                                                le_snps_arr_vec,
                                                                                pheno_arr,
                                                                                num_random_vecs,
                                                                                trace_estimates)
                    }
                }
            }
        };

        match heritability_estimate_result {
            Ok((a, _b, h,
                   normalized_geno_arr, normalized_le_snps_arr, _)) => {
                println!("\nvariance estimates on the normalized phenotype at {}:\nG variance: {}", pheno_path, h[0]);
                let mut gxg_var_sum = 0.;
                for i in 1..=num_gxg_components {
                    println!("GxG component {} variance: {}", i, h[i]);
                    gxg_var_sum += h[i];
                }
                println!("noise variance: {}", h[num_gxg_components + 1]);
                println!("total GxG variance: {}", gxg_var_sum);

                // reassign for the remaining phenotypes' heritability estimation
                geno_arr = normalized_geno_arr;
                le_snps_arr_vec = normalized_le_snps_arr;

                // only write the trace out to a file once
                if pheno_index == 0 {
                    if let Some(outpath) = &trace_outpath {
                        println!("\n=> writing the trace estimates to {}", outpath);
                        write_trace_estimates(&a, outpath).unwrap_or_exit(None::<String>);
                    }
                }

                // save the trace to a temporary file for the remaining phenotypes' heritability estimation
                saved_traces_in_memory = Some(a);
            }
            Err(why) => {
                eprintln!("{}", why);
                return ();
            }
        };
    }
}
