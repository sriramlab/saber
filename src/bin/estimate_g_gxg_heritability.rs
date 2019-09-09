use biofile::plink_bed::PlinkBed;
use biofile::plink_bim::PlinkBim;
use clap::{Arg, clap_app};
use program_flow::argparse::{
    extract_numeric_arg, extract_optional_str_arg, extract_str_arg, extract_str_vec_arg,
};
use program_flow::OrExit;

use saber::heritability_estimator::estimate_g_gxg_heritability;
use saber::util::get_bed_bim_fam_path;

fn main() {
    let mut app = clap_app!(estimate_multi_gxg_heritability =>
        (version: "0.1")
    );
    app = app
        .arg(
            Arg::with_name("plink_filename_prefix")
                .long("bfile").short("b").takes_value(true).required(true)
                .help(
                    "If we have files named \n\
                    PATH/TO/x.bed PATH/TO/x.bim PATH/TO/x.fam \n\
                    then the <plink_filename_prefix> should be path/to/x"
                )
        )
        .arg(
            Arg::with_name("le_snps_filename_prefix")
                .long("le").takes_value(true).required(true)
                .help(
                    "The SNPs that are in linkage equilibrium.\n\
                    To be used to construct the GxG matrix.\n\
                    If we have files named \n\
                    PATH/TO/x.bed PATH/TO/x.bim PATH/TO/x.fam \n\
                    then the <le_snps_filename_prefix> should be path/to/x"
                )
        )
        .arg(
            Arg::with_name("num_random_vecs")
                .long("nrv").takes_value(true).required(true)
                .help(
                    "The number of random vectors used to estimate traces\n\
                    Recommends at least 100 for small datasets, and 10 for huge datasets"
                )
        )
        .arg(
            Arg::with_name("num_rand_vecs_gxg")
                .long("nrv-gxg").takes_value(true).required(true)
                .help(
                    "The number of random vectors used to estimate traces related to the GxG matrix"
                )
        )
        .arg(
            Arg::with_name("pheno_path")
                .long("pheno").short("p").takes_value(true).required(true)
                .multiple(true).number_of_values(1)
                .help(
                    "The header line should be\n\
                    FID IID PHENOTYPE_NAME\n\
                    where PHENOTYPE_NAME can be any string without white spaces.\n\
                    The rest of the lines are of the form:\n\
                    1000011 1000011 -12.11363"
                )
        )
        .arg(
            Arg::with_name("partition_file").long("partition").takes_value(true)
                                            .help(
                                                "A file to partition the G SNPs into multiple components.\n\
                    Each line consists of two values of the form:\n\
                    SNP_ID PARTITION\n\
                    For example,\n\
                    rs3115860 1\n\
                    will assign SNP with ID rs3115860 in the BIM file to a partition named 1"
                                            )
        )
        .arg(
            Arg::with_name("gxg_partition_file").long("gxg-partition").takes_value(true)
                                                .help(
                                                    "Form GxG for each of the partitions instead of\n\
                    over the entire range of LE SNPs.\n\
                    Taking the same file format as the --partition option"
                                                )
        )
        .arg(
            Arg::with_name("num_jackknife_partitions")
                .long("--num-jackknifes").short("k").takes_value(true).default_value("20")
                .help("The number of jackknife partitions")
        );
    let matches = app.get_matches();

    let plink_filename_prefix = extract_str_arg(&matches, "plink_filename_prefix");
    let le_snps_filename_prefix = extract_str_arg(&matches, "le_snps_filename_prefix");
    let pheno_path_vec = extract_str_vec_arg(&matches, "pheno_path")
        .unwrap_or_exit(None::<String>);
    let num_jackknife_partitions = extract_numeric_arg::<usize>(
        &matches, "num_jackknife_partitions",
    ).unwrap_or_exit(Some(format!("failed to extract num_jackknife_partitions")));

    let [bed_path, bim_path, fam_path] = get_bed_bim_fam_path(&plink_filename_prefix);
    let [le_snps_bed_path, le_snps_bim_path, le_snps_fam_path] = get_bed_bim_fam_path(
        &le_snps_filename_prefix
    );

    let num_random_vecs = extract_str_arg(&matches, "num_random_vecs")
        .parse::<usize>()
        .unwrap_or_exit(Some("failed to parse num_random_vecs"));
    let num_rand_vecs_gxg = extract_str_arg(&matches, "num_rand_vecs_gxg")
        .parse::<usize>()
        .unwrap_or_exit(Some("failed to parse num_rand_vecs_gxg"));
    let g_partition_filepath = extract_optional_str_arg(&matches, "partition_file");
    let gxg_partition_filepath = extract_optional_str_arg(&matches, "gxg_partition_file");

    println!(
        "PLINK bed path: {}\n\
        PLINK bim path: {}\n\
        PLINK fam path: {}",
        bed_path,
        bim_path,
        fam_path
    );
    println!(
        "LE SNPs bed path: {}\n\
        LE SNPs bim path: {}\n\
        LE SNPs fam path: {}",
        le_snps_bed_path,
        le_snps_bim_path,
        le_snps_fam_path
    );
    println!("phenotype paths:");
    for (i, path) in pheno_path_vec.iter().enumerate() {
        println!("[{}/{}] {}", i + 1, pheno_path_vec.len(), path);
    }
    println!("num_random_vecs: {}\nnum_rand_vecs_gxg: {}\nnum_jackknife_partitions: {}",
             num_random_vecs, num_rand_vecs_gxg, num_jackknife_partitions);
    println!(
        "G partition filepath: {}\n\
        gxg_partition_filepath: {}",
        g_partition_filepath.as_ref().unwrap_or(&"".to_string()),
        gxg_partition_filepath.as_ref().unwrap_or(&"".to_string())
    );

    println!("\n=> generating the phenotype array and the genotype matrix");
    let geno_bed = PlinkBed::new(&bed_path, &bim_path, &fam_path)
        .unwrap_or_exit(None::<String>);
    let geno_bim = match &g_partition_filepath {
        Some(p) => PlinkBim::new_with_partition_file(&bim_path, p)
            .unwrap_or_exit(Some(format!(
                "failed to create PlinkBim from bim file: {} and partition file: {}",
                &bim_path, p
            ))),
        None => PlinkBim::new(&bim_path)
            .unwrap_or_exit(Some(format!("failed to create PlinkBim from {}", &bim_path))),
    };

    let le_snps_bed = PlinkBed::new(&le_snps_bed_path, &le_snps_bim_path, &le_snps_fam_path)
        .unwrap_or_exit(None::<String>);
    let le_snps_bim = match &gxg_partition_filepath {
        Some(p) => PlinkBim::new_with_partition_file(&le_snps_bim_path, p)
            .unwrap_or_exit(Some(format!(
                "failed to create PlinkBim from bim file: {} and partition file: {}",
                &le_snps_bim_path, p
            ))),
        None => PlinkBim::new(&le_snps_bim_path)
            .unwrap_or_exit(Some(format!(
                "failed to create PlinkBim for {}", le_snps_bim_path
            ))),
    };
    match estimate_g_gxg_heritability(
        geno_bed,
        geno_bim,
        le_snps_bed,
        le_snps_bim,
        pheno_path_vec.clone(),
        num_random_vecs,
        num_rand_vecs_gxg,
        num_jackknife_partitions,
    ) {
        Err(why) => println!("failed to get heritability estimate: {}", why),
        Ok(est) => {
            for (pheno_index, pheno_path) in pheno_path_vec.iter().enumerate() {
                println!(
                    "\n=> [{}/{}] phenotype {} heritability estimate: {}",
                    pheno_index + 1, pheno_path_vec.len(), pheno_path, est[pheno_path]
                );
            }
        }
    };
}
