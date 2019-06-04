#[macro_use]
extern crate clap;
#[macro_use]
extern crate ndarray;
extern crate saber;

use std::fs::OpenOptions;
use std::io::{BufRead, BufReader};

use bio_file_reader::plink_bed::PlinkBed;
use saber::heritability_estimator::estimate_multi_gxg_heritability;
use saber::program_flow::OrExit;
use saber::util::{extract_str_arg, get_pheno_arr};

fn get_le_snp_counts(count_filename: &String) -> Result<Vec<usize>, String> {
    let buf = match OpenOptions::new().read(true).open(count_filename.as_str()) {
        Err(why) => return Err(format!("failed to open {}: {}", count_filename, why)),
        Ok(f) => BufReader::new(f)
    };
    let count_vec: Vec<usize> = buf.lines().map(|l| l.unwrap().parse::<usize>().unwrap()).collect();
    Ok(count_vec)
}

fn main() {
    let matches = clap_app!(Saber =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg plink_filename_prefix: --bfile <BFILE> "required; the prefix for x.bed, x.bim, x.fam is x")
        (@arg le_snps_filename: --le <LE_SNPS> "required; plink file prefix to the SNPs in linkage equilibrium")
        (@arg pheno_filename: --pheno <PHENO> "required; each row is one individual containing one phenotype value")
        (@arg gxg_component_count_filename: --counts <PHENO> "required; a file where each line is the number of LE SNPs for the corresponding GxG component")
        (@arg num_random_vecs: --nrv +takes_value "number of random vectors used to estimate traces; required")
    ).get_matches();

    let plink_filename_prefix = extract_str_arg(&matches, "plink_filename_prefix");
    let le_snps_filename = extract_str_arg(&matches, "le_snps_filename");
    let pheno_filename = extract_str_arg(&matches, "pheno_filename");

    let plink_bed_path = format!("{}.bed", plink_filename_prefix);
    let plink_bim_path = format!("{}.bim", plink_filename_prefix);
    let plink_fam_path = format!("{}.fam", plink_filename_prefix);

    let le_snps_bed_path = format!("{}.bed", le_snps_filename);
    let le_snps_bim_path = format!("{}.bim", le_snps_filename);
    let le_snps_fam_path = format!("{}.fam", le_snps_filename);

    let num_random_vecs = extract_str_arg(&matches, "num_random_vecs")
        .parse::<usize>()
        .unwrap_or_exit(Some("failed to parse num_random_vecs"));
    let gxg_component_count_filename = extract_str_arg(&matches, "gxg_component_count_filename");

    println!("PLINK bed path: {}\nPLINK bim path: {}\nPLINK fam path: {}",
             plink_bed_path, plink_bim_path, plink_fam_path);
    println!("LE SNPs bed path: {}\nLE SNPs bim path: {}\nLE SNPs fam path: {}",
             le_snps_bed_path, le_snps_bim_path, le_snps_fam_path);
    println!("pheno_filepath: {}\ngxg_component_count_filename: {}\nnum_random_vecs: {}",
             pheno_filename, gxg_component_count_filename, num_random_vecs);

    println!("\n=> generating the phenotype array and the genotype matrix");

    let pheno_arr = get_pheno_arr(&pheno_filename).unwrap_or_exit(None::<String>);

    let mut bed = PlinkBed::new(&plink_bed_path,
                                &plink_bim_path,
                                &plink_fam_path).unwrap_or_exit(None::<String>);
    let geno_arr = bed.get_genotype_matrix().unwrap_or_exit(Some("failed to get the genotype matrix"));

    let mut le_snps_bed = PlinkBed::new(&le_snps_bed_path,
                                        &le_snps_bim_path,
                                        &le_snps_fam_path).unwrap_or_exit(None::<String>);
    let le_snps_arr = le_snps_bed.get_genotype_matrix().unwrap_or_exit(Some("failed to get the le_snps genotype matrix"));
    let counts = get_le_snp_counts(&gxg_component_count_filename).unwrap_or_exit(Some("failed to get GxG component LE SNP counts"));

    let mut le_snps_arr_vec = Vec::new();
    let mut acc = 0usize;
    for c in counts.into_iter() {
        println!("GxG component {} expects {} LE SNPs", le_snps_arr_vec.len() + 1, c);
        le_snps_arr_vec.push(le_snps_arr.slice(s![..,acc..acc+c]).to_owned());
        acc += c;
    }

    match estimate_multi_gxg_heritability(geno_arr,
                                          le_snps_arr_vec,
                                          pheno_arr,
                                          num_random_vecs) {
        Ok(h) => h,
        Err(why) => {
            eprintln!("{}", why);
            return ();
        }
    };
}
