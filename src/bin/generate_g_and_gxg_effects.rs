use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, BufWriter, Write};

use clap::clap_app;
use ndarray::{Array, Ix1, s};

use biofile::plink_bed::PlinkBed;
use saber::program_flow::OrExit;
use saber::simulation::sim_pheno::{generate_g_contribution, generate_gxg_contribution_from_gxg_basis};
use saber::util::{extract_optional_numeric_arg, extract_optional_str_arg, extract_str_arg, get_bed_bim_fam_path};

fn get_le_snp_counts_and_effect_sizes(count_filename: &String) -> Result<Vec<(usize, f64)>, String> {
    let buf = match OpenOptions::new().read(true).open(count_filename.as_str()) {
        Err(why) => return Err(format!("failed to open {}: {}", count_filename, why)),
        Ok(f) => BufReader::new(f)
    };
    let count_vec: Vec<(usize, f64)> = buf.lines().map(|l| {
        let vec2: Vec<String> = l.unwrap()
                                 .split_whitespace()
                                 .map(|t| t.to_string()).collect();
        (vec2[0].parse::<usize>().unwrap(), vec2[1].parse::<f64>().unwrap())
    }
    ).collect();
    Ok(count_vec)
}

fn write_effects_to_file(effects: &Array<f32, Ix1>, out_path: &str) -> Result<(), std::io::Error> {
    let mut buf = BufWriter::new(OpenOptions::new().create(true).truncate(true).write(true).open(out_path)?);
    for val in effects.iter() {
        buf.write_fmt(format_args!("{}\n", val))?;
    }
    Ok(())
}

enum EffectMechanism {
    G,
    // GxG component index
    GxG(usize),
}

fn get_gxg_output_filepath(prefix: &str, effect_mechanism: EffectMechanism) -> String {
    match effect_mechanism {
        EffectMechanism::G => format!("{}.g.effects", prefix),
        EffectMechanism::GxG(component_index) => format!("{}.gxg{}.effects", prefix, component_index)
    }
}

fn main() {
    let app = clap_app!(generate_g_and_gxg_effects =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg bfile: --bfile -b [BFILE] "the PLINK prefix for x.bed, x.bim, x.fam is x")
        (@arg le_snps_bfile: --le [LE_SNPS] "plink file prefix to the SNPs in linkage equilibrium")
        (@arg gxg_component_count_filename: --counts -c [COUNTS]
        "If provided, will generate GxG effects; this is a file where each line is the number of LE SNPs for the corresponding GxG component, a whitespace, and the variance due to that component")
        (@arg g_var: --g [G_VAR] "G variance; If provided, will generate G effects")
        (@arg out_path_prefix: --out -o <OUT> "required; output file path prefix; output will be named OUT.gxg0.effects etc.")
    );
    let matches = app.get_matches();

    let g_var = extract_optional_numeric_arg::<f64>(&matches, "g_var")
        .unwrap_or_exit(None::<String>);
    println!("g_var: {:?}", g_var);

    let out_path_prefix = extract_str_arg(&matches, "out_path_prefix");
    println!("out_path_prefix: {}", out_path_prefix);

    if let Some(g_var) = g_var {
        if g_var > 0. {
            println!("\n=> generating G effects");
            let bfile = extract_optional_str_arg(&matches, "bfile")
                .unwrap_or_exit(Some(format!("must provide --bfile as g_var: {} > 0.", g_var)));
            let [bed_path, bim_path, fam_path] = get_bed_bim_fam_path(&bfile);
            println!("\nPLINK bed path: {}\nPLINK bim path: {}\nPLINK fam path: {}", bed_path, bim_path, fam_path);
            let bed = PlinkBed::new(&bed_path, &bim_path, &fam_path)
                .unwrap_or_exit(None::<String>);
            let geno_arr = bed.get_genotype_matrix(None)
                              .unwrap_or_exit(Some("failed to get the genotype matrix"));

            let out_path = get_gxg_output_filepath(&out_path_prefix, EffectMechanism::G);
            let effects = generate_g_contribution(geno_arr, g_var);
            println!("\n=> writing the effects due to G to {}", out_path);
            write_effects_to_file(&effects, &out_path)
                .unwrap_or_exit(Some(format!("failed to write the simulated effects to file: {}", out_path)));
        }
    }

    if let Some(le_snps_bfile) = extract_optional_str_arg(&matches, "le_snps_bfile") {
        println!("\n=> generating GxG effects");
        let [le_snps_bed_path, le_snps_bim_path, le_snps_fam_path] = get_bed_bim_fam_path(&le_snps_bfile);
        let gxg_component_count_filename = extract_optional_str_arg(&matches, "gxg_component_count_filename")
            .unwrap_or_exit(Some("must provide --counts as le_snps_bfile is specified"));
        println!("\nLE SNPs bed path: {}\nLE SNPs bim path: {}\nLE SNPs fam path: {}", le_snps_bed_path, le_snps_bim_path, le_snps_fam_path);
        println!("gxg_component_count_filename: {}", gxg_component_count_filename);
        let le_snps_bed = PlinkBed::new(&le_snps_bed_path, &le_snps_bim_path, &le_snps_fam_path)
            .unwrap_or_exit(None::<String>);
        let le_snps_arr = le_snps_bed.get_genotype_matrix(None)
                                     .unwrap_or_exit(Some("failed to get the le_snps genotype matrix"));

        let counts_and_effect_sizes = get_le_snp_counts_and_effect_sizes(&gxg_component_count_filename)
            .unwrap_or_exit(Some("failed to get GxG component LE SNP counts and effect sizes"));
        let num_gxg_components = counts_and_effect_sizes.len();

        let mut acc = 0usize;
        for (i, (c, effect_size)) in counts_and_effect_sizes.into_iter().enumerate() {
            println!("GxG component [{}/{}] expects {} LE SNPs", i + 1, num_gxg_components, c);
            let out_path = get_gxg_output_filepath(&out_path_prefix, EffectMechanism::GxG(i + 1));
            let gxg_basis = le_snps_arr.slice(s![..,acc..acc+c]).to_owned();
            let effects = generate_gxg_contribution_from_gxg_basis(gxg_basis, effect_size);
            println!("\n=> writing the effects due to GxG component {} to {}", i + 1, out_path);
            write_effects_to_file(&effects, &out_path)
                .unwrap_or_exit(Some(format!("failed to write the simulated effects to file: {}", out_path)));
            acc += c;
        }
    }
}
