use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, BufWriter, Write};

use clap::{Arg, clap_app};

use bio_file_reader::plink_bed::PlinkBed;
use saber::program_flow::OrExit;
use saber::util::{extract_str_arg, get_bed_bim_fam_path};

const CHUNK_SIZE: usize = 1024;

fn get_chrom_sizes(chrom_size_filepath: &String) -> Result<Vec<usize>, String> {
    let buf = match OpenOptions::new().read(true).open(chrom_size_filepath.as_str()) {
        Err(why) => return Err(format!("failed to open {}: {}", chrom_size_filepath, why)),
        Ok(f) => BufReader::new(f)
    };
    let count_vec: Vec<usize> = buf.lines().map(|l| l.unwrap().parse::<usize>().unwrap()).collect();
    Ok(count_vec)
}

fn main() {
    let mut app = clap_app!(leave_one_chrom_out =>
        (version: "0.1")
        (author: "Aaron Zhou")
        (@arg bfile: --bfile -b <BFILE> "required; the prefix for x.bed, x.bim, x.fam is x")
    );
    app = app.arg(
        Arg::with_name("out_prefix")
            .long("out-prefix")
            .alias("out")
            .short("o")
            .takes_value(true)
            .required(true)
            .help("output path prefix. /PATH/PREFIX will generate files named /PATH/PREFIX_minus_chrom_{CHROM} etc.")
    );

    app = app.arg(
        Arg::with_name("chrom_size_file")
            .long("chrom-size-file")
            .short("c")
            .required(true)
            .takes_value(true)
            .help("the file path to a file containing the chromosome sizes, \
            where the n-th line is the number of SNPs in the n-th chromosome")
    );
    let matches = app.get_matches();

    let bfile = extract_str_arg(&matches, "bfile");
    let chrom_size_filepath = extract_str_arg(&matches, "chrom_size_file");
    let out_path_prefix = extract_str_arg(&matches, "out_prefix");

    let [bed_path, bim_path, fam_path] = get_bed_bim_fam_path(&bfile);
    println!("PLINK bed path: {}\nPLINK bim path: {}\nPLINK fam path: {}\nchrom_size_filepath: {}\nout_path_prefix: {}",
             bed_path, bim_path, fam_path, chrom_size_filepath, out_path_prefix);
    let mut bed = PlinkBed::new(&bed_path, &bim_path, &fam_path)
        .unwrap_or_exit(None::<String>);

    let chrom_sizes = get_chrom_sizes(&chrom_size_filepath).unwrap();

    let num_bytes_per_snp = bed.num_bytes_per_snp;
    let magic_bytes = PlinkBed::get_magic_bytes();
    for index_to_omit in 0..chrom_sizes.len() {
        let out_filepath = format!("{}_minus_chrom_{}", out_path_prefix, index_to_omit + 1);
        println!("=> writing to file: {}", out_filepath);
        let mut buf_writer = BufWriter::new(OpenOptions::new()
            .create(true).truncate(true).write(true)
            .open(&out_filepath)
            .unwrap_or_exit(None::<String>));

        buf_writer.write(&magic_bytes).unwrap_or_exit(Some(format!("failed to write to file {}", out_filepath)));
        let mut current_byte_index = magic_bytes.len();
        for (i, num_snps) in chrom_sizes.iter().enumerate() {
            let num_bytes = num_snps * num_bytes_per_snp;
            for chunk in bed.byte_chunk_iter(current_byte_index,
                                             current_byte_index + num_bytes,
                                             CHUNK_SIZE)
                            .unwrap_or_exit(Some("failed to create byte_chunk_iter")) {
                if i != index_to_omit {
                    buf_writer.write(chunk.as_slice())
                              .unwrap_or_exit(Some(format!("failed to write to file {}", out_filepath)));
                }
            }
            current_byte_index += num_bytes;
        }
    }
}
