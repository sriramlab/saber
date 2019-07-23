use std::fs::OpenOptions;
use std::io::{BufWriter, Write};

use clap::{Arg, clap_app};

use bio_file_reader::interval::traits::{CoalesceIntervals, Interval};
use bio_file_reader::plink_bed::PlinkBed;
use bio_file_reader::plink_bim::PlinkBim;
use bio_file_reader::set::ContiguousIntegerSet;
use saber::program_flow::OrExit;
use saber::util::{extract_str_arg, get_bed_bim_fam_path};

const CHUNK_SIZE: usize = 1024;

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
    let matches = app.get_matches();

    let bfile = extract_str_arg(&matches, "bfile");
    let out_path_prefix = extract_str_arg(&matches, "out_prefix");

    let [bed_path, bim_path, fam_path] = get_bed_bim_fam_path(&bfile);
    println!("PLINK bed path: {}\nPLINK bim path: {}\nPLINK fam path: {}\nout_path_prefix: {}",
             bed_path, bim_path, fam_path, out_path_prefix);
    let mut bed = PlinkBed::new(&bed_path, &bim_path, &fam_path)
        .unwrap_or_exit(None::<String>);

    let mut bim = PlinkBim::new(&bim_path).unwrap_or_exit(Some("failed to create PlinkBim"));
    let chrom_to_fileline_positions = bim.get_chrom_to_fileline_positions()
                                         .unwrap_or_exit(Some("failed to get the file-line position map"));

    let num_bytes_per_snp = bed.num_bytes_per_snp;
    let magic_bytes = PlinkBed::get_magic_bytes();

    let chrom_list: Vec<String> = chrom_to_fileline_positions.keys().map(|k| k.to_string()).collect();
    for excluded_chrom in chrom_list.iter() {
        let out_filepath = format!("{}_minus_chrom_{}", out_path_prefix, excluded_chrom);
        println!("\n=> writing to file: {}", out_filepath);
        let mut buf_writer = BufWriter::new(OpenOptions::new()
            .create(true).truncate(true).write(true)
            .open(&out_filepath)
            .unwrap_or_exit(None::<String>));

        buf_writer.write(&magic_bytes).unwrap_or_exit(Some(format!("failed to write to file {}", out_filepath)));

        let mut snp_intervals: Vec<ContiguousIntegerSet<usize>> = chrom_to_fileline_positions
            .iter()
            .filter(|(chrom, _positions)| *chrom != excluded_chrom)
            // intra-chromosome coalescence
            .flat_map(|(_chrom, positions)| positions.get_intervals_by_ref().to_coalesced_intervals())
            .collect::<Vec<ContiguousIntegerSet<usize>>>();
        // inter-chromosome coalescence
        snp_intervals.coalesce_intervals_inplace();
        println!("Excluding chromosome {}. Using file lines {:?}", excluded_chrom, snp_intervals);

        for interval in snp_intervals.into_iter() {
            let start_byte_index = magic_bytes.len() + num_bytes_per_snp * interval.get_start();
            let end_byte_index = magic_bytes.len() + num_bytes_per_snp * interval.get_end();

            for chunk in bed.byte_chunk_iter(start_byte_index, end_byte_index, CHUNK_SIZE)
                            .unwrap_or_exit(Some("failed to create byte_chunk_iter")) {
                buf_writer.write(chunk.as_slice())
                          .unwrap_or_exit(Some(format!("failed to write to file {}", out_filepath)));
            }
        }
    }
}
