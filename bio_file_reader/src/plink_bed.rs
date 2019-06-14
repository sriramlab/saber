use std::{fmt, io};
use std::cmp::min;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Read, Seek, SeekFrom, Write};

use ndarray::{Array, Ix2, ShapeBuilder};

pub enum Error {
    IO { why: String, io_error: io::Error },
    BadFormat(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Error::IO { why, .. } => write!(f, "IO error: {}", why),
            Error::BadFormat(why) => write!(f, "Bad format: {}", why)
        }
    }
}

pub struct PlinkBed {
    bed_buf: BufReader<File>,
    num_people: usize,
    num_snps: usize,
    num_bytes_per_snp: usize,
}

impl PlinkBed {
    fn get_buf(filename: &String) -> Result<BufReader<File>, Error> {
        match OpenOptions::new().read(true).open(filename.as_str()) {
            Err(io_error) => Err(Error::IO { why: format!("failed to open {}: {}", filename, io_error), io_error }),
            Ok(f) => Ok(BufReader::new(f))
        }
    }

    fn get_line_count(filename: &String) -> Result<usize, Error> {
        let fam_buf = PlinkBed::get_buf(filename)?;
        Ok(fam_buf.lines().count())
    }

    fn usize_div_ceil(a: usize, divisor: usize) -> usize {
        a / divisor + (a % divisor != 0) as usize
    }

    pub fn new(bed_filename: &String, bim_filename: &String, fam_filename: &String) -> Result<PlinkBed, Error> {
        let mut bed_buf = PlinkBed::get_buf(bed_filename)?;

        // check if PLINK bed file has the correct file signature
        let expected_bytes = [0x6c_u8, 0x1b_u8, 0x01_u8];
        let mut magic_bytes = [0u8; 3];
        if let Err(io_error) = bed_buf.read_exact(&mut magic_bytes) {
            return Err(Error::IO { why: format!("Failed to read the first three bytes of {}: {}", bed_filename, io_error), io_error });
        }
        if magic_bytes != expected_bytes {
            return Err(Error::BadFormat(
                format!("The first three bytes of the PLINK bed file are supposed to be 0x{:x?}, but found 0x{:x?}",
                        expected_bytes, magic_bytes)
            ));
        }

        let num_people = PlinkBed::get_line_count(fam_filename)?;
        let num_snps = PlinkBed::get_line_count(bim_filename)?;
        let num_bytes_per_snp = num_people / 4 + (num_people % 4 != 0) as usize;
        println!("{} stats:\nnum_snps: {}\nnum_people: {}\nnum_bytes_per_block: {}\n----------",
                 bed_filename, num_snps, num_people, num_bytes_per_snp);

        Ok(PlinkBed { bed_buf, num_people, num_snps, num_bytes_per_snp })
    }

    pub fn reset_bed_buf(&mut self) -> Result<(), io::Error> {
        // the first three bytes are the file signature
        self.bed_buf.seek(SeekFrom::Start(3))?;
        Ok(())
    }

    /// makes the BufReader point to the start of the byte containing the SNP i individual j
    /// 0-indexing
    fn seek_to_byte_containing_snp_i_person_j(&mut self, snp_i: usize, person_j: usize) -> Result<(), io::Error> {
        // the first three bytes are the file signature
        self.bed_buf.seek(SeekFrom::Start((3 + self.num_bytes_per_snp * snp_i + person_j / 4) as u64))?;
        Ok(())
    }

    pub fn get_genotype_matrix(&mut self) -> Result<Array<f32, Ix2>, io::Error> {
        fn lowest_two_bits_to_geno(byte: u8) -> u8 {
            // 00 -> 2 homozygous for the first allele in the .bim file (usually the minor allele)
            // 01 -> 0 missing genotype
            // 10 -> 1 heterozygous
            // 11 -> 0 homozygous for the second allele in the .bim file (usually the major allele)
            let a = (byte & 0b10) >> 1;
            let b = byte & 1;
            (((a | b) ^ 1) << 1) | (a & (!b))
        }
        self.reset_bed_buf()?;

        let last_byte_index = self.num_bytes_per_snp - 1;
        let num_people_last_byte = match self.num_people % 4 {
            0 => 4,
            x => x
        };

        let mut v = Vec::with_capacity(self.num_people * self.num_snps);
        unsafe {
            v.set_len(self.num_people * self.num_snps);
        }
        let mut vi = 0usize;

        let mut snp_bytes = vec![0u8; self.num_bytes_per_snp];
        for _ in 0..self.num_snps {
            self.bed_buf.read_exact(&mut snp_bytes)?;
            for i in 0..last_byte_index {
                v[vi] = lowest_two_bits_to_geno(snp_bytes[i]) as f32;
                v[vi + 1] = lowest_two_bits_to_geno(snp_bytes[i] >> 2) as f32;
                v[vi + 2] = lowest_two_bits_to_geno(snp_bytes[i] >> 4) as f32;
                v[vi + 3] = lowest_two_bits_to_geno(snp_bytes[i] >> 6) as f32;
                vi += 4;
            }
            // last byte
            for k in 0..num_people_last_byte {
                v[vi] = lowest_two_bits_to_geno(snp_bytes[last_byte_index] >> (k << 1)) as f32;
                vi += 1;
            }
        }
        let geno_arr = Array::from_shape_vec((self.num_people, self.num_snps)
                                                 .strides((1, self.num_people)), v).unwrap();
        Ok(geno_arr)
    }

    /// save the transpose of the BED file into `out_path`, which should have an extension of .bedt
    /// wherein the n-th sequence of bytes corresponds to the SNPs for the n-th person
    /// larger values of `snp_byte_chunk_size` lead to faster performance, at the cost of higher memory requirement
    pub fn create_bed_t(&mut self, out_path: &str, snp_byte_chunk_size: usize) -> Result<(), io::Error> {
        let mut buf_writer = BufWriter::new(OpenOptions::new().create(true).truncate(true).write(true).open(out_path)?);
        let num_bytes_per_person = PlinkBed::usize_div_ceil(self.num_snps, 4);

        let people_stride = snp_byte_chunk_size * 4;
        let mut snp_bytes = vec![0u8; people_stride / 4];

        // write people_stride people at a time
        for j in (0..self.num_people).step_by(people_stride) {
            let mut people_buf = vec![vec![0u8; num_bytes_per_person]; people_stride];
            if self.num_people - j < people_stride {
                let remaining_people = self.num_people % people_stride;
                snp_bytes = vec![0u8; PlinkBed::usize_div_ceil(remaining_people, 4)];
            }
            let relative_seek_offset = (self.num_bytes_per_snp - snp_bytes.len()) as i64;
            // read 4 SNPs to the buffers at a time
            self.seek_to_byte_containing_snp_i_person_j(0, j)?;
            for (s4, k) in (0..self.num_snps).step_by(4).enumerate() {
                for (snp_offset, _) in (k..min(k + 4, self.num_snps)).enumerate() {
                    self.bed_buf.read_exact(&mut snp_bytes)?;
                    for w in 0..snp_bytes.len() {
                        people_buf[w + 0][s4] |= (snp_bytes[w] & 0b11) << (snp_offset << 1);
                        people_buf[w + 1][s4] |= ((snp_bytes[w] >> 2) & 0b11) << (snp_offset << 1);
                        people_buf[w + 2][s4] |= ((snp_bytes[w] >> 4) & 0b11) << (snp_offset << 1);
                        people_buf[w + 3][s4] |= ((snp_bytes[w] >> 6) & 0b11) << (snp_offset << 1);
                    }
                    self.bed_buf.seek_relative(relative_seek_offset)?;
                }
            }
            for (p, buf) in people_buf.iter().enumerate() {
                if j + p < self.num_people {
                    buf_writer.write(buf.as_slice())?;
                }
            }
        }
        Ok(())
    }
}
