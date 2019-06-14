# Saber

ScAlaBle Estimator Regressor

## Build

#### Installing Rust

`curl https://sh.rustup.rs -sSf | sh`

`rustup toolchain install nightly`

`rustup default nightly`

For more details please visit https://www.rust-lang.org/tools/install and https://github.com/rust-lang/rustup.rs#working-with-nightly-rust

#### Build Saber

`RUSTFLAGS='-L /path/to/OpenBLAS -lopenblas -C target-cpu=native' cargo build --release`

where `/path/to/OpenBLAS` is the path to the directory containing the OpenBLAS libraries.
