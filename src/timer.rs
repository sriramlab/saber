use colored::Colorize;
use time::PreciseTime;

fn bold_print(msg: &String) {
    println!("{}", msg.bold());
}

pub struct Timer {
    start_time: PreciseTime,
    last_print_time: PreciseTime,
}

impl Timer {
    pub fn new() -> Timer {
        let now = PreciseTime::now();
        Timer { start_time: now, last_print_time: now }
    }
    pub fn print(&mut self) {
        let now = PreciseTime::now();
        let elapsed = self.last_print_time.to(now);
        let total_elapsed = self.start_time.to(now);
        bold_print(&format!("Timer since last print: {:.3} sec; since creation: {:.3} sec",
                            elapsed.num_milliseconds() as f64 * 1e-3,
                            total_elapsed.num_milliseconds() as f64 * 1e-3));
        self.last_print_time = now;
    }
}
