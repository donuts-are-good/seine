const BANNER_WIDTH: usize = 84;
const KEY_WIDTH: usize = 14;
const LOGO: &[&str] = &[
    "   _____ ______ _____ _   _ ______ ",
    "  / ____|  ____|_   _| \\ | |  ____|",
    " | (___ | |__    | | |  \\| | |__   ",
    "  \\___ \\|  __|   | | | . ` |  __|  ",
    "  ____) | |____ _| |_| |\\  | |____ ",
    " |_____/|______|_____|_| \\_|______|",
];

pub(super) fn startup_banner(title: &str, subtitle: &str, lines: &[(&str, String)]) {
    let border = "=".repeat(BANNER_WIDTH);
    let divider = "-".repeat(BANNER_WIDTH);
    println!();
    println!("{border}");
    for line in LOGO {
        println!("{:^width$}", line, width = BANNER_WIDTH);
    }
    println!("{divider}");
    println!("{:^width$}", title, width = BANNER_WIDTH);
    println!("{:^width$}", subtitle, width = BANNER_WIDTH);
    println!("{border}");
    for (key, value) in lines {
        println!(
            "  {:<key_width$} {}",
            format!("{key}:"),
            value,
            key_width = KEY_WIDTH
        );
    }
    println!("{border}");
    println!();
}

pub(super) fn info(tag: &str, message: impl AsRef<str>) {
    println!("{} {}", prefix("INFO", tag), message.as_ref());
}

pub(super) fn success(tag: &str, message: impl AsRef<str>) {
    println!("{} {}", prefix(" OK ", tag), message.as_ref());
}

pub(super) fn warn(tag: &str, message: impl AsRef<str>) {
    eprintln!("{} {}", prefix("WARN", tag), message.as_ref());
}

pub(super) fn error(tag: &str, message: impl AsRef<str>) {
    eprintln!("{} {}", prefix("ERR ", tag), message.as_ref());
}

fn prefix(level: &str, tag: &str) -> String {
    format!("[{level}][{tag:<8}]")
}
