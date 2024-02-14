randmst: 
	cargo build --release && cp ./target/release/progset1 ./randmst

test:
	cargo test --release
