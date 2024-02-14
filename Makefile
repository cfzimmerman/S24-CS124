.PHONY: randmst
.PHONY: test 

# Normal rust installations should just use Cargo. This is for the grading server
randmst: 
	${HOME}/.cargo/bin/cargo build --release && cp ./target/release/progset1 ./randmst

test:
	${HOME}/.cargo/bin/cargo test --release
