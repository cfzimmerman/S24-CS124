.PHONY: all 

# Normal rust installations should just use Cargo. This is for the grading server
all:
	rm -f ./strassen && ${HOME}/.cargo/bin/cargo build --release && cp ./target/release/progset2 ./strassen
