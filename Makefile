
:set xe

EP=./target/debug/ylang

all: build
	${EP} ./ex/intrinsics.txt
	${EP} ./ex/list.txt
	${EP} ./ex/list_heavy.txt
	${EP} ./ex/simple.txt
	${EP} ./ex/fn.txt
	${EP} ./ex/test.txt

build:
	cargo build
	cargo build --release

