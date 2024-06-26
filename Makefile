
:set xe

EP=./target/debug/ylang

all: build

examples: build
	${EP} ./ex/intrinsics.y
	${EP} ./ex/list.y
	${EP} ./ex/list_heavy.y
	${EP} ./ex/simple.y
	${EP} ./ex/fn.y
	${EP} ./ex/prime.y
	${EP} ./ex/use.y
	${EP} ./ex/hello.y
	${EP} ./ex/string.y
	${EP} ./ex/rule110.y

build:
	cargo build
	cargo build --release

