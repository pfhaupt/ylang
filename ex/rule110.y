let bfr=32*f(x)0
let step = f(curr) {
    let l = len(curr)
    let next = (0)
    let i = 0
    while {i=i+1} < l-1 next = next + (0,1,1,1,0,1,1,0).{4*curr.{i-1}+2*curr.{i}+curr.{i+1}}
    next + 1
}
let s = 0
while {s=s+1} < len(bfr) {
    print(bfr)
    bfr = step(bfr)
}
