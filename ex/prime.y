use "std"
let isPrime = f(n) {
  let c = 3
  let isnt = or(eq(n,2),and(not(eq(mod(n,2),0)),n>1))
  while and(isnt,not(c*c>n)) {
    isnt = mod(n,c)>0
    c = c + 2
  }
  isnt
}
let limit = 100
let listSlow = limit*{f(i)if(isPrime(i),i,0)}/f(n)not(eq(n,0))
let listFast = {limit*f(i)i}/isPrime
