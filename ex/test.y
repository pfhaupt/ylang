let if = f(c,t,e)c*t+{c<1}*e
let undefined = f() while 0 {}
let clamp = f(v,mi,ma)if(v>ma,ma,if(v<mi,mi,v))
let and = f(a,b)a*b
let or = f(a,b)clamp(a+b,0,1)
let not = f(b)1-b
let xor = f(a,b)or(and(not(a),b),and(a,not(b)))
let eq = f(a,b)not(or(a>b,a<b))
let isEven = f(x)eq(x/2*2,x)
let isOdd = f(x)not(isEven(x))
let mod = f(a,b)if(a<b,a,a-{a/b*b})
let isPrime = f(n) {
  let c = 3
  let isnt = or(eq(n,2),and(not(eq(mod(n,2),0)),n>1))
  while and(isnt,not(c*c>n)) {
    isnt = mod(n,c)>0
    c = c + 2
  }
  isnt
}
let limit = 10
let listSlow = limit*{f(i)if(isPrime(i),i,0)}/f(n)not(eq(n,0))
let listFast = {limit*f(i)i}/isPrime