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

undefined()
