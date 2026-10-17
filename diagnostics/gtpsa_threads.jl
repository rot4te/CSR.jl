# Which TPS operations crash when run from several Julia threads? Usage: julia -t 4 gtpsa_threads.jl <op> [n]
using GTPSA
op = ARGS[1]; n = parse(Int, get(ARGS, 2, "200000"))
d = Descriptor(3, 2); x = @vars(d)
t0 = 0.3 + x[1] + 0.1x[2] * x[3]
fs = Dict("add" => t -> t + t, "mul" => t -> t * t, "div" => t -> t / (1 + t), "divc" => t -> 2.0 / t,
          "sqrt" => sqrt, "log" => log, "sin" => sin, "atan" => t -> atan(t, 1.0 + t), "pow" => t -> t^(1/3),
          "abs" => abs, "exp" => exp)
f = fs[op]
Threads.@threads for i in 1:n
  y = f(t0 + i * 1e-9)
  scalar(y) == 0 && println("zero")
end
println("$op ok on $(Threads.nthreads()) threads")
