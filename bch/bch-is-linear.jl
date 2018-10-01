#=
 Ir aumentando tamanho do "field" e 
 perceber que tempo é linear com tamanho de bloco
=#

using BenchmarkTools
using PyCall
@pyimport matplotlib.pyplot as plt

include("./decode.jl")


t=3

ms=4:20
tempos = Array{Float64}(undef, length(ms))
calctempo = x -> reduce(+, x.times)/length(x.times)
for m=ms
	field, a = FlintFiniteField(2, m, "a")
	Qx, x = PolynomialRing(field, "x")
	code = BCHCode(a, x, t)
	gen = code.gen

	# polinômio de informação
	info = a + a^2 * x^2 + a^3 * x^3
	erro = 1 + a^4*x + a*x^2

	bench = @benchmarkable decode($gen * $info + $erro, $code)
	bench = run(bench, samples=2)
	# nanossegundos
	tempos[m-ms[1]+1] = calctempo(bench)
end

fig, ax = plt.subplots()
plt.xlim([2^4,2^20])
plt.xlabel.("tamanho de bloco")
plt.ylabel("tempo de decodificação (ns)")
plt.plot(map(i->2^i, ms), tempos)
ax[:legend]()
plt.show()