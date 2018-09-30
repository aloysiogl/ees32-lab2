#= Performance de um BCH versus não-codificado =#

include("./decode.jl")
function errorrates(periodicwarning=true)

	# Bits a testar
	N = 1000080

	ps = [0.5, 0.2, 0.1, 5e-2, 2e-2, 1e-2, 5e-3,
	 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5, 2e-5, 1e-5, 5e-6, 2e-6]

	function noise(p::Float64, poly::fq_nmod_poly)
		x = gen(parent(poly))
		for power=0:degree(poly)
			if rand() < p
				poly += x^power
			end
		end
		return poly
	end

	m=5
	n=2^m-1
	t=3
	field, a = FlintFiniteField(2, m, "a")
	Qx, x = PolynomialRing(field, "x")
	code = BCHCode(a, x, t)
	g=code.gen
	k=n-degree(g)


	errorrates::Array{Float64} = []
	for p=ps
		errorcount=0
		if periodicwarning
			print("pre-", p)
		end
		for sampleindex=1:floor(N/k)
			infopoly = reduce((a,b) -> a*x + b, rand([0, 1], k))
			transmittedpoly = g*infopoly
			receivedpoly = noise(p, transmittedpoly)
			decodedpoly = decode(receivedpoly, code)
			errorcoefs = coefficients(infopoly+decodedpoly)
			errorcount += reduce(+, map(
				i -> errorcoefs[i] == 1 ? 1 : 0,
				 range(0, length=k)))
		end
		push!(errorrates, errorcount/N)
	end

	return errorrates
end