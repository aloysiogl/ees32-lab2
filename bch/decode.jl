#= Define BCHCode e decode() =#

using Hecke

function Hecke.conjugates(β::fq_nmod)
	conjugates = Set(fq_nmod[])
	while ! (β in conjugates)
	    push!(conjugates, β)
	    β = β^2
	end
	return conjugates
end

function Hecke.minpoly(β::fq_nmod, x::fq_nmod_poly)
	poly = 1
	for conj in conjugates(β)
		poly *= x - conj
	end
	return poly
end

struct BCHCode
	a::fq_nmod
	x::fq_nmod_poly
	t::Int64
	gen::fq_nmod_poly
	BCHCode(a, x, t) = new(
		a,
		x,
		t,
		reduce(lcm, map(i -> minpoly(a^i, x), range(1, length=2*t)))
	)
end

function sindrome(poly::fq_nmod_poly, bch::BCHCode)
	return reduce(+, map(i -> bch.x^i * poly(bch.a^i), range(1, length=2*bch.t)))
end

# Berlekamp-massey algorithm with Chien search at the end
function decode(poly::fq_nmod_poly, bch::BCHCode)
	a = bch.a
	x = bch.x
	t = bch.t
	n::Int64 = order(a.parent)-1

	σ = Array{fq_nmod_poly }(undef, 2*t+1)
	ω = Array{fq_nmod_poly }(undef, 2*t+1)
	τ = Array{fq_nmod_poly }(undef, 2*t+1)
	γ = Array{fq_nmod_poly }(undef, 2*t+1)
	D = Array{Int64}(undef, 2*t+1) 
	B = Array{Int64}(undef, 2*t+1) 
	Δ = Array{fq_nmod}(undef, 2*t+1)

	σ[1] = x^0;
	τ[1] = x^0;
	ω[1] = x^0;
	γ[1] = x^0;
	D[1] = 0;
	B[1] = 0;

	sindpoly = sindrome(poly, bch)

	for k=0:2*t-1
		ki=k+1 # julia 1-based array. algorithm taken from 0-based arrays
	    Δ[ki] = coeff((1+sindpoly) * σ[ki], k+1)
	    σ[ki+1] = σ[ki] - Δ[ki] * x * τ[ki]
	    ω[ki+1] = ω[ki] - Δ[ki] * x * γ[ki]

	    if Δ[ki] == 0 || D[ki] > (k+1)/2 || (Δ[ki] != 0 && D[ki] == (k+1)/2 && B[ki] == 0)
	    	D[ki+1] = D[ki]
			B[ki+1] = B[ki]
			τ[ki+1] = x * τ[ki]
			γ[ki+1] = x * γ[ki]
	    else
	    	D[ki+1] = k + 1 - D[ki]
	    	B[ki+1] = 1 - B[ki]
	    	τ[ki+1] = Δ[ki]^-1 * σ[ki]
	    	γ[ki+1] = Δ[ki]^-1 * ω[ki]
	    end
	end

	decoded = poly
	σ2t = σ[2*t+1]
	maxrootcount = degree(σ2t)
	count = 0
	for power=1:n # n == 2^m - 1
		if σ2t(a^power) == 0
			decoded += x^Int64(n-power)
			count += 1
			if count >= maxrootcount
				break
			end
		end
	end

	return div(decoded, bch.gen)

end