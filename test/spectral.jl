module TestSpectral
using Clustering
import Clustering: spectralsplit, HierarchicalClustering, level!
using Base.Test
type GroundTruth{T} <: ClusteringResult
	nclusters::T
	assignments::Vector{T}
end
nclusters(gt::GroundTruth) = gt.nclusters
assignments(gt::GroundTruth) = gt.assignments

srand(0)
function testclustering(eigskmr, truelabels, q, k)
	@test eigskmr.embedding.nconv >= q
	vi = varinfo(eigskmr.clustering, nclusters(truelabels), assignments(truelabels), :djoint)
	abs(vi)
end

function noisykblock(k, m, n, density, alpha::Float64)
	X = []
	for i=1:k
		x = sprand(m, n, density)
		x.nzval[:] = 1.0
		#x += spdiagm(vec(sum(x,1)))
		push!(X, x)
	end
	M = blkdiag(X...)
	M += 	alpha * sprand(size(M)..., density/4)
	M += M'
	M /=2
	truelabels = GroundTruth(k,vcat([i*Array{Int,1}(ones(n)) for i=1:k]...))
	return M, truelabels
end

m = n = 20
density = 0.3
println("q\tk\ti\talpha\tvidist")
vis = Float64[]
# for k in 2:5
# 	q = k
# 	for i in 1:4
# 		alpha = 0.1 * i /k
# 		M, truelabels = noisykblock(k, m, n, density, alpha)
# 		eigskmr = spectralpartition(M, q, k)
# 		vi = testclustering(eigskmr, truelabels, q, k)
# 		#@show eigskmr.clustering.assignments
# 		println("$q\t$k\t$i\t$alpha\t$vi")
# 		push!(vis, vi)
# 	end
# end
#@test all(vis .< 1e-5)
#end module

k =8
n = 5
m = 5
density = 0.8
M, truelabels = noisykblock(k, m, n, density, 0.1)
assignment = ones(Int, k*n)
@show output = spectralsplit!(M, [1:k*n], assignment, 5, (x)->0.0)
hc = spectralsplit(M, n, (x)->median(x))
@show hc
println("make a wrapper for spectralsplit! that gives a Levels type with methods for querying the assignment at each level.")


assigment = zeros(k*n)
for i=1:hc.nlevels
	@show level!(assignment, hc, i)
end
end