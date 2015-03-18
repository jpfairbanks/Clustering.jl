using Devectorize
import Base: eltype, show
@doc """a type for the output of eigs

   "eigs" returns the "nev" requested eigenvalues in "d", the
   corresponding Ritz vectors "v" (only if "ritzvec=true"), the
   number of converged eigenvalues "nconv", the number of iterations
   "niter" and the number of matrix vector multiplications
   "nmult", as well as the final residual vector "resid".

""" ->
type EigsResult{T, R}
	values::Array{T,1}
	vectors::Array{T,2}
	nconv::Int64
	niter::Int64
	nmult::Int64
	resid::Array{R,1}
end

function EigsResult(A; kwargs...)
	tupl = eigs(A; kwargs...)
	T = eltype(tupl[1])
	R = eltype(tupl[2])
	EigsResult{T,R}(tupl...)
end

function show(io, er::EigsResult)
	show(io, string(er.values))
	show(io, string(er.vectors))
	show(io, string(er.nconv))
	show(io, string(er.niter))
	show(io, string(er.nmult))
	show(io, string(er.resid))
end
#size(er::EigsResult) = size(er.vectors)
eltype{T,R}(::EigsResult{T,R}) = T
value(er::EigsResult, i::Integer) = er.values[i]
vector(er::EigsResult, i::Integer) = er.vectors[:,i]

import Base.truncate
@doc "truncates an EigsResult to a smaller eigenspace" ->
function truncate(er, n)
	return er.values[1:n], er.vectors[:,1:n]
end

function scale!(a::AbstractVector, b::AbstractVector)
	@devec a[:] = a .* b
	return b
end


@doc "multiply by a matrix using the eigenvectors." ->
function mult!(er::EigsResult, x::Vector, storage::Vector)
	#gemv!(tA, alpha, A, x, beta, y)
	#Update the vector y as alpha*A*x + beta*y or alpha*A'x + beta*y according to tA (transpose A). Returns the updated y.
	#result = er.vectors'*x
	gemv!('T', 1.0, er.vectors, x, 0.0, storage)
	scale!(storage, er.values)
	gemv!('N', 1.0, er.vectors, storage, 0.0, x)
	#result = er.vectors*x
	return x
end

abstract SpectralClusteringResult <: ClusteringResult

type EigsKmeansResult{T,R<:FloatingPoint} <: SpectralClusteringResult
	embedding::EigsResult{T,R}
	clustering::KmeansResult{T}
end
function spectralpartition(similarityMatrix,
 embeddingdim::Integer,
 numclusters::Integer,
 eigstol::FloatingPoint=1e-10;
 kwargs...
)
	#embeddingdim, numclusters, eigstol, kwargs
	er = EigsResult(similarityMatrix, nev=embeddingdim, which=:LR)
	kmr = kmeans(er.vectors[:,2:end]', numclusters; kwargs...)
	return EigsKmeansResult(er, kmr)
end

type EigsRecursiveResult <: SpectralClusteringResult
	assignments::Vector{Int}   # assignments (n)
	counts::Vector{Int}        # number of samples assigned to each cluster (k)
	levels::Int
	totalcost::Float64         # total cost (i.e. objective) (k)
	iterations::Int            # number of elapsed iterations
	converged::Bool            # whether the procedure converged
end

function split!(A, subset::Vector{Int}, assignments::Vector{Int}, threshold::Function, kwargs...)
		subA = A[subset, subset]
		@show size(subA)
		er = embed(subA, kwargs...)
		coordinates = er.vectors[:,2]
		@show n = length(coordinates)
		@show N = length(assignments)
		n == length(subset) || error("subset and coordinates are not the same length")
		(1 <= subset[1]) & (subset[end] <= length(assignments)) || error("subsets will out of bounds into assignments")
		t = threshold(coordinates)
		for i = 1:n
			si = subset[i]
			if coordinates[i] > t
				assignments[si] *= 2
			else
				assignments[si] = 2*assignments[si] + 1
			end
		end
end

function embed(A, kwargs...)
	er = EigsResult(A, nev=2, which=:LR, kwargs...)
	if er.nconv < 2
		warn("Spectral embedding did not converge $kwargs")
	end
	return er
end

function spectralsplit!(A, subset, assignments, minsize, thresholdfunc, kwargs...)
	println("entering: $subset")
	if isempty(subset)
		error("empty subset")
	end
	if length(subset) <= minsize
		error("subset is smaller than minsize")
	end
	er = embed(A, kwargs...)
	coords = er.vectors[:,2]
	split!(A, subset, assignments, thresholdfunc, kwargs...)
	maxlabel = maximum(assignments)
	for l=1:maxlabel
		vl = find(assignments .==l)
		if length(vl) > minsize
			spectralsplit!(A, vl, assignments, minsize, thresholdfunc, kwargs...)
		end
	end
	return assignments
end

type HierarchicalClustering{V<:AbstractArray} <: ClusteringResult
	assignments::V
	nlevels::Int
end

function level!(output::Vector{Int}, hc::HierarchicalClustering, l::Integer)
	assignments = hc.assignments
	@devec output[:] = assignments
	for i in 1 : hc.nlevels-l
		@devec output[:] = floor(output./2)
	end
	return output
end

function spectralsplit(A, minsize, thresholdfunc, kwargs...)
	n = size(A, 1)
	assignments = ones(Int, n)
	initialsubset = [1:n]
	spectralsplit!(A, initialsubset, assignments, minsize, thresholdfunc, kwargs...)
	nlevels = ceil(Int, log2(maximum(assignments)))
	hc = HierarchicalClustering(assignments, nlevels)	
	return hc
end